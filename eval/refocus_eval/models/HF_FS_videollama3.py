import json
import queue
import random
import shelve
import threading
from functools import partial

import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from decord import VideoReader, cpu
from filelock import FileLock
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from refocus_eval.model_utils.videollama3_resolution_helper import image_resolution
from refocus_eval.model_utils.videoloader import VideoLoader
from transformers import AutoModelForCausalLM, AutoProcessor


def load_video_decord(videoloader, video_path, frame_index):
    if isinstance(video_path, str):
        Vpath = video_path
    else:
        Vpath = video_path[0]

    orig_fps = float(VideoReader(Vpath, ctx=cpu(0)).get_avg_fps())

    spare_frames = videoloader.run(Vpath, frame_index)
    return spare_frames, orig_fps


@register_model("VIDEOLLAMA3_FRAME_SELECTION")
class VIDEOLLAMA3_FRAME_SELECTION(lmms):
    def __init__(
        self,
        pretrained: str,
        frameidx_file="frame_order.db",
        backup_file="rank_state.db",
        lock_file="frame_order.lock",
        max_video_tokens: int = 16384,
        **kwargs,
    ):
        super().__init__()
        self.accelerator = Accelerator()
        if self.accelerator.num_processes > 1:
            self.device = torch.device(f"cuda:{self.accelerator.local_process_index}")
        else:
            self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            trust_remote_code=True,
            dtype=self.dtype,
            attn_implementation="flash_attention_2",
            device_map=self.device,
        ).eval()
        # self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(
            "DAMO-NLP-SG/VideoLLaMA3-2B",
            trust_remote_code=True,
            revision="1de2364338c2ab3b40333202aa5142f5bf8f6845",
        )

        self.backup_file = backup_file
        self.frameidx_file = frameidx_file
        self.lock_file = lock_file
        self.max_video_tokens = max_video_tokens
        self.num_workers = 8

    @property
    def rank(self):
        return self.accelerator.local_process_index

    @property
    def world_size(self):
        return self.accelerator.num_processes

    def _load_frame_set(self):
        with FileLock(self.lock_file):
            with shelve.open(self.frameidx_file) as db:
                return dict(db)

    def _load_last_state(self):
        with FileLock(self.lock_file):
            with shelve.open(self.backup_file) as db:
                return dict(db)

    def _load_last_state_keys(self):
        with FileLock(self.lock_file):
            with shelve.open(self.backup_file) as db:
                return set(db.keys())

    def _save_last_state(self, cache_dict):
        with FileLock(self.lock_file):
            with shelve.open(self.backup_file, writeback=False) as db:
                for key, value in cache_dict.items():
                    db[key] = value

    def loglikelihood(self, requests) -> list[str]:
        raise NotImplementedError("TODO: Implement generation for loglikelihood")

    def _preprocessing_worker(
        self, requests, order, cached_keys, frame_sets, task_dict, q: queue.Queue
    ):
        skip_count = 0
        vl_cache: dict[int, VideoLoader] = {}

        for idx in order:
            context, gen_kwargs, doc_to_visual, doc_id, task, split = requests[idx].args
            sample = task_dict[task][split][doc_id]
            sample.pop("image", None)
            req_key = json.dumps((context, sample))

            if req_key in cached_keys:
                skip_count += 1
                continue

            frame_set = frame_sets[req_key]
            nframes = len(frame_set)

            if nframes not in vl_cache:
                resolver_n = partial(
                    image_resolution, max_tokens=self.max_video_tokens // nframes
                )
                vl_cache[nframes] = VideoLoader(
                    resolver_n, num_workers=self.num_workers
                )

            videoloader = vl_cache[nframes]
            video_path = doc_to_visual(sample)
            frames_np, fps = load_video_decord(videoloader, video_path, frame_set)

            cached_keys.add(req_key)
            q.put((skip_count, idx, req_key, frames_np, frame_set, fps))
            skip_count = 0

        q.put((skip_count, None, None, None, None, None))

    @torch.inference_mode()
    def generate_until(self, requests):
        cache = {}
        cached_keys = self._load_last_state_keys()
        frame_sets = self._load_frame_set()
        q = queue.Queue(maxsize=1)

        order = list(range(len(requests)))
        random.shuffle(order)

        producer = threading.Thread(
            target=self._preprocessing_worker,
            args=(requests, order, cached_keys, frame_sets, self.task_dict, q),
        )
        producer.start()

        pbar = tqdm.tqdm(
            total=len(order),
            disable=(self.rank != 0),
            desc=f"Rank{self.rank} responding",
            mininterval=10,
        )

        while True:
            skip_count, idx, req_key, frames_np, frames_indices, video_fps = q.get()
            if idx is None:
                break
            pbar.update(skip_count + 1)

            context, gen_kwargs, doc_to_visual, doc_id, task, split = requests[idx].args
            generation_config = dict(
                max_new_tokens=gen_kwargs.get("max_new_tokens", 128),
                temperature=gen_kwargs.get("temperature", 0.0),
                top_p=gen_kwargs.get("top_p", None),
                num_beams=gen_kwargs.get("num_beams", 1),
                do_sample=True if gen_kwargs.get("temperature", 0.0) > 0 else False,
            )

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": frames_np,
                            "num_frames": int(frames_np.shape[0]),
                            "timestamps": np.asarray(frames_indices, dtype=float)
                            / video_fps,
                        },
                        {"type": "text", "text": context},
                    ],
                }
            ]

            inputs = self.processor(conversation=conversation, return_tensors="pt")
            inputs = {
                k: (
                    (
                        v.to(self.device, self.dtype)
                        if k == "pixel_values"
                        else v.to(self.device)
                    )
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in inputs.items()
            }

            output_ids = self.model.generate(**inputs, **generation_config)
            texts = self.processor.batch_decode(
                output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = texts[0] if isinstance(texts, list) else texts

            cache[req_key] = response
            if len(cache) % 8 == 0:
                self._save_last_state(cache)
                cache.clear()
                torch.cuda.empty_cache()

        if cache:
            self._save_last_state(cache)

        pbar.close()
        producer.join()

        response_db = self._load_last_state()
        results = []
        for req in requests:
            context, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
            sample = self.task_dict[task][split][doc_id]
            sample.pop("image", None)
            req_key = json.dumps((context, sample))
            results.append(response_db[req_key])

        return results

    def generate_until_multi_round(self, requests) -> list[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
