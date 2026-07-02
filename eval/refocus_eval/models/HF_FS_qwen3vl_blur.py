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
from refocus_eval.model_utils.gaussian_blur import apply_gaussian_blur
from refocus_eval.model_utils.qwen3vl_resolution_helper import video_resolution
from refocus_eval.model_utils.videoloader import VideoLoader
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def load_video_decord(videoloader, video_path, frame_index):
    if isinstance(video_path, str):
        Vpath = video_path
    else:
        Vpath = video_path[0]

    orig_fps = float(VideoReader(Vpath, ctx=cpu(0)).get_avg_fps())

    spare_frames = videoloader.run(Vpath, frame_index)
    return spare_frames, orig_fps


@register_model("QWEN3_VL_FRAME_SELECTION_BLUR")
class QWEN3_VL_FRAME_SELECTION_BLUR(lmms):
    def __init__(
        self,
        pretrained: str,
        frameidx_file="frame_order.db",
        backup_file="rank_state.db",
        lock_file="frame_order.lock",
        image_patch_size: int = 16,
        spatial_merge_size: int = 2,
        video_min_token_num: int = 128,
        video_max_token_num: int = 768,
        frame_factor: int = 2,
        model_seq_len: int = 32000,
        max_ratio: int = 200,
        blur_sigma: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.accelerator = Accelerator()
        if self.accelerator.num_processes > 1:
            self.device = torch.device(f"cuda:{self.accelerator.local_process_index}")
        else:
            self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained,
            dtype=self.dtype,
            attn_implementation="flash_attention_2",
            device_map=self.device,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained)

        self.backup_file = backup_file
        self.frameidx_file = frameidx_file
        self.lock_file = lock_file

        self._resolution_kwargs = dict(
            image_patch_size=image_patch_size,
            spatial_merge_size=spatial_merge_size,
            video_min_token_num=video_min_token_num,
            video_max_token_num=video_max_token_num,
            frame_factor=frame_factor,
            model_seq_len=model_seq_len,
            max_ratio=max_ratio,
        )
        self.num_workers = 8
        self.blur_sigma = blur_sigma

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

    def _apply_gaussian_blur_inplace(self, frames_np: np.ndarray, blur_flags):
        """
        frames_np: (T, H, W, C) uint8
        blur_flags: List[bool] length T
        """
        frames_np[blur_flags] = apply_gaussian_blur(
            frames=frames_np[blur_flags],
            sigma=(self.blur_sigma, self.blur_sigma),
        )

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
            indices = frame_set["indices"]
            blur_flags = frame_set["blur"]
            nframes = len(indices)
            assert len(blur_flags) == len(indices)
            if nframes not in vl_cache:
                resolver_n = partial(
                    video_resolution, nframes, **self._resolution_kwargs
                )
                vl_cache[nframes] = VideoLoader(
                    resolver_n, num_workers=self.num_workers
                )

            videoloader = vl_cache[nframes]
            video_path = doc_to_visual(sample)
            frames_np, fps = load_video_decord(videoloader, video_path, indices)
            self._apply_gaussian_blur_inplace(frames_np, blur_flags)

            cached_keys.add(req_key)
            q.put((skip_count, idx, req_key, frames_np, indices, fps))
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

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames_np},
                        {"type": "text", "text": context},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                videos_kwargs={
                    "do_sample_frames": False,
                    "video_metadata": {
                        "fps": video_fps,
                        "frames_indices": frames_indices,
                        "total_num_frames": int(frames_np.shape[0]),
                    },
                },
            )

            inputs = inputs.to(self.device)

            generated_ids = self.model.generate(**inputs, **generation_config)
            trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            texts = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = texts[0] if isinstance(texts, list) else texts

            cache[req_key] = response
            if len(cache) % 8 == 0:
                self._save_last_state(cache)
                cache.clear()

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
