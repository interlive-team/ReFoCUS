import json
import queue
import random
import shelve
import threading

import torch
import torchvision.transforms.functional as VF
import tqdm
from accelerate import Accelerator, DistributedType
from filelock import FileLock
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from refocus_eval.model_utils.videoloader import VideoLoader
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

fused_mean = [m * 255 for m in IMAGENET_MEAN]
fused_std = [s * 255 for s in IMAGENET_STD]


def load_video_decord(videoloader, video_path, frame_index):
    if isinstance(video_path, str):
        Vpath = video_path
    else:
        Vpath = video_path[0]

    spare_frames = videoloader.run(Vpath, frame_index)
    return spare_frames


@register_model("InternVL2_5_FRAME_SELECTION")
class InternVL2_5_FRAME_SELECTION(lmms):
    def __init__(
        self,
        pretrained: str,
        frameidx_file="frame_order.db",
        backup_file="rank_state.db",
        lock_file="frame_order.lock",
        **kwargs,
    ):
        super().__init__()
        self.accelerator = Accelerator()
        if self.accelerator.num_processes > 1:
            self.device = torch.device(f"cuda:{self.accelerator.local_process_index}")
        else:
            self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

        self.model = AutoModel.from_pretrained(
            pretrained,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=self.device,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=True
        )

        if self.accelerator.num_processes > 1:
            distributed_list = [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ]
            assert (
                self.accelerator.distributed_type in distributed_list
            ), "Unsupported distributed type. Only DDP/FSDP/DS are supported."
            if self.accelerator.distributed_type == DistributedType.FSDP:
                self.model = self.accelerator.prepare(self.model)
            else:
                self.model = self.accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
        else:
            self.model.to(self.device)

        self.backup_file = backup_file
        self.frameidx_file = frameidx_file
        self.lock_file = lock_file

        input_size = self.model.config.force_image_size
        self.videoloader = VideoLoader((input_size, input_size), num_workers=4)

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

        for idx in order:
            context, gen_kwargs, doc_to_visual, doc_id, task, split = requests[idx].args
            sample = task_dict[task][split][doc_id]
            sample.pop("image", None)
            req_key = json.dumps((context, sample))

            if req_key in cached_keys:
                skip_count += 1
                continue

            frame_set = frame_sets[req_key]
            video_path = doc_to_visual(sample)
            frames_np = load_video_decord(self.videoloader, video_path, frame_set)

            cached_keys.add(req_key)
            q.put((skip_count, idx, req_key, frames_np))
            skip_count = 0

        q.put((skip_count, None, None, None))

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
            skip_count, idx, req_key, frames_np = q.get()
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
                eos_token_id=151645,
                pad_token_id=151645,
            )

            # process model here
            pixel_values = (
                torch.from_numpy(frames_np)
                .to(dtype=self.dtype, device=self.device)
                .permute(0, 3, 1, 2)
            )
            VF.normalize(pixel_values, mean=fused_mean, std=fused_std, inplace=True)
            num_patches_list = [1] * len(pixel_values)
            video_prefix = "".join(
                [
                    f"Frame{i}: <image>\n"
                    for i, _ in enumerate(num_patches_list, start=1)
                ]
            )
            question = video_prefix + context
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
            )
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
