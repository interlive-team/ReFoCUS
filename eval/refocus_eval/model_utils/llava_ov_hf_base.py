"""HuggingFace-native LLaVA-OneVision backend for ReFoCUS frame-selection evaluation.

Replaces the original LLaVA-NeXT (`llava.model.builder.load_pretrained_model`) path with
the official `transformers` implementation, so no LLaVA-NeXT clone is required.

Use the HF-format checkpoints, e.g.:
    llava-hf/llava-onevision-qwen2-0.5b-ov-hf
    llava-hf/llava-onevision-qwen2-7b-ov-hf

The class consumes a precomputed frame-index DB (shelve) keyed by json.dumps((context, sample)),
exactly like the other HF_FS_* models, and writes per-sample responses to a resumable backup DB.
"""

import json
import queue
import random
import re
import shelve
import threading

import torch
import tqdm
from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from filelock import FileLock
from lmms_eval.api.model import lmms
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

IMAGE_PLACEHOLDERS = re.compile(r"<image>|<video>|<image>\n")


def load_frames_native(video_path, frame_index):
    """Decode the selected frames at their native resolution and return them as a
    (T, H, W, 3) uint8 RGB array. Resizing/normalization is left to the HF processor."""
    vpath = video_path if isinstance(video_path, str) else video_path[0]
    vr = VideoReader(vpath, ctx=cpu(0))
    idx = [int(i) for i in frame_index]
    return vr.get_batch(idx).asnumpy()  # (T, H, W, 3) uint8 RGB


class LlavaOVHFBase(lmms):
    """Shared machinery; subclasses implement `_frames_for(video_path, frame_set)`."""

    def __init__(
        self,
        pretrained: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        frameidx_file: str = "frame_order.db",
        backup_file: str = "rank_state.db",
        lock_file: str = "frame_order.lock",
        # accepted for CLI compatibility with the original model_args; unused by the HF path:
        conv_template: str | None = None,
        model_name: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.accelerator = Accelerator()
        if self.accelerator.num_processes > 1:
            self.device = torch.device(f"cuda:{self.accelerator.local_process_index}")
        else:
            self.device = torch.device("cuda")
        self.dtype = torch.float16

        self.processor = AutoProcessor.from_pretrained(pretrained)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            pretrained, torch_dtype=self.dtype, device_map=self.device
        )
        # The original LLaVA-OneVision projector was trained on the RAW penultimate SigLIP
        # encoder output. In transformers >=4.5x the vision tower's hidden_states[-1] returned
        # to LlavaOnevision is post_layernorm-ed (the output-capture machinery overwrites the
        # last entry with last_hidden_state = post_layernorm(out)), which rescales the feature
        # distribution and lowers accuracy (~3.4% on Video-MME). Neutralize post_layernorm so
        # the projector receives the raw features, matching the original implementation exactly.
        try:
            self.model.model.vision_tower.vision_model.post_layernorm = (
                torch.nn.Identity()
            )
        except AttributeError:
            pass
        self.model.eval()

        if self.accelerator.num_processes > 1:
            assert self.accelerator.distributed_type in (
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ), "Only DDP/FSDP/DS are supported."
            if self.accelerator.distributed_type == DistributedType.FSDP:
                self.model = self.accelerator.prepare(self.model)
            else:
                self.model = self.accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
        else:
            self.model.to(self.device)

        self.frameidx_file = frameidx_file
        self.backup_file = backup_file
        self.lock_file = lock_file

    @property
    def rank(self):
        return self.accelerator.local_process_index

    @property
    def world_size(self):
        return self.accelerator.num_processes

    # ---- frame extraction (overridden by the blur subclass) ----
    def _frames_for(self, video_path, frame_set):
        return load_frames_native(video_path, frame_set)

    # ---- shelve helpers ----
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
                for k, v in cache_dict.items():
                    db[k] = v

    def loglikelihood(self, requests):
        raise NotImplementedError

    def _preprocessing_worker(
        self, requests, order, cached_keys, frame_sets, task_dict, q
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
            video_path = doc_to_visual(sample)
            frames_np = self._frames_for(video_path, frame_sets[req_key])
            cached_keys.add(req_key)
            q.put((skip_count, idx, req_key, frames_np))
            skip_count = 0
        q.put((skip_count, None, None, None))

    def _build_inputs(self, context, frames_np):
        question = IMAGE_PLACEHOLDERS.sub("", context).strip()
        # Use the processor's default chat template and default frame resize (plain HF path).
        conversation = [
            {
                "role": "user",
                "content": [{"type": "video"}, {"type": "text", "text": question}],
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, videos=[frames_np], return_tensors="pt")
        moved = {}
        for k, v in inputs.items():
            moved[k] = (
                v.to(self.device, self.dtype)
                if torch.is_floating_point(v)
                else v.to(self.device)
            )
        return moved

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

            temperature = gen_kwargs.get("temperature", 0.0)
            gen_cfg = dict(
                max_new_tokens=gen_kwargs.get("max_new_tokens", 128),
                num_beams=gen_kwargs.get("num_beams", 1),
                do_sample=temperature > 0,
            )
            if temperature > 0:
                gen_cfg["temperature"] = temperature
                if gen_kwargs.get("top_p", None) is not None:
                    gen_cfg["top_p"] = gen_kwargs["top_p"]

            inputs = self._build_inputs(context, frames_np)
            generated = self.model.generate(**inputs, use_cache=True, **gen_cfg)
            new_tokens = generated[:, inputs["input_ids"].shape[1] :]
            response = self.processor.batch_decode(
                new_tokens, skip_special_tokens=True
            )[0].strip()

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
            results.append(response_db[json.dumps((context, sample))])
        return results

    def generate_until_multi_round(self, requests):
        raise NotImplementedError
