import json
import os
import queue
import random
import shelve
import threading

import numpy as np
import torch
import tqdm
from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from filelock import FileLock
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from refocus_eval.model_utils.videoloader import VideoLoader
from transformers import AutoTokenizer
from transformers.trainer_utils import set_seed

from refocus.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from refocus.conversation import conv_templates
from refocus.mm_utils import tokenizer_image_token
from refocus.model.language_model.modeling_refocus import RefocusForFrameSelection


def load_video_decord(
    videoloader,
    video_path,
    max_num_frames,
    max_fps=None,
    return_index=False,
    return_sampled_fps=False,
):
    if isinstance(video_path, str):
        Vpath = video_path
    else:
        Vpath = video_path[0]

    vr = VideoReader(Vpath, ctx=cpu(0))
    total_frame_num = len(vr)
    orig_fps = vr.get_avg_fps()

    if max_fps is not None:
        max_frames_by_fps = max(1, int((total_frame_num / orig_fps) * max_fps))
        num_frames = min(max_num_frames, max_frames_by_fps)
    else:
        num_frames = max_num_frames

    sampled_fps = num_frames / total_frame_num * orig_fps

    frame_idx = np.linspace(0, total_frame_num - 1, num_frames, dtype=int).tolist()
    frames = videoloader.run(Vpath, frame_idx)
    outputs = [frames]
    if return_index:
        outputs.append(frame_idx)
    if return_sampled_fps:
        outputs.append(sampled_fps)

    return tuple(outputs) if len(outputs) > 1 else outputs[0]


def list_shuffle(lst):
    seed = int.from_bytes(os.urandom(16), "big")
    rng = random.Random(seed)
    rng.shuffle(lst)
    return lst


def logmeanexp(x, dim=None, keepdim=False):
    if dim is None:
        n = x.numel()
    elif isinstance(dim, int):
        n = x.shape[dim]
    else:
        n = 1
        for d in dim:
            n *= x.shape[d]
    return torch.logsumexp(x, dim=dim, keepdim=keepdim) - torch.log(
        torch.tensor(float(n))
    )


@register_model("ReFoCUS_FrameSelection_Noise")
class ReFoCUS_FrameSelection_Noise(lmms):
    def __init__(
        self,
        pretrained: str,
        *,
        database_file="database.db",
        lock_file="frame_order.lock",
        max_fps: float = 1.0,
        max_num_frames: int = 1024,
        conv_template: str = "fpo",
        temperature: float = 1.0,
        replace: bool = True,
        seed: int | None = 42,
        mask_time_s: float = 0,
        num_candidates: int = 64,
        num_query_frames: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.max_fps = max_fps
        self.max_num_frames = max_num_frames
        self.conv_template = conv_template
        self.temperature = temperature
        self.replace = replace
        self.seed = seed
        self.mask_time_s = mask_time_s
        self.database_file = database_file
        self.num_candidates = num_candidates
        self.num_query_frames = num_query_frames

        self.accelerator = Accelerator()
        if self.accelerator.num_processes > 1:
            self.device = torch.device(f"cuda:{self.accelerator.local_process_index}")
        else:
            self.device = torch.device("cuda")

        self.model = RefocusForFrameSelection.from_pretrained(
            pretrained,
            trust_remote_code=True,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )
        self.image_processor = self.model.get_model().get_vision_tower().image_processor
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

        if self.accelerator.num_processes > 1:
            distributed_type_list = [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ]
            assert (
                self.accelerator.distributed_type in distributed_type_list
            ), "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if self.accelerator.distributed_type == DistributedType.FSDP:
                self._model = self.accelerator.prepare(self.model)
            else:
                self._model = self.accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
        else:
            self.model.to(self.device)

        # Override query_frames config
        if self.num_query_frames is not None:
            self.model.config.query_frames = num_query_frames

        self.lock_file = lock_file

        input_size = self.model.get_vision_tower().image_processor.size[0]
        self.videoloader = VideoLoader((input_size, input_size), num_workers=10)

    @property
    def rank(self):
        return self.accelerator.local_process_index

    @property
    def world_size(self):
        return self.accelerator.num_processes

    def _load_last_state_keys(self):
        with FileLock(self.lock_file):
            with shelve.open(self.database_file) as db:
                return set(db.keys())

    def _save_last_state(self, cache_dict):
        with FileLock(self.lock_file):
            with shelve.open(self.database_file, writeback=False) as db:
                db.update(cache_dict)

    def _preprocessing_worker(
        self, requests, order, cached_keys, task_dict, q: queue.Queue
    ):
        skip_count = 0

        for idx in order:
            context, gen_kwargs, doc_to_visual, doc_id, task, split = requests[idx].args
            sample = task_dict[task][split][doc_id]
            sample.pop("image", None)  # MMMU
            req_key = json.dumps((context, sample))

            if req_key in cached_keys:
                skip_count += 1
                continue

            try:
                video_path = doc_to_visual(sample)
                frames_np, frame_idx, sample_fps = load_video_decord(
                    self.videoloader,
                    video_path,
                    max_num_frames=self.max_num_frames,
                    max_fps=self.max_fps,
                    return_index=True,
                    return_sampled_fps=True,
                )
                cached_keys.add(req_key)
                q.put((skip_count, req_key, frames_np, frame_idx, sample_fps, idx))
                skip_count = 0

            except Exception as e:
                print(f"[WARN] Skipping {req_key} due to error: {e}")
                skip_count += 1
                continue

        q.put((skip_count, None, None, None, None, None))

    def loglikelihood(self, requests) -> list[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

    @torch.inference_mode()
    def generate_until(self, requests):
        N_QUERY = self.model.config.query_frames
        cache = {}
        cached_keys = self._load_last_state_keys()
        q = queue.Queue(maxsize=1)

        order = list(range(len(requests)))
        list_shuffle(order)

        producer = threading.Thread(
            target=self._preprocessing_worker,
            args=(requests, order, cached_keys, self.task_dict, q),
        )
        producer.start()

        pbar = tqdm.tqdm(
            total=len(order),
            disable=(self.rank != 3),
            desc=f"Rank{self.rank} responding",
            mininterval=10,
        )

        while True:
            skip_count, req_key, frames_np, frame_idx, sample_fps, idx = q.get()
            if req_key is None:
                break
            context, gen_kwargs, doc_to_visual, doc_id, task, split = requests[idx].args

            pbar.update(skip_count + (1 if req_key else 0))

            if (
                len(frames_np) > N_QUERY
                and len(frame_idx)
                >= (int(self.mask_time_s * sample_fps) * 2 + 1) * N_QUERY
            ):
                if isinstance(self.seed, int):
                    set_seed(self.seed)
                frames = self.image_processor.preprocess(
                    frames_np, return_tensors="pt"
                )["pixel_values"].to(self.device, torch.bfloat16)

                if DEFAULT_IMAGE_TOKEN not in context:
                    context = DEFAULT_IMAGE_TOKEN + "\n" + context

                conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], context)
                # conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                input_ids = tokenizer_image_token(
                    prompt_question,
                    self.tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                ).to(self.device)
                opt = {}
                if self.mask_time_s:
                    opt["mask_radius"] = int(self.mask_time_s * sample_fps)

                C = self.num_candidates
                Q = self.model.config.query_frames
                F = frames.shape[0]
                forced_idx = torch.full((1, C, Q), -1, dtype=torch.long)
                for i in range(C):
                    perm = torch.randperm(F)
                    take = min(i + 1, Q, F)
                    if take > 0:
                        forced_idx[0, i, :take] = perm[:take]

                candidates, logits, logp, sampler_logp, rank_logit = self.model(
                    input_ids=[input_ids],
                    visual_clips=[frames],
                    num_candidates=self.num_candidates,
                    temperature=self.temperature,
                    replace=self.replace,
                    forced_idx=forced_idx.to(self.device),
                    **opt,
                )

                candidates = candidates.int().cpu().numpy()
                prob_mat = logmeanexp(logp[0].double(), (0, 1)).half().cpu().numpy()
                sample_transition = logp[0].half().cpu().numpy()
                if rank_logit is not None:
                    rank_logit = rank_logit.flatten().double().cpu().numpy()
            else:
                candidates = None
                prob_mat = None
                sample_transition = None
                rank_logit = None

            cache[req_key] = dict(
                frame_idx=frame_idx,
                candidates=candidates,
                distribution=prob_mat,
                transition=sample_transition,
                rank_logit=rank_logit,
            )
            if len(cache) % 16 == 0:
                self._save_last_state(cache)
                cache.clear()

        if cache:
            self._save_last_state(cache)

        pbar.close()
        producer.join()

        return [""] * len(requests)

    def generate_until_multi_round(self, requests) -> list[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
