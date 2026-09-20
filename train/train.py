"""FPO training for ReFoCUS; scripts/train_refocus.sh is the argv contract."""

import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset, get_worker_info
from transformers import AutoConfig
from transformers.trainer_utils import set_seed

from refocus import conversation as conversation_lib
from refocus.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN
from refocus.mm_utils import tokenizer_image_token
from refocus.model import RefocusConfig, RefocusForFrameSelection
from refocus.utils import VideoLoader, rank0_print
from train.metrics_callback import GlobalMetricsCallback
from train.refocus_trainer import RefocusTrainer

# Whole videos (up to frames_upbound frames) cross the DataLoader worker
# boundary; the default file-descriptor sharing runs out of descriptors.
torch.multiprocessing.set_sharing_strategy("file_system")


@dataclass
class ModelArguments:
    model_name_or_path: str
    version: str
    vision_tower: str
    mm_tunable_parts: str
    mm_vision_select_layer: int
    mm_projector_type: str
    mm_spatial_pool_mode: str
    mm_use_im_start_end: bool
    mm_use_im_patch_token: bool
    model_reinit_backbone_layers: int
    query_frames: int
    value_embedding_init_range: float
    value_head_init_gain: float
    value_head_norm: str
    value_head_scale: float
    score_head_init_gain: float
    score_head_norm: str
    logit_scale_fn: str
    logit_scale_init: float
    logit_scale_type: str
    mixer_norm_rescale: float
    frame_generation_method: str
    attention_scaling: bool
    frame_replacement: bool
    use_rank_head: bool
    # Read by LlavaMetaModel.initialize_vision_modules; not exposed by train_refocus.sh.
    mm_vision_select_feature: str = "patch"
    mm_patch_merge_type: str = "flat"
    pretrain_mm_mlp_adapter: Optional[str] = None
    vision_tower_pretrained: Optional[str] = None


@dataclass
class DataArguments:
    data_path: str
    video_folder: str
    video_folder_remote: str
    lazy_preprocess: bool
    frames_upbound: int
    video_fps: int
    train_step: str
    join_clips: bool
    num_candidates: int
    remote_evaluation_urls: List[str]


@dataclass(kw_only=True)
class TrainingArguments(transformers.TrainingArguments):
    # transformers 4.57.1 types this without Optional, unlike its sibling
    # _VALID_DICT_FIELDS, so HfArgumentParser binds type=dict and rejects the
    # CLI dict string.
    lr_scheduler_kwargs: Optional[Union[dict[str, Any], str]] = field(
        default_factory=dict
    )
    # Batches carry keys the model forward does not take (frame_indices,
    # API_kwargs); the default column pruning would drop them before collation.
    remove_unused_columns: bool = False
    model_max_length: int
    mm_fpo_adhoc_lr: float
    model_reinit: bool
    group_by_varlen: bool
    reward_normalize: str
    reward_modeling: str
    shuffle_options: str
    reinforce_scale: str
    entropy_coef: float
    score_epsilon: float


def preprocess_fpo(sample: Dict, tokenizer) -> torch.Tensor:
    """Prompt tokens for the human turns, with IMAGE_TOKEN_INDEX where the video goes."""
    conv = conversation_lib.default_conversation.copy()
    for turn in sample["conversations"][::2]:
        assert turn["from"] == "human"
        value = turn["value"]
        if DEFAULT_VIDEO_TOKEN in value:
            value = value.replace(DEFAULT_VIDEO_TOKEN, "").strip()
            value = (DEFAULT_IMAGE_TOKEN + "\n" + value).strip()
        conv.append_message(conv.roles[0], value)
    return tokenizer_image_token(conv.get_prompt(), tokenizer, return_tensors="pt")


class FPODataset(Dataset):
    TOKENS_PER_FRAME = 144
    TOKENS_PER_CHAR = 0.25

    def __init__(self, tokenizer, image_processor, data_args: DataArguments):
        with open(data_args.data_path) as f:
            self.samples = json.load(f)
        rank0_print(f"Loaded {len(self.samples)} samples from {data_args.data_path}")
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data_args = data_args
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        # Its thread pool must be created inside the DataLoader worker, not forked.
        self.videoloader = None
        self.fallback_index = None

    def __len__(self):
        return len(self.samples)

    @property
    def lengths(self):
        """Token estimate per sample, for length grouping."""
        lengths = []
        for sample in self.samples:
            tokens = 0
            for scene in sample["scenes"]:
                stride = (scene["start_fps"] + scene["end_fps"]) / 2
                stride /= self.data_args.video_fps
                frames = (scene["end_frame"] - scene["start_frame"]) // stride
                tokens += min(frames, self.data_args.frames_upbound) * self.TOKENS_PER_FRAME
            chars = sum(len(turn["value"]) for turn in sample["conversations"])
            lengths.append(tokens + chars * self.TOKENS_PER_CHAR)
        return lengths

    def load_video(self, sample):
        """Frames of every scene at video_fps, at most frames_upbound per scene."""
        frame_indices = []
        for scene in sample["scenes"]:
            stride = (scene["start_fps"] + scene["end_fps"]) / 2
            stride /= self.data_args.video_fps
            n_frames = int((scene["end_frame"] - scene["start_frame"]) // stride + 1)
            n_frames = min(n_frames, self.data_args.frames_upbound)
            frame_indices += (
                np.linspace(scene["start_frame"], scene["end_frame"] - 1, n_frames)
                .round()
                .astype(int)
                .tolist()
            )
        if self.videoloader is None:
            self.videoloader = VideoLoader(
                target_size=self.image_processor.size, multiprocessing=False
            )
        video_path = os.path.join(self.data_args.video_folder, sample["video"])
        frames = self.videoloader.run(video_path, frame_indices)
        processed = self.image_processor.preprocess(frames, return_tensors="pt")
        return processed["pixel_values"], frame_indices

    def process_sample(self, i):
        sample = self.samples[i]
        visual_clips, frame_indices = self.load_video(sample)
        return dict(
            input_ids=preprocess_fpo(sample, self.tokenizer),
            visual_clips=visual_clips,
            frame_indices=frame_indices,
            API_kwargs=dict(
                url=self.data_args.remote_evaluation_urls[self.rank],
                video_path=os.path.join(
                    self.data_args.video_folder_remote, sample["video"]
                ),
                question=sample["question"],
                options=sample["options"],
                answer=sample["answer"],
            ),
        )

    def __getitem__(self, i):
        try:
            return self.process_sample(i)
        except Exception:
            # An undecodable video is replaced by the next sample of this
            # worker's own slice, so the step count is unchanged.
            if self.fallback_index is None:
                rank, world = self.rank, self.world_size
                if (worker := get_worker_info()) is not None:
                    rank = rank * worker.num_workers + worker.id
                    world = world * worker.num_workers
                self.fallback_index = len(self.samples) * rank // world
            self.fallback_index += 1
            return self.process_sample(self.fallback_index - 1)


def collate(instances: List[Dict]) -> Dict[str, list]:
    return {key: [instance[key] for instance in instances] for key in instances[0]}


def get_model(model_args: ModelArguments, training_args: TrainingArguments):
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    assert isinstance(config, RefocusConfig)
    overwrite = dict(
        reinit_backbone_layers=model_args.model_reinit_backbone_layers,
        query_frames=model_args.query_frames,
        value_embedding_init_range=model_args.value_embedding_init_range,
        value_head_init_gain=model_args.value_head_init_gain,
        value_head_norm=model_args.value_head_norm,
        value_head_scale=model_args.value_head_scale,
        score_head_init_gain=model_args.score_head_init_gain,
        score_head_norm=model_args.score_head_norm,
        logit_scale_fn=model_args.logit_scale_fn,
        logit_scale_init=model_args.logit_scale_init,
        logit_scale_type=model_args.logit_scale_type,
        mixer_norm_rescale=model_args.mixer_norm_rescale,
        frame_generation_method=model_args.frame_generation_method,
        attention_scaling=model_args.attention_scaling,
        frame_replacement=model_args.frame_replacement,
        use_rank_head=model_args.use_rank_head,
        mm_spatial_pool_mode=model_args.mm_spatial_pool_mode,
    )
    # <|startofframe|> token: if the base model does not already define it,
    # claim the first vocab-padding slot (= base vocab_size) as its id. The
    # embedding table is padded to a multiple of pad_vocab_size_multiple, so
    # that row already exists; no resize needed.
    if config.start_of_frame_token_id is None:
        overwrite["start_of_frame_token_id"] = config.vocab_size
        overwrite["vocab_size"] = config.vocab_size + 1
    rank0_print(f"Overwriting config with {overwrite}")
    for key, value in overwrite.items():
        setattr(config, key, value)

    model = RefocusForFrameSelection.from_pretrained(
        model_args.model_name_or_path, config=config, torch_dtype=torch.bfloat16
    )
    if training_args.model_reinit:
        model.reinit_for_finetune()
    return model


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # train_refocus.sh fixes these; only that configuration is implemented.
    assert model_args.version == "fpo"
    assert not model_args.mm_use_im_start_end
    assert not model_args.mm_use_im_patch_token
    assert set(model_args.mm_tunable_parts.split(",")) == {
        "mm_fpo_adhoc",
        "mm_language_model",
        "mm_mlp_adapter",
    }
    assert data_args.train_step == "fpo"
    assert data_args.join_clips
    assert training_args.bf16
    assert training_args.deepspeed
    assert training_args.group_by_varlen
    assert training_args.reward_modeling == "probability-margin"
    assert training_args.reward_normalize == "normalize"
    assert training_args.reinforce_scale == "step"
    assert training_args.shuffle_options == "shared"
    # Each rank queries its own reward server.
    assert len(data_args.remote_evaluation_urls) == training_args.world_size
    set_seed(training_args.seed)

    model = get_model(model_args, training_args)
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b", model_max_length=training_args.model_max_length
    )
    tokenizer.pad_token = "<|padding|>"
    # Must land on the vocab-padding slot get_model claimed as start_of_frame_token_id.
    start_of_frame_token = "<|startofframe|>"
    assert (
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [start_of_frame_token]}
        )
        == 1
    )
    assert (
        tokenizer.convert_tokens_to_ids(start_of_frame_token)
        == model.config.start_of_frame_token_id
    )
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        model_args.version
    ]

    model.get_model().initialize_vision_modules(
        model_args=model_args, fsdp=training_args.fsdp
    )
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=training_args.device)
    model.config.mm_tunable_parts = model_args.mm_tunable_parts
    # Recipe metadata kept in config.json, as in the released checkpoints.
    model.config.mm_fpo_adhoc_lr = training_args.mm_fpo_adhoc_lr
    model.config.reward_modeling = training_args.reward_modeling
    model.config.reward_normalize = training_args.reward_normalize

    # Everything but the vision tower is trained; of the token-embedding table
    # only the <|startofframe|> row receives gradient.
    for name, param in model.named_parameters():
        param.requires_grad_("vision_tower" not in name)
    sof = model.config.start_of_frame_token_id
    model.backbone.embedding.weight.register_hook(
        lambda g: g * torch.zeros_like(g).index_fill_(0, g.new_tensor([sof], dtype=torch.long), 1.0)
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Trainable parameters: {trainable / 1e6:.1f}M")

    trainer = RefocusTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=FPODataset(tokenizer, vision_tower.image_processor, data_args),
        data_collator=collate,
        num_candidates=data_args.num_candidates,
    )
    trainer.add_callback(GlobalMetricsCallback(trainer))
    trainer.train(
        resume_from_checkpoint=any(
            pathlib.Path(training_args.output_dir).glob("checkpoint-*")
        )
    )
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
