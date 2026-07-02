#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import copy
import math
import random
import re
import time
import warnings
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from refocus.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from refocus.mm_utils import get_anyres_image_grid_shape
from refocus.utils import rank0_print, rank_print

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

warnings.filterwarnings("ignore")


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.mm_projector = build_vision_projector(
                config, vision_cfg=self.vision_tower.config
            )

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(
            model_args, "vision_tower_pretrained", ""
        )

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower

            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(
            model_args, "mm_projector_type", "linear"
        )
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if not hasattr(self.config, "add_faster_video"):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(
                    torch.tensor(self.config.hidden_size, dtype=self.dtype)
                )
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(
                self.config, vision_cfg=vision_tower.config
            )

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(
                    torch.tensor(self.config.hidden_size, dtype=self.dtype)
                )
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            incompatible_keys = self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )
            rank0_print(
                f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}"
            )


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, weight = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(weight / stride)]
            image_feature = nn.functional.interpolate(
                image_feature, size=scaled_shape, mode="bilinear"
            )

        else:
            raise ValueError(
                f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}"
            )
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images, chunk_size=32):
        vision_tower = self.get_model().get_vision_tower()
        mm_projector = self.get_model().mm_projector

        match (
            any(p.requires_grad for p in vision_tower.parameters()),
            any(p.requires_grad for p in mm_projector.parameters()),
            torch.is_grad_enabled(),
        ):
            case (True, _, True):
                forward_fn = lambda x: torch.utils.checkpoint.checkpoint(
                    lambda y: mm_projector(self.get_2dPool(vision_tower(y))),
                    x,
                    use_reentrant=False,
                )
            case (False, True, True):
                forward_fn = lambda x: torch.utils.checkpoint.checkpoint(
                    mm_projector, self.get_2dPool(vision_tower(x)), use_reentrant=False
                )
            case _:
                forward_fn = lambda x: mm_projector(self.get_2dPool(vision_tower(x)))

        return torch.cat(
            [forward_fn(chunk) for chunk in torch.split(images, chunk_size)]
        )

    def prepare_inputs_labels_for_multimodal(self, input_ids, visual_clips):
        assert isinstance(input_ids, (list, tuple))
        assert isinstance(visual_clips, (list, tuple))
        assert all(map(torch.is_tensor, input_ids))
        assert all(map(torch.is_tensor, visual_clips))
        assert all(i.ndim == 4 for i in visual_clips)

        visual_features = self.encode_images(
            torch.cat(visual_clips, 0),
            getattr(self.config, "vision_encode_chunk_size", 8),
        ).split(list(map(len, visual_clips)), 0)

        if getattr(self.config, "mm_patch_merge_type", "flat") == "flat":
            visual_tails = [
                [(i + 1) * f.shape[1] - 1 for i in range(f.shape[0])]
                for f in visual_features
            ]
            visual_features = [f.flatten(0, 1) for f in visual_features]

        else:
            raise NotImplementedError()

        new_input_embeds, visual_indices = [], []
        cur_visual_clip_idx = 0
        for cur_input_ids in input_ids:
            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            num_images = len(image_token_indices) - 2
            text_segs = [
                cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                for i in range(len(image_token_indices) - 1)
            ]
            split_sizes = [seg.shape[0] for seg in text_segs]
            text_embeds = torch.split(
                self.get_model().embed_tokens(torch.cat(text_segs)), split_sizes, dim=0
            )
            segments, vis_idx = [], []
            cur_len = 0

            for i in range(num_images + 1):
                seg = text_embeds[i]
                segments.append(seg)
                cur_len += seg.shape[0]
                if i < num_images:
                    vf = visual_features[cur_visual_clip_idx]
                    vt = visual_tails[cur_visual_clip_idx]
                    seg_len = vf.shape[0]
                    vis_idx.extend([cur_len + tail for tail in vt])
                    segments.append(vf)
                    cur_len += seg_len
                    cur_visual_clip_idx += 1

            new_input_embeds.append(segments)
            visual_indices.append(vis_idx)

        # max padding
        seq_len = [sum(map(len, embed)) for embed in new_input_embeds]
        max_len = max(seq_len)

        # ## Mamba optimization
        # if hasattr(self, "backbone"):
        #     chunk_size = self.backbone.layers[0].mixer.chunk_size
        #     max_len = (max_len + chunk_size - 1) // chunk_size * chunk_size

        new_input_embeds = torch.stack(
            [
                torch.cat(
                    segs + [segs[0].new_zeros((max_len - s, segs[0].shape[1]))], dim=0
                )
                for segs, s in zip(new_input_embeds, seq_len)
            ]
        )
        B = len(new_input_embeds)
        vmasks = new_input_embeds.new_zeros(B, max_len, dtype=torch.bool)

        for mask, indices in zip(vmasks, visual_indices):
            mask[indices] = 1

        return new_input_embeds, vmasks

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )

        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
