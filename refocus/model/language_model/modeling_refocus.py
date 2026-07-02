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
import re
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from torch.distributions import Categorical
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from refocus.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from refocus.model.llava_arch import LlavaMetaForCausalLM, LlavaMetaModel

warnings.filterwarnings("ignore")

LOGIT_SCALE_FNS = {
    "exp": torch.exp,
    "softplus": F.softplus,
}

LOGIT_SCALE_INV_FNS = {
    "exp": torch.log,
    "softplus": lambda x: torch.log(torch.expm1(x)),
}


class RefocusConfig(PretrainedConfig):
    model_type = "refocus"

    query_frames = 32

    reinit_backbone_layers = 1
    value_embedding_init_range = 0.02
    value_head_init_gain = 0.1
    value_head_norm = False
    value_head_scale = 1.0
    score_head_init_gain = 0.1
    score_head_norm = False
    logit_scale_fn = "softplus"
    logit_scale_init = 1.0
    logit_scale_type = "scalar"
    start_of_frame_token_id = None
    mixer_norm_rescale = 1.0
    frame_generation_method = "block"
    attention_scaling = False
    frame_replacement = True
    use_rank_head = False

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def scale_fn(self):
        return LOGIT_SCALE_FNS[self.logit_scale_fn]

    @property
    def scale_inv(self):
        return LOGIT_SCALE_INV_FNS[self.logit_scale_fn]

    def to_dict(self):
        return PretrainedConfig.to_dict(self)


class dummy_cls(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class RefocusModel(LlavaMetaModel, dummy_cls):
    config_class = RefocusConfig

    def __init__(self, config: RefocusConfig):
        super(RefocusModel, self).__init__(config)
        self.config = config


class RefocusForFrameSelection(PreTrainedModel, MambaLMHeadModel, LlavaMetaForCausalLM):
    config_class = RefocusConfig
    get_input_embeddings = MambaLMHeadModel.get_input_embeddings
    gradient_checkpointing_enable = MambaLMHeadModel.gradient_checkpointing_enable
    gradient_checkpointing_disable = MambaLMHeadModel.gradient_checkpointing_disable

    def __init__(self, config: RefocusConfig):
        super(PreTrainedModel, self).__init__(config)
        MambaLMHeadModel.__init__(self, config)
        config.model_type = "refocus"
        self.mm_model = RefocusModel(config)
        self.mm_model.embed_tokens = lambda x: self.backbone.embedding(x)

        linear_types = {
            False: nn.Linear,
            True: NormLinear,
            "False": nn.Linear,
            "True": NormLinear,
            "norm": NormLinear,
            "rmsnorm": RMSNormLinear,
        }
        score_linear = linear_types[self.config.score_head_norm]
        value_linear = linear_types[self.config.value_head_norm]

        self.key_head = score_linear(config.hidden_size, config.hidden_size, bias=False)
        self.query_head = score_linear(
            config.hidden_size, config.hidden_size, bias=False
        )

        if config.frame_generation_method == "block":
            self.value_embedding = nn.Parameter(
                torch.empty((config.query_frames, config.hidden_size))
            )
        elif config.frame_generation_method == "autoregressive":
            self.value_head = value_linear(
                config.hidden_size, config.hidden_size, bias=True
            )

            if config.value_head_norm:
                self.value_head_scale = nn.Parameter(torch.empty((config.hidden_size,)))
                self.value_head_scale._no_weight_decay = True
            else:
                self.value_head_scale = None

            # The start-of-frame-sequence prefix is a real vocabulary token
            # (<|startofframe|>); its vector is the embedding-table row at
            # config.start_of_frame_token_id (see `start_of_frame_embedding`).
            assert (
                config.start_of_frame_token_id is not None
            ), "autoregressive frame generation requires config.start_of_frame_token_id"

        self.logit_scale = nn.Parameter(
            torch.empty(
                self.config.hidden_size
                if self.config.logit_scale_type == "vector"
                else ()
            )
        )
        self.logit_scale._no_weight_decay = True

        if getattr(self.config, "use_rank_head", False):
            self.rank_head = nn.Linear(config.hidden_size, 1, bias=False)

        self._no_weight_decay_params = [
            name
            for name, param in self.named_parameters()
            if getattr(param, "_no_weight_decay", False)
        ]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        params = dict(model.named_parameters())
        for name in getattr(model, "_no_weight_decay_params", []):
            if name in params:
                setattr(params[name], "_no_weight_decay", True)
        return model

    @torch.no_grad()
    def reinit_for_finetune(self):
        def init_orthogonal(param: nn.Parameter, gain: float = 1.0):
            tmp = nn.init.orthogonal_(
                torch.empty_like(param, dtype=torch.float32), gain=gain
            )
            with torch.no_grad():
                param.copy_(tmp.to(param.dtype))

        if getattr(self.config, "reinit_backbone_layers", 0) > 0:
            reinit_count = self.config.reinit_backbone_layers
            self_layers = self.backbone.layers[-reinit_count:]
            new_layers = MambaLMHeadModel(self.config).backbone.layers[-reinit_count:]
            for self_layer, new_layer in zip(self_layers, new_layers):
                self_layer.load_state_dict(new_layer.state_dict())
                self_layer.mixer.norm.weight.fill_(self.config.mixer_norm_rescale)

        score_gain = self.config.score_head_init_gain
        init_orthogonal(self.key_head.weight, gain=score_gain)
        init_orthogonal(self.query_head.weight, gain=score_gain)

        init_range = self.config.value_embedding_init_range
        value_gain = self.config.value_head_init_gain
        if self.config.frame_generation_method == "block":
            nn.init.normal_(self.value_embedding, mean=0.0, std=init_range)
        elif self.config.frame_generation_method == "autoregressive":
            init_orthogonal(self.value_head.weight, gain=value_gain)
            nn.init.zeros_(self.value_head.bias)
            if self.value_head_scale is not None:
                self.value_head_scale.fill_(self.config.value_head_scale)
            nn.init.normal_(
                self.backbone.embedding.weight[self.config.start_of_frame_token_id],
                mean=0.0,
                std=init_range,
            )

        nn.init.ones_(self.backbone.norm_f.weight)
        if getattr(self.backbone.norm_f, "bias", None) is not None:
            nn.init.zeros_(self.backbone.norm_f.bias)

        init_val = torch.tensor(self.config.logit_scale_init)
        self.logit_scale.fill_(self.config.scale_inv(init_val))

        if getattr(self.config, "use_rank_head", False):
            nn.init.normal_(self.rank_head.weight, mean=0.0, std=0.01)

    def get_model(self):
        return self.mm_model

    @property
    def start_of_frame_embedding(self):
        # (1, d) view into the token-embedding table for the <|startofframe|>
        # token; differentiable, so training can update this single row.
        tok = self.config.start_of_frame_token_id
        return self.backbone.embedding.weight[tok].unsqueeze(0)

    def can_generate(self):
        return False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        visual_clips: List[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        num_candidates: int = 4,
        temperature: float = 1.0,
        replace: Optional[bool] = None,
        forced_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, List[torch.FloatTensor]]:

        if inputs_embeds is None:
            (
                inputs_embeds,
                visual_mask,
            ) = self.prepare_inputs_labels_for_multimodal(input_ids, visual_clips)
        else:
            assert torch.is_tensor(visual_mask)
            assert torch.is_tensor(query_mask)

        B = inputs_embeds.size(0)
        device = inputs_embeds.device
        logit_scale = self.config.scale_fn(self.logit_scale) / temperature
        if self.config.attention_scaling:
            logit_scale = logit_scale / math.sqrt(self.config.hidden_size)
        replacement = self.config.frame_replacement if replace is None else replace
        if forced_idx is not None:
            assert forced_idx.dim() == 3
            assert forced_idx.shape == (B, num_candidates, self.config.query_frames)
            forced_flat = forced_idx.reshape(
                B * num_candidates, self.config.query_frames
            )
        else:
            forced_flat = None

        hidden_states, conv_states, ssm_states = self.backbone(
            None, inputs_embeds, return_cache=True
        )
        visual_hidden = hidden_states[visual_mask]  # [B * F, d]
        num_frames = visual_mask.sum(1).detach().cpu().tolist()
        visual_key = self.key_head(visual_hidden).split(
            num_frames, 0
        )  # list of [F_i, d] per batch
        if hasattr(self, "value_head"):
            visual_value = self.value_head(visual_hidden)
            if self.value_head_scale is not None:
                visual_value = visual_value * self.value_head_scale
            visual_value = visual_value.split(num_frames, 0)  # list of [F_i, d]

        candidate_list = []
        selection_logp_list = []
        sampling_logp_list = []  # [C, Q, F] shape tensor with B length list

        match self.config.frame_generation_method:
            case "block":
                new_embedding = self.value_embedding.unsqueeze(0).expand(B, -1, -1)
                query_hidden = self.backbone(
                    None, new_embedding, conv_states, ssm_states, return_cache=False
                )
                visual_query = self.query_head(query_hidden)  # [B, Q, d]

                for vk, vq in zip(visual_key, visual_query):
                    logits = torch.einsum("qd,kd->qk", vq, vk * logit_scale)  # [Q, F]
                    candidates_for_sample = []
                    logp_for_sample = []
                    candidates_logp = []
                    for _ in range(num_candidates):
                        candidate = []
                        total_logp = 0
                        for q_logit in logits:
                            if not replacement and candidate:
                                idx_tensor = torch.tensor(
                                    candidate, dtype=torch.long, device=device
                                )
                                masked_logit = q_logit.index_fill_(
                                    0, idx_tensor, -float("inf")
                                )
                            else:
                                masked_logit = q_logit

                            logp = masked_logit.log_softmax(-1, dtype=torch.float32)
                            prob = masked_logit.softmax(-1, dtype=torch.float32)
                            candidates_logp.append(logp)
                            idx = torch.multinomial(prob, 1).item()
                            candidate.append(idx)
                            total_logp += logp[idx]
                        candidates_for_sample.append(candidate)
                        logp_for_sample.append(total_logp)
                    candidate_list.append(
                        torch.tensor(candidates_for_sample, device=device)
                    )  # shape: [C, Q]
                    selection_logp_list.append(
                        torch.stack(logp_for_sample, dim=0)
                    )  # shape: [C]
                    sampling_logp_list.append(
                        torch.stack(candidates_logp, dim=0).unflatten(
                            0, [num_candidates, self.config.query_frames]
                        )
                    )
                return (
                    torch.cat(candidate_list),
                    torch.cat(selection_logp_list),
                    sampling_logp_list,
                )
            case "autoregressive":
                # Expand visual_key and visual_value per candidate
                visual_key_ = [vk for vk in visual_key for _ in range(num_candidates)]
                visual_value_ = [
                    vv for vv in visual_value for _ in range(num_candidates)
                ]
                with torch.inference_mode():
                    conv_states_ = [
                        st.repeat_interleave(num_candidates, dim=0)
                        for st in conv_states
                    ]
                    ssm_states_ = [
                        st.repeat_interleave(num_candidates, dim=0) for st in ssm_states
                    ]
                    new_embedding = self.start_of_frame_embedding.unsqueeze(0).expand(
                        B * num_candidates, -1, -1
                    )
                    candidate_list = torch.empty(
                        [B * num_candidates, self.config.query_frames],
                        dtype=torch.long,
                        device=device,
                    )
                    sampler_logp = torch.full(
                        [B * num_candidates, self.config.query_frames],
                        float("nan"),
                        dtype=torch.float32,
                        device=device,
                    )

                    for Qk in range(self.config.query_frames):
                        qh, conv_states_, ssm_states_ = self.backbone.step(
                            None,
                            new_embedding,
                            conv_states_,
                            ssm_states_,
                            cache_inplace=True,
                        )
                        # Compute query vector: [B*num_candidates, 1, d] -> [B*num_candidates, d]
                        vq = self.query_head(qh).squeeze(1)
                        for row_i, (vk, vq_j, cl, slp) in enumerate(
                            zip(visual_key_, vq, candidate_list, sampler_logp)
                        ):
                            logits = torch.einsum("kd,d->k", vk, vq_j * logit_scale)
                            if not replacement and Qk > 0:
                                masked_logits = logits.index_fill_(0, cl, -float("inf"))
                            else:
                                masked_logits = logits
                            prob = masked_logits.softmax(0, dtype=torch.float32)
                            sampled_idx = torch.multinomial(prob, 1).view(())
                            if forced_flat is not None:
                                forced_val = forced_flat[row_i, Qk]
                                use_forced = forced_val.ge(0)
                                idx = torch.where(
                                    use_forced, forced_val, sampled_idx
                                ).to(torch.long)
                            else:
                                idx = sampled_idx
                            slp[Qk] = masked_logits.log_softmax(0, dtype=torch.float32)[
                                idx
                            ]
                            if Qk > 0:
                                cl[Qk].copy_(idx)
                            else:
                                cl.fill_(idx)
                        new_embedding = torch.stack(
                            [
                                vv[idx]
                                for vv, idx in zip(visual_value_, candidate_list[:, Qk])
                            ],
                            0,
                        ).unsqueeze(1)
                    del conv_states_, ssm_states_, new_embedding
                candidate_list = candidate_list.detach().clone()
                conv_states_ = [
                    st.repeat_interleave(num_candidates, dim=0) for st in conv_states
                ]
                ssm_states_ = [
                    st.repeat_interleave(num_candidates, dim=0) for st in ssm_states
                ]
                new_embedding = torch.stack(
                    [
                        torch.cat([self.start_of_frame_embedding, vv[cl]], 0)
                        for vv, cl in zip(visual_value_, candidate_list)
                    ],
                    0,
                )  # [B * C, Q, d]
                query_hidden, rank_hidden = self.backbone(
                    None, new_embedding, conv_states_, ssm_states_, return_cache=False
                ).split([self.config.query_frames, 1], 1)
                visual_query = self.query_head(query_hidden).reshape(
                    B, num_candidates, self.config.query_frames, -1
                )
                if getattr(self.config, "use_rank_head", False):
                    rank_logits = self.rank_head(rank_hidden).squeeze([1, 2])  # (B*C)
                else:
                    rank_logits = None
                candidate_tensor = candidate_list.reshape(
                    B, num_candidates, -1
                )  # [B, C, Q]
                for vk, vq, cl in zip(visual_key, visual_query, candidate_tensor):
                    # vk: [F, d], vq: [C, Q, d], cl: [C, Q]
                    # Compute logits: [C, Q, F]

                    logits = torch.einsum("cqd,fd->cqf", vq, vk * logit_scale)
                    if not replacement:
                        logits = torch.stack(
                            [
                                logit.scatter(1, cl.narrow(1, 0, i), -float("inf"))
                                for i, logit in enumerate(logits.unbind(1))
                            ],
                            dim=1,
                        )
                    logp = logits.log_softmax(-1, dtype=torch.float32)
                    sample_logp = logp.gather(-1, cl.unsqueeze(-1)).flatten(1)
                    selection_logp_list.append(sample_logp)
                    sampling_logp_list.append(logp)

                return (
                    candidate_list,
                    torch.cat(selection_logp_list),
                    sampling_logp_list,
                    sampler_logp.detach().clone(),
                    rank_logits,
                )

    def floating_point_ops(self, input_args=None):
        return 0


class NormLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(super().forward(x), p=2, dim=-1, eps=1e-6)


class RMSNormLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        rms = y.square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        return y / rms


# Register the canonical "refocus" model_type, and keep the legacy "mamba2" /
# "llava_mamba" strings mapped to RefocusConfig for backward compatibility so
# AutoConfig.from_pretrained still resolves base Video-Ma2mba checkpoints and any
# selector checkpoint whose config.json predates the rename.
RefocusConfig.model_type = "mamba2"
AutoConfig.register("mamba2", RefocusConfig, exist_ok=True)
RefocusConfig.model_type = "llava_mamba"
AutoConfig.register("llava_mamba", RefocusConfig, exist_ok=True)
RefocusConfig.model_type = "refocus"
AutoConfig.register("refocus", RefocusConfig, exist_ok=True)
AutoModelForCausalLM.register(RefocusConfig, RefocusForFrameSelection)
