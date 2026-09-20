"""Frame-preference optimization (FPO) on top of transformers.Trainer."""

from math import tanh

import torch
import torch.nn as nn
from mamba_ssm.ops.triton.layer_norm import RMSNorm as RMSNorm_A
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNorm_B
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer_pt_utils import get_parameter_names

from refocus.utils import xy
from train.metrics_callback import GlobalMetricsCallback
from train.preference import query_reward_api


def probability_margin_reward(record) -> float:
    """tanh((z_correct - max_{j != correct} z_j) / 2) over the option log-probabilities."""
    others = list(record["all_logp"])
    others.remove(record["ans_logp"])
    return tanh((record["ans_logp"] - max(others)) / 2)


def get_variable_length_grouped_indices(lengths, batch_size, world_size, megabatch_mult=8):
    """Global batches of similar length: sort into megabatches of megabatch_mult
    batches, shuffle within each, and yield a batch of the longest megabatch
    first so an OOM surfaces at step 0. Uses the global torch RNG, which the
    seed keeps identical across ranks."""
    WBS = world_size * batch_size
    MBS = WBS * megabatch_mult
    elements = list(range(len(lengths)))
    if remain := len(lengths) % WBS:
        elements += torch.randint(0, len(lengths), (WBS - remain,)).tolist()
    N = len(elements)
    B = N // WBS
    tiebreak = torch.randperm(N).tolist()
    megaindex = [None] * N
    for i, j in enumerate(sorted(elements, key=lambda i: (lengths[i], tiebreak[i]))):
        megaindex[j] = i // MBS
    shuffled = sorted(elements, key=lambda i: (megaindex[i], tiebreak[i]))
    batch_order = [B - 1] + torch.randperm(max(0, B - 1)).tolist()
    return [j for b in batch_order for j in shuffled[b * WBS : (b + 1) * WBS]]


class VariableLengthGroupedSampler(Sampler):
    def __init__(self, lengths, batch_size, world_size):
        self.lengths = lengths
        self.batch_size = batch_size
        self.world_size = world_size

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        return iter(
            get_variable_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size
            )
        )


class RefocusTrainer(Trainer):
    def __init__(self, *args, num_candidates: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_candidates = num_candidates

    def _get_train_sampler(self, train_dataset):
        return VariableLengthGroupedSampler(
            train_dataset.lengths, self.args.train_batch_size, self.args.world_size
        )

    def create_optimizer(self):
        if self.optimizer is None:
            params = dict(self.model.named_parameters())
            # mamba's RMSNorm classes are not in transformers' ALL_LAYERNORM_LAYERS.
            # get_parameter_names also lists the tied lm_head.weight, which
            # named_parameters() reports once as backbone.embedding.weight.
            no_decay_modules = (nn.LayerNorm, RMSNorm_A, RMSNorm_B)
            decay = {
                n
                for n in get_parameter_names(self.model, no_decay_modules)
                if n in params
                and "bias" not in n
                and not getattr(params[n], "_no_weight_decay", False)
            }
            # The re-initialized parts (last reinit_backbone_layers blocks,
            # selection heads, token embedding, final norm) use mm_fpo_adhoc_lr.
            n_layers = len(self.model.backbone.layers)
            adhoc = (
                "key_head",
                "query_head",
                "logit_scale",
                "value_head",
                "backbone.embedding",
                "backbone.norm_f",
            ) + tuple(
                f"backbone.layers.{i}."
                for i in range(n_layers - self.model.config.reinit_backbone_layers, n_layers)
            )
            groups = []
            for is_adhoc in (False, True):
                for is_decay in (True, False):
                    group = {
                        "params": [
                            p
                            for n, p in params.items()
                            if p.requires_grad
                            and n.startswith(adhoc) == is_adhoc
                            and (n in decay) == is_decay
                        ],
                        "weight_decay": self.args.weight_decay if is_decay else 0.0,
                    }
                    if is_adhoc:
                        group["lr"] = self.args.mm_fpo_adhoc_lr
                    groups.append(group)
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        B, C = len(inputs["input_ids"]), self.num_candidates
        Q = getattr(model, "module", model).config.query_frames  # DeepSpeed wraps the model
        frame_indices = inputs.pop("frame_indices")
        API_kwargs = inputs.pop("API_kwargs")

        candidate_, selection_logp_, candidate_logp_, sampler_logp_, _ = model(
            input_ids=inputs["input_ids"],
            visual_clips=inputs["visual_clips"],
            num_candidates=C,
        )
        sample_index = candidate_.cpu().reshape(B, C, Q).tolist()
        selection_logp = selection_logp_.cpu().reshape(B, C, Q).double()
        sampler_logp = sampler_logp_.cpu().reshape(B, C, Q).double()
        candidate_logp = [lp.cpu().double() for lp in candidate_logp_]  # B x [C, Q, F]

        # Every candidate of this step is scored under the same option order, so
        # reward differences within a sample come from the frames alone.
        seed = torch.randint(0, 2**31 - 1, (1,)).item()
        results = query_reward_api(
            [
                kw | dict(ref_indices=sorted(fi[i] for i in I), seed=seed)
                for kw, fi, si in zip(API_kwargs, frame_indices, sample_index)
                for I in si
            ]
        )
        success = torch.tensor([r["success"] for r in results]).reshape(B, C)
        correct = torch.tensor([r["success"] and r["is_correct"] for r in results]).reshape(B, C)
        reward = torch.tensor(
            [probability_margin_reward(r) if r["success"] else 0.0 for r in results],
            dtype=torch.float64,
        ).reshape(B, C)
        entropy = -torch.stack([xy(lp.exp(), lp).sum(-1) for lp in candidate_logp])  # [B, C, Q]

        with torch.inference_mode():
            n_success = success.sum().item()
            r = reward[success]
            record = GlobalMetricsCallback.record_metric
            record(self, "accuracy", correct.sum().item(), n_success)
            record(self, "reward", r.sum().item(), n_success)
            record(self, "reward_var", (r.square().sum() - r.sum().square() / max(n_success, 1)).item(), n_success)
            record(self, "entropy", entropy[success].nansum().item(), entropy[success].isfinite().sum().item())
            record(self, "request_fail", B * C - n_success, B * C)

        # Rewards are standardized per sample over its successful candidates.
        m = success.double()
        c = m.sum(1, keepdim=True).clamp_min(2)
        mu = (reward * m).sum(1, keepdim=True) / c
        var = ((reward - mu).square() * m).sum(1, keepdim=True) / (c - 1)
        score = (reward - mu) / (var.sqrt() + self.args.score_epsilon)

        ratio = (selection_logp - sampler_logp).mean(-1).exp()  # per-step importance ratio
        loss = -(ratio * score)[success].mean()
        loss = loss - self.args.entropy_coef * entropy.mean(-1)[success].mean()
        return loss.to(selection_logp_.device)
