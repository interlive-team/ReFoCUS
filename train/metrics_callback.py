import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
from transformers import Trainer, TrainerCallback


class GlobalMetricsCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        state._local_metrics = {}
        return control

    @staticmethod
    def record_metric(
        trainer: Trainer,
        metric_name: str,
        value: float,
        weight: float,
    ):
        stats = trainer.state._local_metrics.setdefault(metric_name, [0.0, 0.0])
        stats[0] += value
        stats[1] += weight

    def on_optimizer_step(self, args, state, control, **kwargs):
        device = args.device
        names = sorted(state._local_metrics.keys())
        sums = [state._local_metrics[n][0] for n in names]
        weights = [state._local_metrics[n][1] for n in names]
        flat_stats = torch.tensor(sums + weights, dtype=torch.float64, device=device)

        if dist.is_initialized():
            work = dist.reduce(flat_stats, dst=0, op=ReduceOp.SUM, async_op=True)
            if dist.get_rank() == 0:
                work.wait()
            else:
                state._local_metrics.clear()
                return control

        global_sums, global_weights = flat_stats.tensor_split(2, 0)
        global_avgs = (global_sums / global_weights).tolist()

        if not dist.is_initialized() or dist.get_rank() == 0:
            control = self._trainer.callback_handler.on_log(
                args,
                state,
                control,
                logs=dict(zip(names, global_avgs)),
            )

        state._local_metrics.clear()
        return control
