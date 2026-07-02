import torch


class XLogX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        mask = x != 0
        ctx.save_for_backward(x, mask)
        return torch.where(mask, x.xlogy(x), 0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, mask = ctx.saved_tensors
        return torch.where(mask, grad_output * (1 + x.log()), 0)


def xlogx(x: torch.Tensor) -> torch.Tensor:
    return XLogX.apply(x)


class XY(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mask = ((x == 0) & ~torch.isfinite(y)) | ((y == 0) & ~torch.isfinite(x))
        ctx.save_for_backward(x, y, mask)
        return (x * y).masked_fill_(mask, 0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, y, mask = ctx.saved_tensors
        dx = (grad_output * y).masked_fill_(mask, 0)
        dy = (grad_output * x).masked_fill_(mask, 0)
        return dx, dy


def xy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return XY.apply(x, y)
