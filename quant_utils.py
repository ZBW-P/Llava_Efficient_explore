import torch

def dequantize_int8(q_x: torch.Tensor, scale: torch.Tensor):
    """
    Dequantize int8 tensor using stored scale.
    q_x: int8 tensor
    scale: float tensor
    """
    return (q_x.float() * scale)

def quantize_int8(x: torch.Tensor, bits: int = 8):
    """
    Quantize tensor into int8 using symmetric quantization.
    """
    assert 1 <= bits <= 8
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1

    x_max = x.abs().max()
    scale = x_max / qmax if x_max > 0 else torch.tensor(1.0, device=x.device, dtype=x.dtype)

    q_x = torch.round(x / scale).clamp(qmin, qmax)

    return q_x.to(torch.int8), scale
