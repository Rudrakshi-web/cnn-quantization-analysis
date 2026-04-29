import torch

def quantize_tensor(x, bits=2):
    qmin = -(2**(bits-1))
    qmax = (2**(bits-1)) - 1

    max_val = x.abs().max()
    scale = max_val / qmax + 1e-8

    q_x = torch.round(x / scale)
    q_x = torch.clamp(q_x, qmin, qmax)

    return q_x * scale