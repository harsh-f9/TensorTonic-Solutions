import torch

def compute_gradient(values):
    t=torch.tensor(values, dtype=torch.float32)
    t.requires_grad_(True)
    y=(t**3 + 2*t).sum()
    y.backward()
    return t.grad.tolist()
