import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    alpha=0.01
    t=torch.tensor(x,dtype=torch.float32)
    if method=="relu":
        return torch.clamp(t,min=0).tolist()
    elif method=="sigmoid":
        return (1/(1+torch.exp(-t))).tolist()
    elif method=="tanh":
        return torch.tanh(t).tolist()
    elif method=="leaky_relu":
        return torch.where(t>0,t,alpha*t).tolist()
