import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    """
    Returns: y = x W^T + b without using nn.Linear
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight= nn.Parameter(torch.empty(out_features,in_features))
        self.bias= nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        #bound for bias
        fan_in,_=nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound=1/math.sqrt(fan_in) if fan_in>0 else 0
        nn.init.uniform_(self.bias,-bound,bound)
    def forward(self, x):
        return x@self.weight.T + self.bias
