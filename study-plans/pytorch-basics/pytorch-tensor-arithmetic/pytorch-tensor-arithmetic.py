import torch

def tensor_op(x, y, op):
    x=torch.tensor(x)
    y=torch.tensor(y)
    # for operations add, mul 
    def is_broadcastable(x,y):
        try:
            torch.broadcast_shapes(x.shape,y.shape)
            return True
        except RuntimeError:
            return False
    def is_multipliable(x,y):
        if x.shape[1]==y.shape[0]:
            return True
        else:
            return False
    if op=="add":
        if is_broadcastable(x,y):
            return torch.add(x,y).tolist()
    elif op=="multiply":
        if is_broadcastable(x,y):
            return torch.mul(x,y).tolist()
    elif op=="matmul":
        if is_multipliable(x,y):
            return torch.matmul(x,y).tolist()
    elif op=="power":
        return (x**y).tolist()
    elif op=="max":
        return torch.maximum(x,y).tolist()
    else:
        return "Invalid operation"
    


