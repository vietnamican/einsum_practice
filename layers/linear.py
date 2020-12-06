import torch
from torch import as_strided
from torch import einsum
from torch import nn

from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce

def to_tensor(*args):
    return (torch.Tensor(x) for x in args)

class EinLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(EinLinear, self).__init__(*args, **kwargs)
    def forward(self, x):
        result = einsum('bi,oi->bo', x, self.weight)
        # print(x.shape)
        # print(self.weight.shape)
        # result = super().forward(x)
        # print(result.shape)
        return result