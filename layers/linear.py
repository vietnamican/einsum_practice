import torch
from torch import as_strided
# from torch import einsum
from torch import nn

from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce

from opt_einsum import contract as einsum


def to_tensor(*args):
    return (torch.Tensor(x) for x in args)


class EinLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(EinLinear, self).__init__(*args, **kwargs)

    def forward(self, x):
        result = einsum('bi,oi->bo', x, self.weight)
        return result
