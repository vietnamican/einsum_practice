import torch
from torch import as_strided
from torch import as_strided
from torch import einsum
from torch import nn

from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce

def to_tensor(*args):
    return (torch.Tensor(x) for x in args)

class EinMaxPool2d(nn.MaxPool2d):
    def __init__(self, *args, **kwargs):
        super(EinMaxPool2d, self).__init__(*args, **kwargs)
    
    # using einops package
    def max_pool2d_layer(self, x):
        result = reduce(x, 'b c (h h1) (w w1) -> b c h w', 'max', h1=2, w1=2)
        return result

    # just use torch 
    def max_pool2d_layer_numpy(self, x):
        b, c, h, w = x.shape
        b_strided, c_strided, h_strided, w_strided = x.stride()
        x_strided = as_strided(
            x, 
            (b, c, h // 2, w // 2, 2, 2), 
            (b_strided, c_strided, h_strided * 2, w_strided * 2, h_strided, w_strided)    
        )
        result = torch.amax(x_strided, dim=(-1, -2))
        return result

    def forward(self, x):
        result = self.max_pool2d_layer_numpy(x)
        return result