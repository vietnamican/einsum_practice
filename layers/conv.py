import torch
from torch import as_strided
from torch import einsum
from torch import nn

# from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce

def to_tensor(*args):
    return (torch.Tensor(x) for x in args)

class EinConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(EinConv2d, self).__init__(*args, **kwargs)
    def convolution_layer(self, m, f):
        m_h, m_w = m.shape[-2:]
        f_h, f_w = f.shape[-2:]
        batch_size = m.shape[0]
        m_c = m.shape[1]
        Hout = m_h - f_h + 1
        Wout = m_w - f_w + 1
        stride_batch_size, stride_c, stride_h, stride_w = m.stride()
        m_strided = as_strided(
            m, 
            (batch_size, Hout, Wout, m_c, f_h, f_w), 
            (stride_batch_size, stride_h, stride_w, stride_c, stride_h, stride_w)
        )
        result = einsum('bmncuv,kcuv->bkmn', m_strided, f)
        return result
    def forward(self, input):
        x = input
        # x = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        result = self.convolution_layer(x, self.weight)
        return result