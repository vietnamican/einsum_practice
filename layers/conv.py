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
    def forward(self, x):
        # x = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        result = self.convolution_layer(x, self.weight)
        return result

class PureEinConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size):
        super(PureEinConv2d, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.kernel_size = kernel_size
        self.weight = torch.nn.init.xavier_normal_(torch.empty(outplanes, inplanes, kernel_size, kernel_size)).cuda()

    def convolution_layer(self, m):
        f = self.weight
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
    def forward(self, x):
        result = self.convolution_layer(x)
        return result