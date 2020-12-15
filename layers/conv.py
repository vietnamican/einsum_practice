import torch
from torch import as_strided
from torch import einsum
from torch import nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from einops import rearrange, reduce, asnumpy, parse_shape, repeat
from einops.layers.torch import Rearrange, Reduce


def to_tensor(*args):
    return (torch.Tensor(x) for x in args)


class EinConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(EinConv2d, self).__init__(*args, **kwargs)

    def convolution_layer(self, m):
        f = self.weight
        b = self.bias
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
        b_repeated = repeat(b, 'k->b k m n', b=batch_size, m=Hout, n=Wout)
        result += b_repeated
        return result

    def forward(self, x):
        # x = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        result = self.convolution_layer(x)
        return result


class PureEinConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size):
        super(PureEinConv2d, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.kernel_size = kernel_size
        weight = nn.init.xavier_normal_(torch.empty(
            outplanes, inplanes, kernel_size, kernel_size)).to(device)
        bias = nn.init.zeros_(torch.empty(outplanes))
        self.register_parameter('weight', nn.Parameter(weight))
        self.register_parameter('bias', nn.Parameter(bias))

    def convolution_layer(self, m):
        f = self.weight
        b = self.bias
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
        b_repeated = repeat(b, 'k->b k m n', b=batch_size, m=Hout, n=Wout)
        result += b_repeated

        return result

    def forward(self, x):
        result = self.convolution_layer(x)
        return result
