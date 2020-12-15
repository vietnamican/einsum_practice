import torch
from torch import as_strided
from torch import einsum
from torch import nn

# from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, asnumpy, parse_shape, repeat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def to_tensor(*args):
    return (torch.Tensor(x) for x in args)


class EinConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(EinConv2d, self).__init__(*args, **kwargs)

    def convolution(self, m):
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

    def add_bias(self, x):
        batch_size, _, m, n = x.shape
        b = self.bias
        b_repeated = repeat(b, 'k->b k m n', b=batch_size, m=m, n=n)
        return x + b_repeated

    def forward(self, x):
        # x = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        result = self.convolution(x)
        if self.bias is not None:
            result = self.add_bias(result)
        return result


class PureEinConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, bias=True):
        super(PureEinConv2d, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.kernel_size = kernel_size
        weight = nn.init.xavier_normal_(torch.empty(
            outplanes, inplanes, kernel_size, kernel_size)).to(device)
        self.register_parameter('weight', nn.Parameter(weight))
        if bias:
            _bias = nn.init.zeros_(torch.empty(outplanes))
            self.register_parameter('bias', nn.Parameter(_bias))

    def convolution(self, m):
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

    def add_bias(self, x):
        batch_size, _, m, n = x.shape
        b = self.bias
        b_repeated = repeat(b, 'k->b k m n', b=batch_size, m=m, n=n)
        return x + b_repeated

    def forward(self, x):
        result = self.convolution(x)
        if self.bias is not None:
            result = self.add_bias(result)
        return result
