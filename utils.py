import numpy as np
from torch import as_strided
from torch import einsum
import torch
from torch import nn

def to_tensor(*args):
    return (torch.Tensor(x) for x in args)

def convolution_layer(m ,f):
    m, f = to_tensor(m, f)
    m_h, m_w = m.shape[-2:]
    f_h, f_w = f.shape[-2:]
    batch_size = m.shape[0]
    m_c = m.shape[1]
    Hout = m_h - f_h + 1
    Wout = m_w - f_w + 1
    m_strided = as_strided(
        m, 
        (batch_size, Hout, Wout, m_c, f_h, f_w), 
        (batch_size, m_h, m_w, m_c, m_h, m_w)
    )
    m_strided, f = to_tensor(m_strided, f)
    result = einsum('bmncuv,kcuv->bkmn', m_strided, f)
    return result

class EinConv2D(Conv2D):
    def __init__(self, *args, *kwargs):
        super(EinConv2D, self).__init__(*args, *kwargs)
    def forward(self, input):
        input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        return convolution_layer(input, self.weight)

if __name__ == '__main__':
    tensor = np.random.rand(100, 1, 28, 28)
    filters = np.random.rand(10, 1, 5, 5)
    result = convolution_layer(tensor, filters)
    print(result.shape)