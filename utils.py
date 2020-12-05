import numpy as np
from numpy.lib.stride_tricks import as_strided
from torch import einsum
import torch

def to_tensor(*args):
    return (torch.Tensor(x) for x in args)

def convolution_layer(m ,f):
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
    m_strided, f = to_tensor(m_strided.copy(), f)
    result = einsum('bmncuv,kcuv->bkmn', m_strided, f)
    return result

if __name__ == '__main__':
    tensor = np.arange(18).reshape(2, 3, 3)
    batch_tensor = as_strided(tensor, (2, *tensor.shape), (0, *tensor.strides))
    filters = np.arange(16).reshape(2, 2, 2, 2)
    result = convolution_layer(batch_tensor, fil)
    print(result)