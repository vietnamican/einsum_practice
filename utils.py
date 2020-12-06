import numpy as np
from torch import as_strided
from torch import einsum
import torch
from torch import nn

# from einops.layers.torch import Reduce
from einops import reduce

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
    stride_batch_size, stride_c, stride_h, stride_w = m.stride()
    m_strided = as_strided(
        m, 
        (batch_size, Hout, Wout, m_c, f_h, f_w), 
        (stride_batch_size, stride_h, stride_w, stride_c, stride_h, stride_w)
    )
    m_strided, f = to_tensor(m_strided, f)
    result = einsum('bmncuv,kcuv->bkmn', m_strided, f)
    return result

def max_pool2d_layer(x):
    result = reduce(x, 'b c (h h1) (w w1) -> b c h w', 'max', h1=2, w1=2)
    return result

def max_pool2d_layer_numpy(x):
    b, c, h, w = x.shape
    b_strided, c_strided, h_strided, w_strided = x.stride()
    x_strided = as_strided(
        x, 
        (b, c, h // 2, w // 2, 2, 2), 
        (b_strided, c_strided, h_strided * 2, w_strided * 2, h_strided, w_strided)    
    )
    print(x_strided.shape)
    result = torch.amax(x_strided, dim=(-1, -2))
    return result

class EinMaxPool2d(nn.MaxPool2d):
    def __init__(self, *args, **kwargs):
        super(MaxPool2D, self).__init__(*args, **kwargs)
    
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
        print(x_strided.shape)
        result = torch.amax(x_strided, dim=(-1, -2))
        return result

    def forward(self, x):
        result = self.max_pool2d_layer_numpy(x)
        return result
    

class EinConv2D(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(EinConv2D, self).__init__(*args, **kwargs)
    def forward(self, input):
        input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        return convolution_layer(input, self.weight)

if __name__ == '__main__':
    tensor = torch.arange(36).reshape(1, 1, 6, 6)
    filters = torch.rand(10, 1, 5, 5)
    result1 = max_pool2d_layer(tensor)
    result2 = max_pool2d_layer_numpy(tensor)
    print(tensor)
    print(result1.shape, result2.shape)
    print(result1)
    print(result2)
    print(np.all(np.isclose(result1, result2)))