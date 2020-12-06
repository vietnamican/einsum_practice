import numpy as np
from torch import as_strided
from torch import einsum
import torch
from torch import nn

from einops.layers.torch import Reduce
from einops import reduce

def to_tensor(*args):
    return (torch.Tensor(x) for x in args)



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



# if __name__ == '__main__':
#     tensor = torch.arange(36).reshape(1, 1, 6, 6)
#     filters = torch.rand(10, 1, 5, 5)
#     result1 = max_pool2d_layer(tensor)
#     result2 = max_pool2d_layer_numpy(tensor)
#     print(tensor)
#     print(result1.shape, result2.shape)
#     print(result1)
#     print(result2)
#     print(np.all(np.isclose(result1, result2)))