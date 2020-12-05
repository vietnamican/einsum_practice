import torch
from torch import einsum
import numpy as np

def to_tensor(*args):
    return (torch.Tensor(x) for x in args)

def matrix_transpose(m):
    m = to_tensor(m)
    result = einsum('ij->ji', m)
    return result

def matrix_sum(m):
    m = to_tensor(m)
    result = einsum('ij->', m)
    return result

def matrix_column_sum(m):
    m = to_tensor(m)
    result = einsum('ij->j', m)
    return result

def matrix_row_sum(m):
    m = to_tensor(m)
    result = einsum('ij->i', m)
    return result

def matrix_vector_multiply(m, v):
    m, v = to_tensor(m, v)
    result = einsum('ik,k->i', m, v)
    return result

def matrix_multiply(m, n):
    m, n = to_tensor(m, n)
    result = einsum('ij,jk->ik', m, n)
    return result

def dot_product(a, b):
    m, n = to_tensor(m, n)
    result = einsum('i,i->', a, b)
    return result

def matrix_dot_product(m ,n):
    m, n = to_tensor(m, n)
    result = einsum('ij,ij->', m, n)
    return result

def matrix_hadamard_product(m, n):
    m, n = to_tensor(m, n)
    result = einsum('ij,ij->ij', m, n)
    return result

def outer_product(m, n):
    m, n = to_tensor(m, n)
    result = einsum('i,j->ij', m, n)
    return result

def batch_matrix_multiply(m, n):
    m, n = to_tensor(m, n)
    result = einsum('bij,bjk->bik', m, n)
    return result
