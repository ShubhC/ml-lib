import numpy as np

def tensor_multiply_3d(a,b):
    return np.einsum('mnr,ndr->mdr', a, b)
