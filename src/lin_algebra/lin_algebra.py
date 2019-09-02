import numpy as np
import pandas as pd
import math
from numpy.linalg import svd

def eigen_decomposition(A):
    """
        Does eigen decomposition of the given matrix M.
        M = V diag(λ) V-1
        Return V, diag(λ) and V-1
    """
    return None

def calculate_magnitude(v):
    return math.sqrt(np.sum(v**2))

def normalize_vector(v):
    return v/calculate_magnitude(v)

def dot_product(v1,v2):
    return np.sum(v1*v2)

def cosine_similarity(v1,v2):
    mag_v1 = calculate_magnitude(v1)
    mag_v2 = calculate_magnitude(v2)

    return dot_product(v1,v2)/(mag_v1*mag_v2)

def remove_vector_projection(v,projection_vec):
    """
        Returns component of v that is perpendicular to projection_vec
        result : cos(theta)*|v|*projection_vec
    """
    return cosine_similarity(v,projection_vec)*calculate_magnitude(v)*projection_vec

def gram_schmidt_process(M):
    """
        Run gram-schmidt on the columns of matrix M.
        Returns: another matrix M
    """
    n = M.shape[0]
    m = M.shape[1]

    orthogonal_M = np.zeros((n,m))
    orthogonal_M[:,0] = normalize_vector(M[:,0])

    for i in range(1,m):
        v = M[:,i]
        projection_vectors_sum = np.zeros(v.shape)
        for j in range(0,i):
            u = orthogonal_M[:,j]
            projection_vectors_sum += remove_vector_projection(v,u)

        v = v - projection_vectors_sum
        v = normalize_vector(v)
        orthogonal_M[:,i] = v

    return orthogonal_M

def decompose_QR(M):
    q = gram_schmidt_process(M)
    r = np.dot(q.transpose(), M)
    return (q,r)

def pca(M, k):
    """
        Principle components are eigenvectors of MT*M = V in SVD
        Select k principle components from V ( call it Y ) and return X*Y 
    """
    w, V = np.linalg.eig(np.dot(np.transpose(M),M))
    #_, _, VT = svd(M)
    #V = np.transpose(VT)
    Y = V[:,:k]
    return np.dot(M,Y)