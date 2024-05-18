import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from activation_function import NonLinear

class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, gradients):
        pass

class Activation(Layer):
    def __init__(self, activation_function: NonLinear):
        self.activation_function = activation_function
        self.x = None

    def forward(self, x):
        self.x = x
        return self.activation_function.a(x)

    def backward(self, gradients):
        derivative = self.activation_function.derivative_at(self.x)
        return derivative*gradients

class Linear(Layer):
    def __init__(self, input_size: int, output_size: int, lr=5e-1):
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.rand(self.output_size, self.input_size)
        self.b = np.random.rand(self.output_size, 1)
        self.x = None
        self.lr = lr

    def _tensor_multiply_3d(self,a,b):
        if len(a.shape) == 2:
            return np.einsum('pq,qsr->psr',a,b)
        
        if len(b.shape) == 2:
            return np.einsum('pqr,qs->psr',a,b)

        return np.einsum('mnr,ndr->mdr', a, b)

    def forward(self, x):
        self.x = x
        Wx = self._tensor_multiply_3d(self.W, self.x)
        b = np.reshape(self.b, self.b.shape + (1,))
        return Wx + b

    def _update_params(self, gradients):
        x_transpose = np.transpose(self.x, (1,0,2))
        W_gradients = self._tensor_multiply_3d(gradients, x_transpose)
        
        # add along axis 3
        W_gradients = np.sum(W_gradients, axis=2)
        
        b_gradients = np.sum(gradients, axis=2)

        self.W = self.W - self.lr*W_gradients
        self.b = self.b - self.lr*b_gradients

    def backward(self, gradients):
        self._update_params(gradients)
        return self._tensor_multiply_3d( np.transpose(self.W), gradients )


class Embedding(Layer):
    def __init__(self, 
                 embedding_dim: int,
                 vocabulary_size: int,
                 lr: int = 1e-5):
        self.embedding_dim = embedding_dim
        self.vocabulary_size = vocabulary_size

        self.embedding_table = np.random.rand((self.vocabulary_size, 
                                               self.embedding_dim))
        self.lr = lr
        
    def forward(self, 
                x: np.array):
        """
            x is numpy array of shape (indices, ) or (1, batch)
            Does a embeddings lookup for the given indices
        """
        if len(x.shape) == 2:
            assert x.shape[1] == 1
            self.x = np.reshape(x, (-1))

        return self.embedding_table[self.x]

    def backward(self, 
                 gradients: np.array):
        """
            gradients : numpy array of shape (embedding_dim, 1, batch_size) 
                        Represents partial derivative of model output with respect to emb layer output

            No return value as embedding layer is usually an input layer 
        """
        
        # Update weights
        # W = W - lr * dy/dW

        # Gradient: (d, 1, b)

        idxs, counts = np.unique(self.x, 
                                 return_counts=True)

        self.embedding_table[idxs] -= self.lr * gradients[:,:,idxs] * counts

        return None