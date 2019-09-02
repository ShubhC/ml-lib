import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class NonLinear(ABC):
    @abstractmethod
    def a(self,x):
        pass

    @abstractmethod
    def derivative_at(self, x):
        pass

class Sigmoid(NonLinear):
    """
        a(x)  = 1/(1+exp(-x))
        a'(x) = a(x) *( 1-a(x) )
    """
    def a(self, x):
        return 1/(1+np.exp(-x))

    def derivative_at(self, x):
        return self.a(x)*(1-self.a(x))

class Relu(NonLinear):
    """
        a(x)  = max( 0, x )
        a'(x) = 0 if x < 0 otherwise 1
    """

    def a(self, x):
        return np.where(x>0., x, 0.)

    def derivative_at(self, x):
        return np.where(x>0., 1., 0.)