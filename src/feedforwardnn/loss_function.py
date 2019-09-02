from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import pandas as pd
import math

class LossFunction(ABC):
    def __init__(self):
        self.y_actuals = None
        self.y_predicted = None

    @abstractmethod
    def loss(self, y_actuals, y_predicted):
        pass

    @abstractmethod
    def derivative(self):
        pass

class BCE(LossFunction):
    """
        BinaryCrossEntropy loss function
        loss = -1 * SUM( actuals*log(pred) + (1-actuals)*log(1-preds) ) / N
        derivative = -1 * SUM( actuals/pred + (1-actuals)/(1-preds)*-1 ) / N        
    """
    def loss(self, y_actuals, y_predicted):
        self.eps = 1e-7

        self.y_actuals = y_actuals.reshape(y_predicted.shape)
        self.y_predicted = y_predicted

        self._y_actuals_ones = y_actuals == 1
        self._y_actuals_zeros = y_actuals == 0

        loss = np.zeros(self.y_predicted.shape)
        loss[self._y_actuals_ones] = np.log(self.eps + self.y_predicted[self._y_actuals_ones])
        loss[self._y_actuals_zeros] = np.log(self.eps + 1.-self.y_predicted[self._y_actuals_zeros])
        loss = -1.*np.sum(loss)/self.y_actuals.shape[-1]

        return loss

    def derivative(self):
        grads = np.zeros(self.y_predicted.shape)
        grads[self._y_actuals_ones] =  1./(self.eps + self.y_predicted[self._y_actuals_ones] )
        grads[self._y_actuals_zeros] = -1./(self.eps + 1. - self.y_predicted[self._y_actuals_zeros] )
        grads /= -1.*self.y_actuals.shape[-1]
        return grads

class MeanSquaredError(LossFunction):
    """
        Mean Squared Error loss function.

        loss = SUM( (y_actuals - y_predictions)**2 ) / N
        derivative = SUM( (y_actuals - y_predictions) ) / N
    """
    def loss(self, y_actuals, y_predicted):
        self.y_actuals = y_actuals
        self.y_predicted = y_predicted

        loss = np.sum((y_actuals-y_predicted)**2)/self.y_actuals.shape[-1]

        return loss

    def derivative(self):
        grads = (self.y_predicted-self.y_actuals)/self.y_actuals.shape[-1]
        grads = np.reshape(grads, (1,1,self.y_actuals.shape[-1]) )
        return grads