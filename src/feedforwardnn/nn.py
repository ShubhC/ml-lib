from layer import *
from activation_function import *
from loss_function import *
import numpy as np
import math
from data_loader import *

class FeedforwardNeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        x_intermediate = x

        for i, layer in enumerate(self.layers):
            x_intermediate = layer.forward(x_intermediate)

        y_predicted = x_intermediate

        return y_predicted

    def backward(self, grad):
        for i in range(len(self.layers)):
            layer = layers[len(self.layers)-1-i]
            grad = layer.backward(grad)


lr = 1e-1
layers = [Linear(2,2,lr),
          Activation(Relu()),
          Linear(2,2,lr),
          Activation(Relu()),
          Linear(2,1,lr),
          Activation(Sigmoid())]

loss_function = BCE()

model = FeedforwardNeuralNetwork(layers)

train_x = np.array([[1,0],[0,1],[0,0],[1,1]])
train_y = np.array([1,0,0,1])

# train on full batch
dataloader = DataLoader(train_x, train_y, train_x.shape[0])

eps = 1e-9
for epochs in range(0,400):

    for (i, (x,y)) in enumerate(dataloader):
        # forward pass
        yp = model.forward(x)
 
        # calculate loss
        loss = loss_function.loss(y, yp)
        grad = loss_function.derivative()

        # backprop
        model.backward(grad)

    if epochs%100 == 0:
        print(str.format("Iteration: {0}, loss: {1}", epochs, loss))

    if loss < eps:
        print(str.format('Loss less than : {0}. Stopping training...',eps))

print('Predictions ' + str(yp))
print('Actuals ' + str(y))
print(str.format("Final Loss: {0}",loss))
