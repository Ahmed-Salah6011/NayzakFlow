import numpy as np
import nayzakflow as nf

class GD():
    def __init__(self, lr=0.01):
        self.alpha= lr

    
    def update(self,layers):
        for i in range(len(layers)):
            layers[i].W = layers[i].W - self.alpha * layers[i].dW
            layers[i].b = layers[i].b - self.alpha * layers[i].db

