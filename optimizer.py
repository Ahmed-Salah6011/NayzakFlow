import numpy as np
import nayzakflow as nf

class GD():
    def __init__(self, lr=0.01):
        self.alpha= lr

    
    def update(self,layers,N):
        for i in range(len(layers)):
            # layers[i].dW=layers[i].dW/N
            # layers[i].db=layers[i].db/N
            layers[i].W = layers[i].W - self.alpha * (layers[i].dW/N)
            layers[i].b = layers[i].b - self.alpha * (layers[i].db/N)
            # print("delta: ")
            # print("dW",layers[i].dW)
            # print("db",layers[i].db)
            # print("Weights: ")
            # print("W",layers[i].W)
            # print("b",layers[i].b)

            

