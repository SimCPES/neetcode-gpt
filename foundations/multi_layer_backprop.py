import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        
        z1 = W1@x+b1
        a1 = np.maximum(0, z1)
        z2 = W2@a1+b2

        d = z2 - y_true
        n = len(y_true)

        res = {"loss": np.round(np.mean(d**2), 4), "dW2": np.expand_dims(np.round(2*np.mean(d)*a1, 4), axis=0), "db2": np.expand_dims(np.round(2*np.mean(d), 4), axis=0)}

        print(np.shape(W2))
        print(np.shape(np.where(z1>0, z1, 0)))
        print(np.shape(np.expand_dims(x, axis=1)))

        res["dW1"] = np.round(2*np.mean(d)*np.transpose(W2*np.where(z1>0, 1, 0))@np.expand_dims(x, axis=0), 4)
        res["db1"] = np.squeeze(np.round(2*np.mean(d)*W2*np.where(z1>0, 1, 0), 4), axis=0)

        return res




