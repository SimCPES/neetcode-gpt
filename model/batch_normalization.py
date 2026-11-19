import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        if not training:
            x_hat = (np.array(x) - np.array(running_mean))/np.sqrt(np.array(running_var)+eps)
            y = np.round(np.array(gamma) * x_hat + np.array(beta), 4)

            return (y.tolist(), running_mean, running_var)

        else:
            mu = np.mean(np.array(x), axis=0)
            sigma = np.var(np.array(x), axis=0)
            x_hat = (np.array(x) - mu)/np.sqrt(sigma+eps)
            y = np.round(np.array(gamma)*x_hat + np.array(beta), 4)

            running_mean = np.round((1 - momentum) * np.array(running_mean) + momentum * mu, 4)
            running_var = np.round((1 - momentum) * np.array(running_var) + momentum * sigma, 4)

            return (y.tolist(), running_mean.tolist(), running_var.tolist())
