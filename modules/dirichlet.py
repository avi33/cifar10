import torch
import torch.nn as nn
from torch.distributions.dirichlet import Dirichlet
from torch.special import digamma, polygamma
import numpy as np


class EstDirichlet(nn.Module):
    def __init__(self, n_classes=10, n_iters=100, tol=1e-3) -> None:
        super().__init__()
        self.a = torch.ones(n_classes)
        self.n_iters = n_iters
        self.tol = tol

    def forward(self, x):
        x = x.softmax(-1)
        with torch.no_grad():
            n = x.shape[0]        
            self.a = x.mean(0) * ((x.mean(0)-(x**2).mean(0)) / ((x**2).mean(0) - x.mean(0)**2)).mean(-1)
            for k in range(self.n_iters):            
                log_p_avg = torch.log(x).mean(0)
                g = (digamma(self.a.sum(-1)) - digamma(self.a) + log_p_avg) * n
                q = -n * polygamma(n=1, input=self.a)
                z = n * polygamma(n=1, input=self.a.sum(-1))
                q_inv = q**(-1)
                z_inv = 1/z
                # H_inv = q_inv - (q_inv**2).sum(-1) / (z_inv * q_inv.sum(-1))
                b = (g * q_inv).sum(-1) / (z_inv + q_inv.sum(-1))
                H_inv_g = (g - b) / q
                a = self.a - H_inv_g
                if torch.norm(a-self.a, 1) < self.tol:
                    break            
                self.a = a
        return a.float()


if __name__ == "__main__":    
    alpha = [10*0.5, 10*0.5, 0.5]
    x = torch.from_numpy(np.random.dirichlet(alpha, size=1000)).requires_grad_(True)
    # m = Dirichlet(torch.tensor())
    # x = m.sample(10)
    ED = EstDirichlet(x.shape[-1])
    y = ED(x)
    # loss = torch.nn.functional.mse_loss(y, torch.from_numpy(alpha))
    # loss.backward()
    print(y.grad_fn)
    print(x) 