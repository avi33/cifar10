import torch
import torch.nn as nn
from torch.distributions.dirichlet import Dirichlet
from torch.special import digamma, polygamma
import numpy as np

def digamma_approx(x):
    return torch.log(x) - 1 / (2 * x) - 1 / (12 * x**2)

def trigamma_approx(x):
    return 1 / x + 1 / (2 * x**2) + 1 / (6 * x**3)

class EstDirichlet(nn.Module):
    def __init__(self, n_classes=10, n_iters=100, tol=1e-3) -> None:
        super().__init__()
        self.a = torch.ones(n_classes)
        self.n_iters = n_iters
        self.tol = tol

    def forward(self, x):
        x = x.softmax(-1)
        eps = 1e-10  # numerical stability
        x = x.clamp(min=eps, max=1 - eps)

        N = x.shape[0]
        log_p = torch.log(x)
        log_p_mean = log_p.mean(0)

        # Method-of-moments initialization
        mean = x.mean(0)
        var = x.var(0, unbiased=False)
        mean_log = log_p_mean
        alpha0_init = (mean * (1 - mean) / (var + eps) - 1).clamp(min=1.01)
        alpha = alpha0_init.clone().detach()

        for i in range(self.n_iters):
            alpha_sum = alpha.sum()
            g = N * (digamma(alpha_sum) - digamma(alpha) + log_p_mean)
            q = -N * polygamma(1, alpha)
            z = -N * polygamma(1, alpha_sum)
            b = (g / q).sum() / (1.0 / z + (1.0 / q).sum())
            delta = (g - b) / q
            alpha_new = alpha - delta

            if (alpha_new <= 0).any():  # keep alpha positive
                alpha_new = alpha.clamp(min=1e-3)

            if torch.norm(alpha_new - alpha, p=1) < self.tol:
                break

            alpha = alpha_new
        return alpha.float()


if __name__ == "__main__":    
    # alpha = [10, 10, 10]
    # x = torch.from_numpy(np.random.dirichlet(alpha, size=1000)).reshape(-1)#.requires_grad_(True)
    # # x = torch.rand(3, requires_grad=True)
    # # m = Dirichlet(torch.tensor())
    # # x = m.sample(10)
    # ED = EstDirichlet(x.shape[-1])
    # # ED.train()
    # y = ED(x)
    # print(y)
    # m = Dirichlet(y)
    # for i in range(10):    
    #     z = m.sample(3)
    #     print(z)

    # # loss = torch.nn.functional.mse_loss(y, torch.from_numpy(alpha))
    # # loss.backward()
    # print("requires_grad:", y.requires_grad)
    # print("grad_fn:", y.grad_fn)

    # loss = y.sum()
    # loss.backward()
    # print(x.grad)
    # for name, param in ED.named_parameters():
    #     print(name, param.grad is not None)
    true_alpha = torch.tensor([5.0, 1.0, 8])  # K=3
    N = 500  # number of samples

    dirichlet = Dirichlet(true_alpha)
    samples = dirichlet.sample((N,))  # shape (N, 3)
    estimator = EstDirichlet(n_classes=3)
    estimated_alpha = estimator(samples)
    estimated_alpha = estimated_alpha * (true_alpha.sum() / estimated_alpha.sum())

    print("True α:", true_alpha)
    print("Estimated α:", estimated_alpha)