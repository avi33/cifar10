import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class STGLayer(nn.Module):
    def __init__(self, input_dim, sigma=0.5):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(input_dim))
        self.sigma = sigma
        self.normal = Normal(0, sigma)

    def forward(self, x):
        eps = self.normal.sample(self.mu.shape).to(x.device)
        z = (self.mu + eps).clamp(0.,1.)
        # For regularization: use CDF of Gaussian
        reg = torch.distributions.Normal(0,1).cdf(self.mu / self.sigma).sum()
        return x * z, reg


class STGModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, sigma=0.5):
        super().__init__()
        self.stg = STGLayer(input_dim, sigma)
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y=None, lam=0.1):
        xg, reg = self.stg(x)
        out = self.net(xg)
        return out, lam * reg


if __name__ == "__main__":
    # Example usage
    input_dim = 10
    hidden_dims = [20, 15]
    output_dim = 1
    model = STGModel(input_dim, hidden_dims, output_dim)

    x = torch.randn(5, input_dim)  # Batch of 5 samples
    out, reg = model(x)
    print("Output:", out)
    print("Regularization term:", reg)
    # Example usage
    y = torch.randn(5, output_dim)  # Dummy target
    loss_fn = nn.MSELoss()
    loss = loss_fn(out, y) + reg
    print("Loss:", loss.item())
    # Backward pass
    loss.backward()
    print("Gradients:", model.stg.mu.grad)