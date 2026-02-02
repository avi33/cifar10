import torch

def heavy_tail_loss(Ws, eps=1e-4):
    """
    Heavy-tail loss for a conv weight tensor.

    Args:
        W: torch.Tensor, shape (out_channels, in_channels, kH, kW)
        eps: stability constant
    Returns:
        torch.Tensor scalar loss
    """
    # Flatten to (out_channels, in_channels * kH * kW)
    loss = 0
    for W in Ws:
        W2d = W.view(W.size(0), -1)

        if W2d.shape[0] == 1:
            W2d = W2d.t().contigueus()
            
        # Gram matrix
        gram = W2d @ W2d.t() + torch.eye(W2d.size(0), device=W.device) * eps

        # Eigenvalues (symmetric PSD matrix)
        eigenvalues = torch.linalg.eigvalsh(gram)

        # Stability clamp
        eigenvalues = eigenvalues.clamp(min=eps)

        lambda_max = eigenvalues[-1]          # largest
        lambda_mean = eigenvalues.mean()      # mean of all

        # Heavy-tail penalty
        loss += -torch.sum(torch.log(-(lambda_max - lambda_mean)))
        
    return loss


def fast_heavy_loss(W, n_iter=5):
    lambda_max = spectral_norm_power(W, n_iter)
    lambda_mean = mean_eigenvalue(W)
    return -(torch.log(lambda_max - lambda_mean))

def mean_eigenvalue(W, eps=1e-4):
    W2d = W.view(W.size(0), -1)
    gram = W2d @ W2d.t()
    lambda_mean = torch.trace(gram) / gram.size(0)
    lambda_mean = lambda_mean.clamp(min=eps)
    return lambda_mean

def spectral_norm_power(W, eps=1e-4, n_iter=5):
    # Flatten conv weights
    W2d = W.view(W.size(0), -1)
    gram = W2d @ W2d.t() + torch.eye(W2d.size(0), device=W.device) * eps

    # Random init vector
    v = torch.rand(gram.size(0), device=W.device)
    v = v / v.norm()

    for _ in range(n_iter):
        v = gram @ v
        v = v / v.norm()

    # Rayleigh quotient ~ largest eigenvalue
    lambda_max = v @ (gram @ v)
    lambda_max = lambda_max.clamp(min=eps)
    return lambda_max