import torch
import torch.nn as nn

class GatedParameter(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.w = nn.Parameter(param.data.clone())
        self.gate_logit = nn.Parameter(torch.zeros_like(param))  # one gate per param

    def forward(self):
        prob = torch.sigmoid(self.gate_logit)
        gate = torch.bernoulli(prob)
        gate = gate + prob - prob.detach()  # straight-through estimator
        return self.w * gate
    

def add_gates_to_model(model):
    gated_params = nn.ModuleDict()
    for name, param in list(model.named_parameters()):
        # Remove from model's parameter list
        delattr(model, name.replace('.', '_'))  # remove from attribute space (if needed)
        gated_param = GatedParameter(param)
        gated_params[name] = gated_param
    model.gated_params = gated_params
    return model


