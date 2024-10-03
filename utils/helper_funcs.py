import os
import torch
import numpy as np
import copy

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, input, repetitions=300, use_16b=False):
    device = torch.device("cuda")
    model_= copy.deepcopy(model)
    model_.eval()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    # repetitions = 300
    timings = np.zeros((repetitions, 1))
    print(input.shape)
    if use_16b:
        input = input.half()
        model_.half()
    else:
        pass
    input = input.to(device)
    model_.to(device)
    for _ in range(10):
        _ = model_(input)
    with torch.no_grad():
        # GPU-WARM-UP
        for rep in range(repetitions):
            starter.record()
            _ = model_(input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    return mean_syn, std_syn

def add_weight_decay(model, weight_decay=1e-5, skip_list=(), additional=None):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        # print(name)
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    if additional is None:
        parameters = [{'params': no_decay, 'weight_decay': 0.},
                    {'params': decay, 'weight_decay': weight_decay}]
    else:
        parameters = [{'params': no_decay.append(additional), 'weight_decay': 0.},
                    {'params': decay, 'weight_decay': weight_decay}]
    return parameters

def save_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)

def load_model(model, model_filepath, device):
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    return model

def save_torchscript_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)

def load_torchscript_model(model_filepath, device):
    model = torch.jit.load(model_filepath, map_location=device)
    return model

def model_equivalence(model_1, model_2, input_size, device, rtol=1e-05, atol=1e-08, num_tests=100):
    model_1.to(device)
    model_2.to(device)
    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False
    return True

def check_receptivefield(net, x):    
    x.requires_grad_(True)
    y = net.cnn(x)
    if len(y.shape) == 4:
        _, c, w, h = y.shape
        grad = y[:, :, w//2, h//2].abs().sum().backward()
        idx = torch.nonzero(x.grad > 0)
        i1 = max(idx[:, -2])-min(idx[:, -2]) + 1
        i2 = max(idx[:, -1])-min(idx[:, -1]) + 1
        rf = (i1, i2)
    elif len(y.shape) == 3:
        grad = y[:, :, w//2].abs().sum().backward()
        idx = torch.nonzero(x.grad > 0)
        i1 = max(idx[:, -1])-min(idx[:, -1]) + 1
        rf = i1
    return rf