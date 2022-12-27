import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union
from transformer_encoder import Encoder

def count_parameters(model):    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation*(3//2),
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        # if dilation > 1:
        #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation, groups=groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class AADownsample(nn.Module):
    def __init__(self, filt_size=3, stride=2, channels=None):
        super(AADownsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        ha = torch.arange(1, filt_size//2+1+1, 1)
        a = torch.cat((ha, ha.flip(dims=[-1,])[1:])).float()
        a = a / a.sum()
        filt = a[:, None] * a[None, :]
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, x):
        x_pad = F.pad(x, [self.filt_size//2]*4, "reflect")
        y = F.conv2d(x_pad, self.filt, stride=self.stride, padding=0, groups=x.shape[1])
        return y


class Down(nn.Module):
    def __init__(self, channels, d=2, k=3):
        super().__init__()
        kk = d + 1
        self.down = nn.Sequential(
            nn.ReflectionPad2d(kk // 2),
            nn.Conv2d(channels, channels*2, kernel_size=kk, stride=1, bias=False),
            nn.BatchNorm2d(channels*2),
            nn.LeakyReLU(0.2, True),
            AADownsample(channels=channels*2, stride=d, filt_size=k)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class TFEncoder(nn.Module):
    def __init__(self, d_emb=256, seq_len=8) -> None:
        super().__init__()                
        #tf_enc = nn.TransformerEncoderLayer(d_model=d_emb, nhead=1, dim_feedforward=d_emb*4, dropout=0.1, activation=F.relu)
        #self.tf = nn.TransformerEncoder(tf_enc, num_layers=4, norm=None)
        self.tf = Encoder(num_heads=4, num_layers=2, d_model=d_emb, ff_hidden_dim=d_emb*4, p=0.1)
        self.seq_len = seq_len

    def forward(self, x):
        x = x.permute(2, 0, 1).contiguous()
        y = self.tf(x)        
        o = y[:self.seq_len, ...].contiguous()
        o = o.permute(1, 2, 0).contiguous()
        return o

class Net(nn.Module):
    def __init__(self, d_emb=128, n_classes=10) -> None:
        super().__init__()
        nf = 16
        self.d_emb = d_emb
        backbone = [
            nn.ReflectionPad2d(5//2),
            nn.Conv2d(3, nf, 5, 1, 0, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True)
        ]
        for k in range(3):            
            backbone += [Down(nf, d=2, k=3)]
            nf *= 2
            backbone += [ResBlock(inplanes=nf, planes=nf, dilation=1)]
            backbone += [ResBlock(inplanes=nf, planes=nf, dilation=3)]

        self.backbone = nn.Sequential(*backbone)
        self.pre_tf = nn.Sequential(
            nn.Conv2d(d_emb, d_emb, 1, 1, bias=False),
            nn.BatchNorm2d(d_emb),
            nn.LeakyReLU(0.2, True)
        )
        self.tf = TFEncoder(d_emb=d_emb)
        self.project = nn.Conv2d(d_emb, n_classes, 1, 1)

    def forward(self, x):
        x = self.backbone(x)
        y = x.view(x.shape[0], self.d_emb, x.shape[2]*x.shape[3])
        y = self.pre_tf(y.unsqueeze(3).contiguous())
        y = self.tf(y.squeeze(3))
        y = self.project(y.unsqueeze(3).contiguous()).squeeze(3).contiguous()
        return y

    
if __name__ == "__main__":
    N = Net(n_classes=30, d_emb=128)
    N.eval()
    x = torch.randn(2, 3, 32, 128)
    y = N(x)
    print(y.shape)    
    print(count_parameters(N) / 1e6)
    x = torch.randn(1, 3, 32, 128, requires_grad=True)
    torch_out = N(x)

    # Export the model
    torch.onnx.export(N,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    "net_prod.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=16,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

    import onnx
    onnx_model = onnx.load("net_prod.onnx")
    onnx.checker.check_model(onnx_model)
    import onnxruntime
    import numpy as np

    ort_session = onnxruntime.InferenceSession("net_prod.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print(ort_outs[0].shape)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
