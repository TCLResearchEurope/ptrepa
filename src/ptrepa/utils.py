from __future__ import annotations

import functools
from collections import OrderedDict
from typing import Optional, Union

import torch
from torch.nn.common_types import _size_2_t

__all__ = [
    "SEBlock",
    "fuse_bn_tensor",
    "get_module",
    "to_size_2_t",
    "get_module",
    "make_conv_bn",
    "make_conv_bn_sequential",
    "pad_1x1_to_kxk",
    "set_1x1_conv_to_identity",
]


def get_module(module: torch.nn.Module, target: str) -> torch.nn.Module:
    # TODO Perhaps use pthelpers.get_module
    names = target.split(sep=".")
    return functools.reduce(getattr, names, module)


def to_size_2_t(t: Union[str, tuple[int, ...]]) -> tuple[int, int]:
    # This function is just to to make typechecker happy
    assert isinstance(t, tuple), len(t) == 2
    return t[0], t[1]


def pad_1x1_to_kxk(kernel1x1: torch.Tensor, k: int) -> torch.Tensor:
    padding = k // 2
    return torch.nn.functional.pad(kernel1x1, [padding, padding, padding, padding])


def set_1x1_conv_to_identity(conv: torch.nn.Conv2d) -> None:
    assert conv.kernel_size == (1, 1)
    assert conv.bias is None

    if conv.groups == 1:
        weight = torch.zeros(
            conv.out_channels, conv.in_channels, 1, 1, dtype=torch.float32
        )

        for i in range(min(conv.in_channels, conv.out_channels)):
            weight[i, i, 0, 0] = 1.0

    elif conv.groups == conv.in_channels and conv.out_channels == conv.in_channels:
        weight = torch.zeros(conv.out_channels, 1, 1, 1, dtype=torch.float32)

        for i in range(min(conv.in_channels, conv.out_channels)):
            weight[i, 0, 0, 0] = 1.0

    else:
        raise ValueError(
            f"Unsupported case "
            f"{conv.in_channels=}, {conv.out_channels=}, {conv.groups=}"
        )
    conv.weight.data = weight.to(conv.weight.device)


def make_conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: _size_2_t,
    stride: _size_2_t = 1,
    padding: _size_2_t = 0,
    use_bias: bool = False,
    groups: int = 1,
    dilation: _size_2_t = 1,
) -> tuple[torch.nn.Conv2d, torch.nn.BatchNorm2d]:
    conv = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation,
        bias=use_bias,
    )
    bn = torch.nn.BatchNorm2d(num_features=out_channels)
    return conv, bn


def make_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: _size_2_t,
    stride: _size_2_t = 1,
    padding: _size_2_t = 0,
    use_bias: bool = False,
    groups: int = 1,
    dilation: _size_2_t = 1,
) -> torch.nn.Conv2d:
    conv = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation,
        bias=use_bias,
    )
    return conv


def make_conv_bn_sequential(
    in_channels: int,
    out_channels: int,
    kernel_size: _size_2_t,
    stride: _size_2_t,
    padding: _size_2_t,
    groups: int = 1,
    dilation: _size_2_t = 1,
) -> torch.nn.Sequential:
    conv, bn = make_conv_bn(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation,
    )
    return torch.nn.Sequential(OrderedDict(conv=conv, bn=bn))


def identity_to_conv(
    in_channels: int,
    kernel_size: tuple[int, ...],
    groups: int,
) -> torch.Tensor:
    input_dim = in_channels // groups
    kernel = torch.zeros(
        (in_channels, input_dim, kernel_size[0], kernel_size[1]),
        dtype=torch.float32,
    )
    for i in range(in_channels):
        kernel[i, i % input_dim, kernel_size[0] // 2, kernel_size[1] // 2] = 1
    return kernel


def fuse_bn_tensor(
    conv: Optional[torch.nn.Conv2d],
    bn: torch.nn.BatchNorm2d,
    in_channels: Optional[int] = None,
    kernel_size: Optional[tuple[int, ...]] = None,
    groups: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if conv is not None:
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
    elif in_channels is not None and groups is not None and kernel_size is not None:
        kernel = identity_to_conv(
            in_channels=in_channels, kernel_size=kernel_size, groups=groups
        )
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
    else:
        raise NotImplementedError
    # Assert to make typechecker happy
    assert isinstance(running_var, torch.Tensor)
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class SEBlock(torch.nn.Module):
    """Squeeze and Excite module.

    Pytorch implementation of `Squeeze-and-Excitation Networks` -
    https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        """Construct a Squeeze and Excite Module.

        :param in_channels: Number of input channels.
        :param rd_ratio: Input channel reduction ratio.
        """
        super().__init__()
        self.reduce = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.expand = torch.nn.Conv2d(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, c, h, w = inputs.size()
        x = torch.nn.functional.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = torch.nn.functional.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x
