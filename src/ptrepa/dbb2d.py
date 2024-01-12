from __future__ import annotations

import torch
from torch.nn.common_types import _size_2_t

__all__ = ["DBBConv2d", "fuse_dbb2d", "make_dbb2d_from_conv2d"]


class DBBConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 1,
        groups: int = 1,
    ):
        super().__init__()


def fuse_dbb2d(m: torch.nn.Module) -> torch.nn.Conv2d:
    raise NotImplementedError("Not implemented")


def make_dbb2d_from_conv2d(m: torch.nn.Conv2d) -> torch.nn.Module:
    raise NotImplementedError("Not implemented")
