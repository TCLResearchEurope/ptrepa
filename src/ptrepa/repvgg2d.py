from __future__ import annotations

from typing import Optional

import torch
from torch.nn.common_types import _size_2_t

from . import distilling, utils

__all__ = [
    "RepVGGConv2d",
    "distill_repvgg2d_in_place",
    "is_repvgg2d",
    "make_conv2d_from_repvgg2d",
    "make_repvgg2d_from_conv2d",
]


class RepVGGConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Optional[_size_2_t] = None,
        groups: int = 1,
        identity_conv: str = "trainable",
    ):
        if identity_conv not in {"trainable", "fixed", "disable"}:
            msg = f"{identity_conv=} not in (trainable, fixed, disable)"
            raise ValueError(msg)
        super().__init__()

        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
            kernel_size = kernel_size[0]

        if padding is None:
            padding = kernel_size // 2

        if isinstance(padding, tuple):
            assert len(padding) == 2 and padding[0] == padding[1]
            padding = padding[0]

        # Original RepVGG was implemented ONLY for kernel_size = 3 and padding 1
        # This implementation works for any odd ks with fixed padding = kernel_size // 2
        if kernel_size % 2 == 0:
            raise ValueError(f"RepVGG2d requires odd kernel_size, got {kernel_size}")
        if padding != kernel_size // 2:
            msg = f"RepVGG2d requires padding=kernel_size/2 , got {padding=} "
            msg += f"instead of {kernel_size//2}"
            raise ValueError(msg)

        self.rvg_dense_conv, self.rvg_dense_bn = utils.make_conv_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )

        self.rvg_1x1_conv, self.rvg_1x1_bn = utils.make_conv_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            groups=groups,
        )
        self.rvg_identity_bn = None
        self.rvg_identity_conv = None

        if out_channels == in_channels and stride == 1:
            # This identity conv is to facilitate pruning
            if identity_conv in {"trainable", "fixed"}:
                self.rvg_identity_conv = torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    groups=groups,
                    kernel_size=1,
                    bias=False,
                )
                utils.set_1x1_conv_to_identity(self.rvg_identity_conv)
                self.rvg_identity_conv.requires_grad_(identity_conv == "trainable")
            self.rvg_identity_bn = torch.nn.BatchNorm2d(num_features=in_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x_dense = self.rvg_dense_bn(self.rvg_dense_conv(inputs))
        x_1x1 = self.rvg_1x1_bn(self.rvg_1x1_conv(inputs))
        x = x_dense + x_1x1
        if self.rvg_identity_bn is not None and self.rvg_identity_conv is not None:
            x += self.rvg_identity_bn(self.rvg_identity_conv(inputs))
        elif self.rvg_identity_bn is not None:
            x += self.rvg_identity_bn(inputs)
        return x

    @property
    def in_channels(self) -> _size_2_t:
        return self.rvg_dense_conv.in_channels

    @property
    def out_channels(self) -> _size_2_t:
        return self.rvg_dense_conv.out_channels

    @property
    def groups(self) -> int:
        return self.rvg_dense_conv.groups

    @property
    def kernel_size(self) -> tuple[int, int]:
        # To make typechecker happy
        ks = self.rvg_dense_conv.kernel_size
        assert len(ks) == 2
        return ks[0], ks[1]


def _calc_repvgg_conv2d_fused_params(
    m: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Assert to make typechecker happy
    assert isinstance(m.rvg_dense_conv, torch.nn.Conv2d)
    assert isinstance(m.rvg_dense_bn, torch.nn.BatchNorm2d)
    kernel, bias = utils.fuse_bn_tensor(conv=m.rvg_dense_conv, bn=m.rvg_dense_bn)

    # Assert to make typechecker happy
    assert isinstance(m.rvg_1x1_conv, torch.nn.Conv2d)
    assert isinstance(m.rvg_1x1_bn, torch.nn.BatchNorm2d)
    kernel_1x1, bias_1x1 = utils.fuse_bn_tensor(conv=m.rvg_1x1_conv, bn=m.rvg_1x1_bn)
    kernel_1x1 = utils.pad_1x1_to_kxk(kernel_1x1, m.rvg_dense_conv.kernel_size[0])

    kernel += kernel_1x1
    bias += bias_1x1

    if (
        getattr(m, "rvg_identity_bn", None) is not None
        and getattr(m, "rvg_identity_conv", None) is not None
    ):
        assert isinstance(m.rvg_identity_conv, torch.nn.Conv2d)
        assert isinstance(m.rvg_identity_bn, torch.nn.BatchNorm2d)
        kernel_id, bias_id = utils.fuse_bn_tensor(
            conv=m.rvg_identity_conv, bn=m.rvg_identity_bn
        )
        kernel_id = utils.pad_1x1_to_kxk(kernel_id, m.rvg_dense_conv.kernel_size[0])
        kernel += kernel_id
        bias += bias_id

    elif getattr(m, "rvg_identity_bn", None) is not None:
        assert isinstance(m.rvg_identity_bn, torch.nn.BatchNorm2d)
        in_channels = m.rvg_dense_conv.in_channels
        groups = m.rvg_dense_conv.groups
        kernel_id, bias_id = utils.fuse_bn_tensor(
            conv=None,
            bn=m.rvg_identity_bn,
            kernel_size=m.rvg_dense_conv.kernel_size,
            in_channels=in_channels,
            groups=groups,
        )
        kernel += kernel_id
        bias += bias_id
    return kernel, bias


def is_repvgg2d(m: torch.nn.Module) -> bool:
    # Chcecking for all vars is a bit paranoid, but ...

    vars_repvgg2d_2branch = set(
        [
            "rvg_1x1_bn.bias",
            "rvg_1x1_bn.num_batches_tracked",
            "rvg_1x1_bn.running_mean",
            "rvg_1x1_bn.running_var",
            "rvg_1x1_bn.weight",
            "rvg_1x1_conv.weight",
            "rvg_dense_bn.bias",
            "rvg_dense_bn.num_batches_tracked",
            "rvg_dense_bn.running_mean",
            "rvg_dense_bn.running_var",
            "rvg_dense_bn.weight",
            "rvg_dense_conv.weight",
        ]
    )

    vars_repvgg2d_3branch = vars_repvgg2d_2branch | set(
        [
            "rvg_identity_bn.bias",
            "rvg_identity_bn.num_batches_tracked",
            "rvg_identity_bn.running_mean",
            "rvg_identity_bn.running_var",
            "rvg_identity_bn.weight",
            "rvg_identity_conv.weight",
        ]
    )

    vars_m = set(m.state_dict().keys())

    return vars_m == vars_repvgg2d_2branch or vars_m == vars_repvgg2d_3branch


def make_conv2d_from_repvgg2d(m: torch.nn.Module) -> torch.nn.Conv2d:
    # Assert to make typechecker happy
    assert hasattr(m, "rvg_dense_conv") and isinstance(
        m.rvg_dense_conv, torch.nn.Conv2d
    )
    fused_m = torch.nn.Conv2d(
        in_channels=m.rvg_dense_conv.in_channels,
        out_channels=m.rvg_dense_conv.out_channels,
        kernel_size=utils.to_size_2_t(m.rvg_dense_conv.kernel_size),
        stride=utils.to_size_2_t(m.rvg_dense_conv.stride),
        padding=utils.to_size_2_t(m.rvg_dense_conv.padding),
        dilation=utils.to_size_2_t(m.rvg_dense_conv.dilation),
        groups=m.rvg_dense_conv.groups,
        bias=True,
    )
    kernel, bias = _calc_repvgg_conv2d_fused_params(m)
    fused_m.weight.data = kernel
    # Assert to make typechecker happy
    assert isinstance(fused_m.bias, torch.nn.Parameter)
    fused_m.bias.data = bias
    return fused_m


def make_repvgg2d_from_conv2d(m: torch.nn.Conv2d) -> torch.nn.Module:
    assert m.bias is None

    m_repvgg = RepVGGConv2d(
        in_channels=m.in_channels,
        out_channels=m.out_channels,
        kernel_size=utils.to_size_2_t(m.kernel_size),
        stride=utils.to_size_2_t(m.stride),
        padding=utils.to_size_2_t(m.padding),
        groups=m.groups,
    )
    with torch.no_grad():
        m_repvgg.rvg_dense_conv.weight.copy_(m.weight.data)
        torch.nn.init.zeros_(m_repvgg.rvg_1x1_conv.weight)
    return m_repvgg


def distill_repvgg2d_in_place(
    m: torch.nn.Module, predecessor_mask: torch.Tensor, output_mask: torch.Tensor
) -> None:
    # Assert to make typechecker happy
    assert isinstance(m.rvg_dense_conv, torch.nn.Conv2d)
    distilling.distill_conv2d_in_place(
        m.rvg_dense_conv,
        predecessor_mask=predecessor_mask,
        output_mask=output_mask,
    )

    # Assert to make typechecker happy
    assert isinstance(m.rvg_dense_bn, torch.nn.BatchNorm2d)
    distilling.distill_batchnorm2d_in_place(
        m.rvg_dense_bn,
        predecessor_mask=output_mask,
        output_mask=output_mask,
    )

    # 1x1 branch
    # Assert to make typechecker happy
    assert isinstance(m.rvg_1x1_conv, torch.nn.Conv2d)
    distilling.distill_conv2d_in_place(
        m.rvg_1x1_conv,
        predecessor_mask=predecessor_mask,
        output_mask=output_mask,
    )
    # Assert to make typechecker happy
    assert isinstance(m.rvg_1x1_bn, torch.nn.BatchNorm2d)
    distilling.distill_batchnorm2d_in_place(
        m.rvg_1x1_bn,
        predecessor_mask=output_mask,
        output_mask=output_mask,
    )

    # Identity branch

    if m.rvg_identity_conv is not None:
        # Assert to make typechecker happy
        assert isinstance(m.rvg_identity_conv, torch.nn.Conv2d)
        distilling.distill_conv2d_in_place(
            m.rvg_identity_conv,
            predecessor_mask=predecessor_mask,
            output_mask=output_mask,
        )

    if m.rvg_identity_bn is not None:
        # Assert to make typechecker happy
        assert isinstance(m.rvg_identity_bn, torch.nn.BatchNorm2d)
        distilling.distill_batchnorm2d_in_place(
            m.rvg_identity_bn,
            predecessor_mask=output_mask,
            output_mask=output_mask,
        )
