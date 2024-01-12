from __future__ import annotations

import copy
from typing import Optional

import torch
from torch.nn.common_types import _size_2_t

from . import distilling, utils

__all__ = [
    "QuantRepVGGConv2d",
    "distill_quant_repvgg2d_in_place",
    "is_quant_repvgg2d",
    "make_conv2d_from_quant_repvgg2d",
    "make_quant_repvgg2d_from_conv2d",
]


class QuantRepVGGConv2d(torch.nn.Module):
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
            msg += f"instead of {kernel_size // 2}"
            raise ValueError(msg)

        self.qrvg_dense_conv = utils.make_conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )

        self.qrvg_1x1_conv, self.qrvg_1x1_bn = utils.make_conv_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            groups=groups,
        )
        self.qrvg_identity_conv = None

        self.identity = False
        if out_channels == in_channels and stride == 1:
            # This identity conv is to facilitate pruning
            if identity_conv in {"trainable", "fixed"}:
                self.qrvg_identity_conv = utils.make_conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    groups=groups,
                    kernel_size=1,
                    use_bias=False,
                )
                utils.set_1x1_conv_to_identity(self.qrvg_identity_conv)
                self.qrvg_identity_conv.requires_grad_(identity_conv == "trainable")
            else:
                self.identity = True
        self.qrvg_final_bn = torch.nn.BatchNorm2d(num_features=out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x_dense = self.qrvg_dense_conv(inputs)
        x_1x1 = self.qrvg_1x1_bn(self.qrvg_1x1_conv(inputs))
        x = x_dense + x_1x1
        if self.qrvg_identity_conv is not None:
            x += self.qrvg_identity_conv(inputs)
        elif self.identity:
            x += inputs
        x = self.qrvg_final_bn(x)
        return x

    @property
    def in_channels(self) -> _size_2_t:
        return self.qrvg_dense_conv.in_channels

    @property
    def out_channels(self) -> _size_2_t:
        return self.qrvg_dense_conv.out_channels

    @property
    def groups(self) -> int:
        return self.qrvg_dense_conv.groups

    @property
    def kernel_size(self) -> tuple[int, int]:
        # To make typechecker happy
        ks = self.qrvg_dense_conv.kernel_size
        assert len(ks) == 2
        return ks[0], ks[1]


def _calc_repvgg_conv2d_fused_params(
    m: QuantRepVGGConv2d,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        kernel, bias = copy.deepcopy(m.qrvg_dense_conv.weight), copy.deepcopy(
            m.qrvg_dense_conv.bias
        )

        kernel_1x1, bias_1x1 = utils.fuse_bn_tensor(
            conv=m.qrvg_1x1_conv, bn=m.qrvg_1x1_bn
        )
        kernel_1x1 = utils.pad_1x1_to_kxk(kernel_1x1, m.qrvg_dense_conv.kernel_size[0])

        kernel += kernel_1x1
        bias = bias + bias_1x1 if bias is not None else bias_1x1

        if m.qrvg_identity_conv is not None:
            kernel_id = m.qrvg_identity_conv.weight
            kernel_id = utils.pad_1x1_to_kxk(
                kernel_id, m.qrvg_dense_conv.kernel_size[0]
            )
            kernel += kernel_id

        elif m.identity:
            in_channels = m.qrvg_dense_conv.in_channels
            groups = m.qrvg_dense_conv.groups
            kernel_id = utils.identity_to_conv(
                kernel_size=m.qrvg_dense_conv.kernel_size,
                in_channels=in_channels,
                groups=groups,
            )
            kernel += kernel_id
        fake_conv = torch.nn.Conv2d(
            in_channels=m.qrvg_dense_conv.in_channels,
            out_channels=m.qrvg_dense_conv.out_channels,
            kernel_size=utils.to_size_2_t(m.qrvg_dense_conv.kernel_size),
            padding=utils.to_size_2_t(m.qrvg_dense_conv.padding),
            stride=utils.to_size_2_t(m.qrvg_dense_conv.stride),
        )
        fake_conv.weight.data = kernel
        if bias is not None:
            assert isinstance(fake_conv.bias, torch.nn.Parameter)
            fake_conv.bias.data = bias
        kernel, bias = utils.fuse_bn_tensor(
            conv=fake_conv,
            bn=m.qrvg_final_bn,
        )
        return kernel, bias


def is_quant_repvgg2d(m: torch.nn.Module) -> bool:
    # Chcecking for all vars is a bit paranoid, but ...

    vars_repvgg2d_2branch = {
        "qrvg_1x1_bn.bias",
        "qrvg_1x1_bn.num_batches_tracked",
        "qrvg_1x1_bn.running_mean",
        "qrvg_1x1_bn.running_var",
        "qrvg_1x1_bn.weight",
        "qrvg_1x1_conv.weight",
        "qrvg_dense_conv.weight",
        "qrvg_final_bn.bias",
        "qrvg_final_bn.num_batches_tracked",
        "qrvg_final_bn.running_mean",
        "qrvg_final_bn.running_var",
        "qrvg_final_bn.weight",
    }

    vars_repvgg2d_3branch = vars_repvgg2d_2branch | {"qrvg_identity_conv.weight"}

    vars_m = set(m.state_dict().keys())

    return vars_m == vars_repvgg2d_2branch or vars_m == vars_repvgg2d_3branch


def make_conv2d_from_quant_repvgg2d(m: QuantRepVGGConv2d) -> torch.nn.Conv2d:
    # Assert to make typechecker happy
    assert hasattr(m, "qrvg_dense_conv") and isinstance(
        m.qrvg_dense_conv, torch.nn.Conv2d
    )
    fused_m = torch.nn.Conv2d(
        in_channels=m.qrvg_dense_conv.in_channels,
        out_channels=m.qrvg_dense_conv.out_channels,
        kernel_size=utils.to_size_2_t(m.qrvg_dense_conv.kernel_size),
        stride=utils.to_size_2_t(m.qrvg_dense_conv.stride),
        padding=utils.to_size_2_t(m.qrvg_dense_conv.padding),
        dilation=utils.to_size_2_t(m.qrvg_dense_conv.dilation),
        groups=m.qrvg_dense_conv.groups,
        bias=True,
    )
    kernel, bias = _calc_repvgg_conv2d_fused_params(m)
    # Assert to make typechecker happy
    assert isinstance(fused_m.weight, torch.nn.Parameter)
    fused_m.weight.data = kernel
    # Assert to make typechecker happy
    assert isinstance(fused_m.bias, torch.nn.Parameter)
    fused_m.bias.data = bias
    return fused_m


def make_quant_repvgg2d_from_conv2d(m: torch.nn.Conv2d) -> torch.nn.Module:
    assert m.bias is None

    m_repvgg = QuantRepVGGConv2d(
        in_channels=m.in_channels,
        out_channels=m.out_channels,
        kernel_size=utils.to_size_2_t(m.kernel_size),
        stride=utils.to_size_2_t(m.stride),
        padding=utils.to_size_2_t(m.padding),
        groups=m.groups,
    )
    with torch.no_grad():
        m_repvgg.qrvg_dense_conv.weight.copy_(m.weight.data)
        torch.nn.init.zeros_(m_repvgg.qrvg_1x1_conv.weight)
    return m_repvgg


def distill_quant_repvgg2d_in_place(
    m: QuantRepVGGConv2d, predecessor_mask: torch.Tensor, output_mask: torch.Tensor
) -> None:
    # kxk branch
    distilling.distill_conv2d_in_place(
        m.qrvg_dense_conv,
        predecessor_mask=predecessor_mask,
        output_mask=output_mask,
    )

    # 1x1 branch
    distilling.distill_conv2d_in_place(
        m.qrvg_1x1_conv,
        predecessor_mask=predecessor_mask,
        output_mask=output_mask,
    )
    distilling.distill_batchnorm2d_in_place(
        m.qrvg_1x1_bn,
        predecessor_mask=output_mask,
        output_mask=output_mask,
    )

    # Identity branch
    if m.qrvg_identity_conv is not None:
        distilling.distill_conv2d_in_place(
            m.qrvg_identity_conv,
            predecessor_mask=predecessor_mask,
            output_mask=output_mask,
        )
    distilling.distill_batchnorm2d_in_place(
        m.qrvg_final_bn,
        predecessor_mask=output_mask,
        output_mask=output_mask,
    )
