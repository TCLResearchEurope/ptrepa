from __future__ import annotations

from typing import Optional

import torch

from . import utils

__all__ = [
    "MobileOneConv2d",
    "distill_mobileone2d_in_place",
    "is_mobileone2d",
    "make_conv2d_from_mobileone2d",
    "make_mobileone2d_from_conv2d",
]


class MobileOneConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        num_conv_branches: int = 1,
        identity_conv: str = "trainable",
    ) -> None:
        super().__init__()

        if identity_conv not in {"trainable", "fixed", "disable"}:
            msg = f"{identity_conv=} not in (trainable, fixed, disable)"
            raise ValueError(msg)

        if padding is None:
            padding = kernel_size // 2

        if kernel_size % 2 == 0:
            raise ValueError(f"MobileOne2d requires odd kernel_size, got {kernel_size}")

        if padding != kernel_size // 2:
            msg = f"MobileOne2d requires padding=kernel_size/2 , got {padding=} "
            msg += f"instead of {kernel_size//2}"
            raise ValueError(msg)

        # Conv branches

        rbr_conv = list()
        for _ in range(num_conv_branches):
            m = utils.make_conv_bn_sequential(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                groups=groups,
                stride=stride,
                padding=padding,
            )
            rbr_conv.append(m)
        self.rbr_conv = torch.nn.ModuleList(rbr_conv)

        # 1x1 branch

        self.rbr_scale = None
        if kernel_size > 1:
            self.rbr_scale = utils.make_conv_bn_sequential(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                groups=groups,
                stride=stride,
                padding=0,
            )

        # Identity branch

        self.rbr_identity_conv = None
        self.rbr_identity_bn = None

        if out_channels == in_channels and stride == 1:
            # This identity conv is to facilitate pruning
            if identity_conv in {"trainable", "fixed"}:
                self.rbr_identity_conv = torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    groups=groups,
                    kernel_size=1,
                    bias=False,
                )
                utils.set_1x1_conv_to_identity(self.rbr_identity_conv)
                self.rbr_identity_conv.requires_grad_(identity_conv == "trainable")
            self.rbr_identity_bn = torch.nn.BatchNorm2d(num_features=in_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        n = len(self.rbr_conv)
        out = self.rbr_conv[0](inputs)

        for i in range(1, n):
            out += self.rbr_conv[i](inputs)

        if self.rbr_scale is not None:
            out += self.rbr_scale(inputs)

        if self.rbr_identity_bn is not None and self.rbr_identity_conv is not None:
            out += self.rbr_identity_bn(self.rbr_identity_conv(inputs))
        elif self.rbr_identity_bn is not None:
            out += self.rbr_identity_bn(inputs)

        return out


def _calc_mobileone_conv2d_fused_params(
    m: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Conv branches

    conv_m = utils.get_module(m, "rbr_conv.0.conv")
    assert isinstance(conv_m, torch.nn.Conv2d)
    bn_m = utils.get_module(m, "rbr_conv.0.bn")
    assert isinstance(bn_m, torch.nn.BatchNorm2d)
    kernel, bias = utils.fuse_bn_tensor(conv=conv_m, bn=bn_m)

    rbr_conv = utils.get_module(m, "rbr_conv")
    assert isinstance(rbr_conv, torch.nn.ModuleList)
    for i in range(1, len(rbr_conv)):
        conv = utils.get_module(m, f"rbr_conv.{i}.conv")
        assert isinstance(conv, torch.nn.Conv2d)
        bn = utils.get_module(m, f"rbr_conv.{i}.bn")
        assert isinstance(bn, torch.nn.BatchNorm2d)
        kernel_i, bias_i = utils.fuse_bn_tensor(conv=conv, bn=bn)
        kernel += kernel_i
        bias += bias_i

    # Scale branch

    rbr_scale = getattr(m, "rbr_scale", None)

    if rbr_scale is not None:
        conv = utils.get_module(m, "rbr_scale.conv")
        assert isinstance(conv, torch.nn.Conv2d)
        bn = utils.get_module(m, "rbr_scale.bn")
        assert isinstance(bn, torch.nn.BatchNorm2d)
        kernel_i, bias_i = utils.fuse_bn_tensor(conv=conv, bn=bn)
        kernel_i = utils.pad_1x1_to_kxk(kernel_i, conv_m.kernel_size[0])
        kernel += kernel_i
        bias += bias_i

    # Identity branch

    if (
        getattr(m, "rbr_identity_bn", None) is not None
        and getattr(m, "rbr_identity_conv", None) is not None
    ):
        assert isinstance(m.rbr_identity_conv, torch.nn.Conv2d)
        assert isinstance(m.rbr_identity_bn, torch.nn.BatchNorm2d)
        kernel_id, bias_id = utils.fuse_bn_tensor(
            conv=m.rbr_identity_conv, bn=m.rbr_identity_bn
        )
        kernel_id = utils.pad_1x1_to_kxk(kernel_id, conv_m.kernel_size[0])
        kernel += kernel_id
        bias += bias_id

    elif getattr(m, "rbr_identity_bn", None) is not None:
        assert isinstance(m.rbr_identity_bn, torch.nn.BatchNorm2d)
        kernel_id, bias_id = utils.fuse_bn_tensor(
            conv=None,
            bn=m.rbr_identity_bn,
            kernel_size=conv_m.kernel_size,
            in_channels=conv_m.in_channels,
            groups=conv_m.groups,
        )
        kernel += kernel_id
        bias += bias_id

    return kernel, bias

    # # Orig implementation

    # kernel_scale = 0
    # bias_scale = 0
    # if self.rbr_scale is not None:
    #     kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
    #     # Pad scale branch kernel to match conv branch kernel size.
    #     pad = self.kernel_size // 2
    #     kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

    # # get weights and bias of skip branch
    # kernel_identity = 0
    # bias_identity = 0
    # if self.rbr_skip is not None:
    #     kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

    # # get weights and bias of conv branches
    # kernel_conv = 0
    # bias_conv = 0
    # for ix in range(self.num_conv_branches):
    #     _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
    #     kernel_conv += _kernel
    #     bias_conv += _bias

    # kernel_final = kernel_conv + kernel_scale + kernel_identity
    # bias_final = bias_conv + bias_scale + bias_identity
    # return kernel_final, bias_final
    # raise NotImplementedError("Not implemented")


def is_mobileone2d(m: torch.nn.Module) -> bool:
    raise NotImplementedError("Not implemented")


def make_conv2d_from_mobileone2d(m: torch.nn.Module) -> torch.nn.Conv2d:
    # Assert to make typechecker happy
    conv = utils.get_module(m, "rbr_conv.0.conv")
    assert isinstance(conv, torch.nn.Conv2d)

    fused_m = torch.nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=utils.to_size_2_t(conv.kernel_size),
        stride=utils.to_size_2_t(conv.stride),
        padding=utils.to_size_2_t(conv.padding),
        dilation=utils.to_size_2_t(conv.dilation),
        groups=conv.groups,
        bias=True,
    )
    kernel, bias = _calc_mobileone_conv2d_fused_params(m)
    fused_m.weight.data = kernel
    # Assert to make typechecker happy
    assert isinstance(fused_m.bias, torch.nn.Parameter)
    fused_m.bias.data = bias
    return fused_m


def make_mobileone2d_from_conv2d(m: torch.nn.Conv2d) -> torch.nn.Module:
    raise NotImplementedError("Not implemented")


def distill_mobileone2d_in_place(
    m: torch.nn.Module, predecessor_mask: torch.Tensor, output_mask: torch.Tensor
) -> None:
    raise NotImplementedError("Not implemented")
