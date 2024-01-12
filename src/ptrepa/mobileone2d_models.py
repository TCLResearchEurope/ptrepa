from __future__ import annotations

from typing import Any

import torch

from . import mobileone2d, utils


class MobileOneBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        use_se: bool = False,
        num_conv_branches: int = 1,
    ) -> None:
        super(MobileOneBlock, self).__init__()
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        if use_se:
            self.se: torch.nn.Module = utils.SEBlock(out_channels, rd_ratio=1.0 / 16.0)
        else:
            self.se = torch.nn.Identity()
        self.activation = torch.nn.ReLU()
        self.conv = mobileone2d.MobileOneConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            num_conv_branches=num_conv_branches,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.se(y)
        return self.activation(y)


class MobileOne(torch.nn.Module):
    """MobileOne Model

    Pytorch implementation of `An Improved One millisecond Mobile Backbone` -
    https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(
        self,
        num_blocks_per_stage: tuple[int, ...] = (2, 8, 10, 1),
        num_classes: int = 1000,
        width_multipliers: tuple[float, ...] = (),
        use_se: bool = False,
        num_conv_branches: int = 1,
    ) -> None:
        super().__init__()

        assert len(width_multipliers) == 4
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        # Build stages
        self.stage0 = MobileOneBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(
            int(64 * width_multipliers[0]), num_blocks_per_stage[0], num_se_blocks=0
        )
        self.stage2 = self._make_stage(
            int(128 * width_multipliers[1]), num_blocks_per_stage[1], num_se_blocks=0
        )
        self.stage3 = self._make_stage(
            int(256 * width_multipliers[2]),
            num_blocks_per_stage[2],
            num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0,
        )
        self.stage4 = self._make_stage(
            int(512 * width_multipliers[3]),
            num_blocks_per_stage[3],
            num_se_blocks=num_blocks_per_stage[3] if use_se else 0,
        )
        self.gap = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = torch.nn.Linear(int(512 * width_multipliers[3]), num_classes)

    def _make_stage(
        self, planes: int, num_blocks: int, num_se_blocks: int
    ) -> torch.nn.Sequential:
        # Get strides for all layers
        strides = [2] + [1] * (num_blocks - 1)
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError(
                    "Number of SE blocks cannot " "exceed number of layers."
                )
            if ix >= (num_blocks - num_se_blocks):
                use_se = True

            # Depthwise conv
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=self.in_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=self.in_planes,
                    use_se=use_se,
                    num_conv_branches=self.num_conv_branches,
                )
            )
            # Pointwise conv
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    use_se=use_se,
                    num_conv_branches=self.num_conv_branches,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return torch.nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def mobileone(num_classes: int = 1000, variant: str = "s0") -> torch.nn.Module:
    mobileon_params: dict[str, dict[str, Any]] = {
        "s0": {"width_multipliers": (0.75, 1.0, 1.0, 2.0), "num_conv_branches": 4},
        "s1": {"width_multipliers": (1.5, 1.5, 2.0, 2.5)},
        "s2": {"width_multipliers": (1.5, 2.0, 2.5, 4.0)},
        "s3": {"width_multipliers": (2.0, 2.5, 3.0, 4.0)},
        "s4": {"width_multipliers": (3.0, 3.5, 3.5, 4.0), "use_se": True},
    }

    variant_params = mobileon_params[variant]
    return MobileOne(num_classes=num_classes, **variant_params)
