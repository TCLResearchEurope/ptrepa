from __future__ import annotations

from typing import Optional

import torch
import torch.utils.checkpoint as checkpoint
from torch.nn.common_types import _size_2_t

from . import repvgg2d, utils


class RepVGGBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        groups: int = 1,
        use_se: bool = False,
        identity_conv: str = "trainable",
    ):
        super().__init__()
        self.groups = groups
        self.in_channels = in_channels

        self.nonlinearity = torch.nn.ReLU()

        if use_se:
            self.se: torch.nn.Module = utils.SEBlock(out_channels, rd_ratio=1.0 / 16.0)
        else:
            self.se = torch.nn.Identity()

        self.conv = repvgg2d.RepVGGConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            identity_conv=identity_conv,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.se(y)
        return self.nonlinearity(y)


class RepVGG(torch.nn.Module):
    def __init__(
        self,
        *,
        num_blocks: list[int],
        num_classes: int = 1000,
        width_multiplier: list[float],
        override_groups_map: Optional[dict[int, int]] = None,
        use_se: bool = False,
        use_checkpoint: bool = False,
        identity_conv: str = "trainable",
    ):
        super().__init__()
        assert len(width_multiplier) == 4
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.use_se = use_se
        self.use_checkpoint = use_checkpoint

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            use_se=self.use_se,
            identity_conv=identity_conv,
        )
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(
            int(64 * width_multiplier[0]), num_blocks[0], stride=2
        )
        self.stage2 = self._make_stage(
            int(128 * width_multiplier[1]), num_blocks[1], stride=2
        )
        self.stage3 = self._make_stage(
            int(256 * width_multiplier[2]), num_blocks[2], stride=2
        )
        self.stage4 = self._make_stage(
            int(512 * width_multiplier[3]), num_blocks[3], stride=2
        )
        self.gap = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = torch.nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(
        self, planes: int, num_blocks: int, stride: int
    ) -> torch.nn.ModuleList:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=cur_groups,
                    use_se=self.use_se,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return torch.nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stage0(x)
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            for block in stage:
                if self.use_checkpoint:
                    out = checkpoint.checkpoint(block, out)
                else:
                    out = block(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def _get_groupwise_layers() -> list[int]:
    return [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]


def _get_g2_map() -> dict[int, int]:
    return {i: 2 for i in _get_groupwise_layers()}


def _get_g4_map() -> dict[int, int]:
    return {i: 4 for i in _get_groupwise_layers()}


def create_RepVGG_A0(
    use_checkpoint: bool = False, identity_conv: str = "trainable"
) -> torch.nn.Module:
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        num_classes=1000,
        width_multiplier=[0.75, 0.75, 0.75, 2.5],
        override_groups_map=None,
        use_checkpoint=use_checkpoint,
        identity_conv=identity_conv,
    )


def create_RepVGG_A1(
    use_checkpoint: bool = False, identity_conv: str = "trainable"
) -> torch.nn.Module:
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        num_classes=1000,
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
        use_checkpoint=use_checkpoint,
        identity_conv=identity_conv,
    )


def create_RepVGG_A2(
    use_checkpoint: bool = False, identity_conv: str = "trainable"
) -> torch.nn.Module:
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        num_classes=1000,
        width_multiplier=[1.5, 1.5, 1.5, 2.75],
        override_groups_map=None,
        use_checkpoint=use_checkpoint,
        identity_conv=identity_conv,
    )


def create_RepVGG_B0(
    use_checkpoint: bool = False, identity_conv: str = "trainable"
) -> torch.nn.Module:
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
        use_checkpoint=use_checkpoint,
        identity_conv=identity_conv,
    )


def create_RepVGG_B1(
    use_checkpoint: bool = False, identity_conv: str = "trainable"
) -> torch.nn.Module:
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=None,
        use_checkpoint=use_checkpoint,
        identity_conv=identity_conv,
    )


def create_RepVGG_B1g2(
    use_checkpoint: bool = False, identity_conv: str = "trainable"
) -> torch.nn.Module:
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=_get_g2_map(),
        use_checkpoint=use_checkpoint,
        identity_conv=identity_conv,
    )


def create_RepVGG_B1g4(
    use_checkpoint: bool = False, identity_conv: str = "trainable"
) -> torch.nn.Module:
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=_get_g4_map(),
        use_checkpoint=use_checkpoint,
        identity_conv=identity_conv,
    )


def create_RepVGG_B2(
    use_checkpoint: bool = False, identity_conv: str = "trainable"
) -> torch.nn.Module:
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=None,
        use_checkpoint=use_checkpoint,
        identity_conv=identity_conv,
    )


def create_RepVGG_B2g2(
    use_checkpoint: bool = False, identity_conv: str = "trainable"
) -> torch.nn.Module:
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=_get_g2_map(),
        use_checkpoint=use_checkpoint,
        identity_conv=identity_conv,
    )


def create_RepVGG_B2g4(
    use_checkpoint: bool = False, identity_conv: str = "trainable"
) -> torch.nn.Module:
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=_get_g4_map(),
        use_checkpoint=use_checkpoint,
        identity_conv=identity_conv,
    )


def create_RepVGG_B3(
    use_checkpoint: bool = False, identity_conv: str = "trainable"
) -> torch.nn.Module:
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=None,
        use_checkpoint=use_checkpoint,
        identity_conv=identity_conv,
    )


def create_RepVGG_B3g2(
    use_checkpoint: bool = False, identity_conv: str = "trainable"
) -> torch.nn.Module:
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=_get_g2_map(),
        use_checkpoint=use_checkpoint,
        identity_conv=identity_conv,
    )


def create_RepVGG_B3g4(
    use_checkpoint: bool = False, identity_conv: str = "trainable"
) -> torch.nn.Module:
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=_get_g4_map(),
        use_checkpoint=use_checkpoint,
        identity_conv=identity_conv,
    )


def create_RepVGG_D2se(
    use_checkpoint: bool = False, identity_conv: str = "trainable"
) -> torch.nn.Module:
    return RepVGG(
        num_blocks=[8, 14, 24, 1],
        num_classes=1000,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=None,
        use_se=True,
        use_checkpoint=use_checkpoint,
        identity_conv=identity_conv,
    )


func_dict = {
    "RepVGG-A0": create_RepVGG_A0,
    "RepVGG-A1": create_RepVGG_A1,
    "RepVGG-A2": create_RepVGG_A2,
    "RepVGG-B0": create_RepVGG_B0,
    "RepVGG-B1": create_RepVGG_B1,
    "RepVGG-B1g2": create_RepVGG_B1g2,
    "RepVGG-B1g4": create_RepVGG_B1g4,
    "RepVGG-B2": create_RepVGG_B2,
    "RepVGG-B2g2": create_RepVGG_B2g2,
    "RepVGG-B2g4": create_RepVGG_B2g4,
    "RepVGG-B3": create_RepVGG_B3,
    "RepVGG-B3g2": create_RepVGG_B3g2,
    "RepVGG-B3g4": create_RepVGG_B3g4,
    # Updated at April 25, 2021. This is not reported in the CVPR paper.
    "RepVGG-D2se": create_RepVGG_D2se,
}
