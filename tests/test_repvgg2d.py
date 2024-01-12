from __future__ import annotations

import logging
from typing import Any, Optional

import pytest
import torch
import utils

import ptrepa

BATCH_SIZE = 5
INPUT_HW = 32

logger = logging.getLogger(__name__)


def _make_params_repvgg2d_basic(kernel_size_list: list[int]) -> list[dict[str, Any]]:
    res = []
    templates: list[dict[str, Any]] = [
        {"in_channels": 3, "out_channels": 6, "groups": 1},
        {"in_channels": 3, "out_channels": 3, "groups": 1},
        {"in_channels": 3, "out_channels": 3, "groups": 3},
        {"in_channels": 4, "out_channels": 8, "groups": 2},
    ]
    for ks in kernel_size_list:
        for identity_conv in ["trainable", "fixed", "disable"]:
            for t in templates:
                res.append({**t, **{"kernel_size": ks, "identity_conv": identity_conv}})
    return res


TEST_PARAMS_REPVGG2D_BASIC = _make_params_repvgg2d_basic([1, 3, 5, 7])


@pytest.mark.parametrize("params", TEST_PARAMS_REPVGG2D_BASIC)
def test_repvgg2d_properties(params: dict[str, Any]) -> None:
    m = ptrepa.RepVGGConv2d(**params)
    assert m.in_channels == params["in_channels"]
    assert m.out_channels == params["out_channels"]
    assert m.groups == params["groups"]
    assert m.kernel_size == (params["kernel_size"], params["kernel_size"])
    if m.in_channels == m.out_channels:
        if params["identity_conv"] == "trainable":
            assert m.rvg_identity_conv is not None
            assert m.rvg_identity_conv.weight.requires_grad is True
        elif params["identity_conv"] == "fixed":
            assert m.rvg_identity_conv is not None
            assert m.rvg_identity_conv.weight.requires_grad is False
        elif params["identity_conv"] == "disable":
            assert m.rvg_identity_conv is None


@pytest.mark.parametrize("params", TEST_PARAMS_REPVGG2D_BASIC)
def test_repvgg2d_forward(params: dict[str, Any]) -> None:
    m = ptrepa.RepVGGConv2d(**params)
    x = torch.rand(BATCH_SIZE, params["in_channels"], INPUT_HW, INPUT_HW)
    with torch.inference_mode():
        y = m(x)

    assert y.shape == (BATCH_SIZE, params["out_channels"], INPUT_HW, INPUT_HW)


@pytest.mark.parametrize("params", TEST_PARAMS_REPVGG2D_BASIC)
def test_fuse_repvgg2d(params: dict[str, Any]) -> None:
    # Original

    m = ptrepa.RepVGGConv2d(**params)
    x = torch.rand(BATCH_SIZE, params["in_channels"], INPUT_HW, INPUT_HW)
    m.eval()
    with torch.inference_mode():
        y = m(x)

    # Fused

    m_fused = ptrepa.make_conv2d_from_repvgg2d(m)
    m_fused.eval()  # OK, this is a conv, this is not so super necessary ;)
    with torch.inference_mode():
        y_fused = m_fused(x)

    torch.testing.assert_close(y, y_fused)


@pytest.mark.parametrize("test_params", TEST_PARAMS_REPVGG2D_BASIC)
def test_fuse_repvgg2d_traced(test_params: dict[str, Any]) -> None:
    # Original

    m = ptrepa.RepVGGConv2d(**test_params)
    x = torch.rand(BATCH_SIZE, test_params["in_channels"], INPUT_HW, INPUT_HW)
    m.eval()
    with torch.inference_mode():
        y = m(x)

    # Fused traced
    tr_m = torch.fx.symbolic_trace(m)
    m_fused = ptrepa.make_conv2d_from_repvgg2d(tr_m)
    m_fused.eval()  # OK, this is a conv, this is not so super necessary ;)
    with torch.inference_mode():
        y_fused = m_fused(x)

    torch.testing.assert_close(y, y_fused)


@pytest.mark.parametrize("params", TEST_PARAMS_REPVGG2D_BASIC)
def test_make_repvgg_from_conv2d(params: dict[str, Any]) -> None:
    padding = params["kernel_size"] // 2

    m = torch.nn.Conv2d(
        bias=False,
        padding=padding,
        kernel_size=params["kernel_size"],
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        groups=params["groups"],
    )

    x = torch.rand(BATCH_SIZE, params["in_channels"], INPUT_HW, INPUT_HW)
    m.eval()
    with torch.inference_mode():
        y = m(x)

    m_repa = ptrepa.make_repvgg2d_from_conv2d(m)
    m_repa.eval()
    with torch.inference_mode():
        y_repa = m_repa(x)
    torch.testing.assert_close(y, y_repa)


TEST_PARAMS_IS_REPVGG2D = [
    {
        "module": ptrepa.RepVGGConv2d(in_channels=3, out_channels=6, kernel_size=3),
        "is_repvgg2d": True,
    },
    {
        "module": ptrepa.RepVGGConv2d(in_channels=3, out_channels=3, kernel_size=3),
        "is_repvgg2d": True,
    },
    {
        "module": torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3),
        "is_repvgg2d": False,
    },
    {
        "module": torch.fx.symbolic_trace(
            ptrepa.RepVGGConv2d(in_channels=3, out_channels=6, kernel_size=3)
        ),
        "is_repvgg2d": True,
    },
    {
        "module": torch.fx.symbolic_trace(
            ptrepa.RepVGGConv2d(in_channels=3, out_channels=3, kernel_size=3)
        ),
        "is_repvgg2d": True,
    },
]


@pytest.mark.parametrize("params", TEST_PARAMS_IS_REPVGG2D)
def test_is_repvgg2(params: dict[str, Any]) -> None:
    m = params["module"]
    is_repvgg2d = params["is_repvgg2d"]
    assert ptrepa.is_repvgg2d(m) == is_repvgg2d


def check_distill_repvgg2d_in_place(
    params: dict[str, Any],
    device: torch.device,
    float64: bool = False,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> None:
    rng = torch.Generator().manual_seed(12345)
    for _ in range(5):
        in_channels = params["in_channels"]
        out_channels = params["out_channels"]
        kernel_size = params["kernel_size"]
        groups = params["groups"]
        predecessor_mask = torch.rand(size=(in_channels,), generator=rng) > 0.5

        # Random predecessor mask + ensure at least one  channel is left
        i = int((torch.rand(size=(1,), generator=rng) * in_channels).item())
        predecessor_mask[i] = True

        if groups == 1:
            # Random output mask + ensure at least one channel is left
            output_mask = torch.rand(size=(out_channels,), generator=rng) > 0.5
            i = int((torch.rand(size=(1,), generator=rng) * out_channels).item())
            output_mask[i] = True
        elif groups == in_channels and in_channels == out_channels:
            # For depthwise conv pruning is supported only if expansion rate == 1
            assert params["in_channels"] == params["out_channels"]
            # For depthwise conv we prune input and output in the same places
            output_mask = predecessor_mask
        else:
            raise ValueError(
                "Testing unsupported case" f"{in_channels=} {out_channels=} {groups=}"
            )
        output_mask = output_mask.to(device)
        predecessor_mask = predecessor_mask.to(device)
        in_channels_new = predecessor_mask.sum().item()
        msg = f"Distilling in_channels={in_channels} -> {in_channels_new}"
        logger.info(msg)
        out_channels_new = output_mask.sum().item()
        msg = f"Distilling out_channels={out_channels} -> {out_channels_new}"
        logger.info(msg)

        x = torch.rand(size=(16, in_channels, 32, 32), generator=rng)
        if float64:
            x = x.to(dtype=torch.float64)
        x = x.to(device)

        x_masked = utils.mask_channels(x, predecessor_mask)
        x_masked = x_masked.to(device)
        m = ptrepa.RepVGGConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
            kernel_size=kernel_size,
        )
        logger.info(f"{m.rvg_identity_conv=} {m.rvg_identity_bn=}")
        m.to(device)
        if float64:
            m.to(dtype=torch.float64)
        m.eval()

        with torch.no_grad():
            y_masked = m(x_masked)[:, output_mask, :, :]

        ptrepa.distill_repvgg2d_in_place(
            m, predecessor_mask=predecessor_mask, output_mask=output_mask
        )
        assert m.rvg_dense_conv.weight.shape[0] == out_channels_new
        assert m.groups != 1 or m.rvg_dense_conv.weight.shape[1] == in_channels_new

        m.eval()
        x_pruned = x[:, predecessor_mask, :, :]
        with torch.no_grad():
            y_pruned = m(x_pruned)
        if atol is not None and rtol is not None:
            torch.testing.assert_close(y_masked, y_pruned, atol=atol, rtol=rtol)
        else:
            torch.testing.assert_close(y_masked, y_pruned)
        m_pruned_fused = ptrepa.fuse_module(m)
        m_pruned_fused.to(device)
        m_pruned_fused.eval()
        with torch.no_grad():
            y_pruned_fused = m(x_pruned)
        logger.info(f"{torch.max(torch.abs(y_pruned - y_pruned_fused))}")
        torch.testing.assert_close(y_pruned, y_pruned_fused)


def _make_params_repvgg2d_distill(kernel_sizes: list[int]) -> list[dict[str, Any]]:
    res = []
    templates: list[dict[str, Any]] = [
        {"in_channels": 3, "out_channels": 6, "groups": 1},
        {"in_channels": 3, "out_channels": 3, "groups": 1},
        {"in_channels": 3, "out_channels": 3, "groups": 3},
        {"in_channels": 16, "out_channels": 48, "groups": 1},
        {"in_channels": 32, "out_channels": 32, "groups": 1},
        {"in_channels": 16, "out_channels": 16, "groups": 16},
    ]
    for ks in kernel_sizes:
        for identity_conv in ["trainable", "fixed"]:
            for t in templates:
                res.append({**t, **{"kernel_size": ks, "identity_conv": identity_conv}})
    return res


TEST_PARAMS_REPVGG2D_DISTILL = _make_params_repvgg2d_distill([1, 3, 5, 7])


@pytest.mark.parametrize("params", TEST_PARAMS_REPVGG2D_DISTILL)
def test_distill_repvgg2d_in_place_cpu(params: dict[str, Any]) -> None:
    logger.info(params)
    check_distill_repvgg2d_in_place(params, torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@pytest.mark.parametrize("params", TEST_PARAMS_REPVGG2D_DISTILL)
def test_distill_repvgg2d_in_place_cuda(params: dict[str, Any]) -> None:
    logger.info(params)
    # During the tests floating point error seems to acumulate, hence degraded *tol
    check_distill_repvgg2d_in_place(
        params, torch.device("cuda"), atol=0.0008, rtol=1.0e-6
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@pytest.mark.parametrize("params", TEST_PARAMS_REPVGG2D_DISTILL)
def test_distill_repvgg2d_in_place_cuda64(params: dict[str, Any]) -> None:
    # During the tests floating point error seems to acumulate
    # float64 test ensure that this is only
    logger.info(params)
    check_distill_repvgg2d_in_place(params, torch.device("cuda"), float64=True)
