from __future__ import annotations

import collections
import copy
import logging
import sys

import torch

from .repvgg2d import is_repvgg2d, make_conv2d_from_repvgg2d

__all__ = ["fuse_module_in_place", "fuse_module", "make_default_fusing_strategy"]

if sys.version_info >= (3, 9):
    from collections.abc import Callable

    FILTER_FN_TYPE = Callable[[torch.nn.Module], bool]
    FUSE_FN_TYPE = Callable[[torch.nn.Module], torch.nn.Module]
    FUSING_STRATEGY_TYPE = list[tuple[str, FILTER_FN_TYPE, FUSE_FN_TYPE]]
    FUSING_STRATEGY_FACTORY_TYPE = Callable[[], FUSING_STRATEGY_TYPE]
else:
    from typing import Any, Callable

    FILTER_FN_TYPE = Callable[[torch.nn.Module], bool]
    FUSE_FN_TYPE = Callable[[torch.nn.Module], torch.nn.Module]
    FUSING_STRATEGY_TYPE = Any
    FUSING_STRATEGY_FACTORY_TYPE = Callable[[], FUSING_STRATEGY_TYPE]


logger = logging.getLogger(__name__)


def _is_compound_module(m: torch.nn.Module) -> bool:
    return len(list(m.children())) > 0


def make_default_fusing_strategy() -> FUSING_STRATEGY_TYPE:
    # Wrapping in function prevents from accidental modification
    return [("RepVGG2d", is_repvgg2d, make_conv2d_from_repvgg2d)]


def _fuse_module_in_place(
    *,
    module: torch.nn.Module,
    module_path: tuple[str, ...],
    fused_counter: collections.Counter[str],
    fusing_strategy: FUSING_STRATEGY_TYPE,
) -> None:
    for rep_name, filter_fn, _ in fusing_strategy:
        if filter_fn(module):
            msg = f"fuse_module_in_place: reparametrization {rep_name} detected"
            msg += "at top level.\nPlease use fuse_module function"
            raise ValueError(msg)

    for child_name, child_module in module.named_children():
        for rep_name, filter_fn, fuse_fn in fusing_strategy:
            if filter_fn(child_module):
                full_child_name = ".".join((*module_path, child_name))
                logger.info(
                    f"Detected {rep_name} at {full_child_name},"
                    "replacing it with conv2d"
                )
                conv = fuse_fn(child_module)
                setattr(module, child_name, conv)
                fused_counter[rep_name] += 1
                break
        else:
            # This is else to for loop, will execute if no repa found
            if _is_compound_module(child_module):
                _fuse_module_in_place(
                    module=child_module,
                    module_path=(*module_path, child_name),
                    fused_counter=fused_counter,
                    fusing_strategy=fusing_strategy,
                )


def fuse_module_in_place(
    module: torch.nn.Module,
    make_fusing_strategy: FUSING_STRATEGY_FACTORY_TYPE = make_default_fusing_strategy,
) -> None:
    fused_counter: collections.Counter[str] = collections.Counter()
    fusing_strategy = make_fusing_strategy()

    _fuse_module_in_place(
        module=module,
        fused_counter=fused_counter,
        module_path=(),
        fusing_strategy=fusing_strategy,
    )

    tot = sum(v for k, v in fused_counter.items())
    logger.info(f"Number of all REPA blocks fused = {tot}")

    if tot > 0:
        for k, v in fused_counter.most_common():
            logger.info(f"Number of {k} blocks fused = {v}")


def fuse_module(
    m: torch.nn.Module,
    make_fusing_strategy: FUSING_STRATEGY_FACTORY_TYPE = make_default_fusing_strategy,
) -> torch.nn.Module:
    fusing_strategy = make_fusing_strategy()

    # Handle edge case when top level model is to be fused
    # That is fuse_module was called directly repa_module (m is repa_module)

    for rep_name, filter_fn, fuse_fn in fusing_strategy:
        if filter_fn(m):
            # logger.info(
            #     f"Detected {rep_name} at top level"
            #     f" replacing the whole module with conv2d"
            # )
            conv = fuse_fn(m)
            return conv

    # Handle non-edge cases

    m_copy = copy.deepcopy(m)
    fuse_module_in_place(m_copy, make_fusing_strategy=make_fusing_strategy)
    return m_copy
