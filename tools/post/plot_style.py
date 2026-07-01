# -*- coding: utf-8 -*-
"""
Shared plot configuration helpers.
Provides centralized figure-size and font-size management for all plotting scripts.

Usage —— add three lines to any plotting script:
    from tools.post.plot_style import add_figsize_args, resolve_figsize
    parser = argparse.ArgumentParser()
    add_figsize_args(parser)                    # adds --fig-width, --fig-height, --fig-scale
    args = parser.parse_args()
    w, h = resolve_figsize(args, default=(8.2, 4.8))
    fig, ax = plt.subplots(figsize=(w, h))
"""

from __future__ import annotations

import argparse
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Default base font sizes
# ---------------------------------------------------------------------------
FONT_SIZES: Dict[str, float] = {
    "title": 14.0,
    "label": 13.0,
    "tick": 12.0,
    "legend": 11.0,
    "suptitle": 16.0,
    "annotation": 10.0,
    "text": 11.0,
}


def get_scaled_sizes(scale: float = 1.0) -> Dict[str, float]:
    """Return FONT_SIZES multiplied by `scale`."""
    return {k: v * scale for k, v in FONT_SIZES.items()}


# ---------------------------------------------------------------------------
# Figure-size arguments  (--fig-width, --fig-height, --fig-scale)
# ---------------------------------------------------------------------------

def add_figsize_args(
    parser: argparse.ArgumentParser,
    default_width: Optional[float] = None,
    default_height: Optional[float] = None,
) -> argparse.ArgumentParser:
    """
    Add --fig-width, --fig-height, --fig-scale to an argparse parser.

    All three are optional; if none is given, the caller's `default` in
    resolve_figsize() is used unchanged.
    """
    group = parser.add_argument_group("Figure size")
    group.add_argument(
        "--fig-width",
        type=float,
        default=None,
        help="Figure width in inches.",
    )
    group.add_argument(
        "--fig-height",
        type=float,
        default=None,
        help="Figure height in inches.",
    )
    group.add_argument(
        "--fig-scale",
        type=float,
        default=None,
        help="Scale both width and height by this factor (e.g. 1.3 = 30%% larger).",
    )
    return parser


# ---------------------------------------------------------------------------
# Font-scale argument  (--font-scale)
# ---------------------------------------------------------------------------


def add_font_scale_arg(parser: argparse.ArgumentParser, default: float = 1.0) -> None:
    """Add --font-scale argument to an argparse parser."""
    parser.add_argument(
        "--font-scale",
        type=float,
        default=default,
        help="Scale factor for title, axis-label, tick-label, and panel-label fonts.",
    )


def resolve_figsize(
    args: argparse.Namespace,
    default: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Compute final (width, height) from argparse args.

    Priority:  --fig-width / --fig-height  override individual dimensions;
               --fig-scale  scales the *default* dimensions.
    """
    w, h = float(default[0]), float(default[1])

    if args.fig_scale is not None:
        w *= args.fig_scale
        h *= args.fig_scale
    if args.fig_width is not None:
        w = args.fig_width
    if args.fig_height is not None:
        h = args.fig_height

    return w, h
