#!/usr/bin/env python3
"""
High-Fidelity Model Validation: Predicting Nickel Coarsening
Generates a bar chart with error bars comparing HF model vs SEM measurements.
Outputs: figures/ni_coarsening_validation.png and .pdf
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class BarSpec:
    label: str
    mean_nm: float
    ci95_nm: float
    color: str


def add_value_labels(ax: plt.Axes, bars: List[Tuple[float, float, BarSpec]]) -> None:
    """Add numeric labels above each bar.

    bars: list of tuples (x_pos, height, spec)
    """
    for x_pos, height, spec in bars:
        text = f"{spec.mean_nm:.1f} nm"
        ax.text(
            x_pos,
            height + max(0.6, 0.05 * height),
            text,
            ha="center",
            va="bottom",
            fontsize=11,
            color="#222222",
        )


def add_connection_bracket(ax: plt.Axes, x0: float, x1: float, y: float, text: str | None = None) -> None:
    """Draw a light bracket between two x positions at height y.

    Optionally add centered text above the bracket.
    """
    bracket_color = "#666666"
    cap = 0.05
    ax.plot([x0, x0, x1, x1], [y, y + cap, y + cap, y], color=bracket_color, lw=1.2)
    if text:
        ax.text((x0 + x1) / 2.0, y + cap + 0.4, text, ha="center", va="bottom", fontsize=11, color=bracket_color)


def main() -> None:
    # Example numbers consistent with description
    hf = BarSpec(label="High-Fidelity (HF) Model", mean_nm=15.0, ci95_nm=1.5, color="#D62728")  # red
    sem = BarSpec(label="Scanning Electron Microscopy (SEM)", mean_nm=14.9, ci95_nm=2.0, color="#222222")  # black/grey

    specs = [hf, sem]

    # Plot settings
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)

    x = np.arange(len(specs), dtype=float)
    bar_width = 0.6

    heights = [s.mean_nm for s in specs]
    errors = [s.ci95_nm for s in specs]
    colors = [s.color for s in specs]
    labels = [s.label for s in specs]

    bars = ax.bar(
        x,
        heights,
        yerr=errors,
        width=bar_width,
        color=colors,
        edgecolor="#333333",
        linewidth=0.8,
        capsize=8,
        error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
    )

    # X axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(["High-Fidelity (HF) Model", "Scanning Electron Microscopy (SEM)"], fontsize=11)
    ax.set_xlabel("Measurement Method", fontsize=12)

    # Y axis label and limits
    ax.set_ylabel("Ni Coarsening, Δd_Ni (nm)", fontsize=12)

    y_max = max(h + e for h, e in zip(heights, errors))
    y_lim_top = max(20.0, math.ceil((y_max + 3.0) / 2.0) * 2.0)  # ensure a bit of headroom
    ax.set_ylim(0, y_lim_top)

    # Add value labels
    bar_info: List[Tuple[float, float, BarSpec]] = []
    for rect, spec in zip(bars, specs):
        x_center = rect.get_x() + rect.get_width() / 2.0
        bar_info.append((x_center, rect.get_height(), spec))
    add_value_labels(ax, bar_info)

    # Add optional connection bracket to emphasize closeness
    bracket_y = y_max + 1.2
    add_connection_bracket(ax, x[0], x[1], bracket_y, text=None)

    # R^2 annotation
    ax.text(
        0.5,
        0.92,
        "Validation R² = 0.97",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="#BBBBBB", boxstyle="round,pad=0.3"),
    )

    # Legend (optional, but included for clarity)
    ax.legend(
        [bars[0], bars[1]],
        [labels[0], labels[1]],
        fontsize=10,
        frameon=True,
        framealpha=0.9,
    )

    # Tight grid only on y-axis
    ax.grid(axis="y", which="major", color="#DDDDDD", linewidth=1.0)
    ax.grid(axis="y", which="minor", color="#EEEEEE", linewidth=0.8)
    ax.minorticks_on()

    # Save outputs
    out_base = "/workspace/figures/ni_coarsening_validation"
    fig.savefig(f"{out_base}.png", dpi=300)
    fig.savefig(f"{out_base}.pdf")

    print(f"Saved figure to {out_base}.png and {out_base}.pdf")


if __name__ == "__main__":
    main()
