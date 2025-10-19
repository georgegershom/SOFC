#!/usr/bin/env python3
"""
Generate a publication-quality bar chart comparing High-Fidelity (HF) model
predictions of Nickel (Ni) nanoparticle coarsening (Δd_Ni) against
Scanning Electron Microscopy (SEM) measurements.

Outputs:
- figures/ni_coarsening_validation.png
- figures/ni_coarsening_validation.svg
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def add_value_labels(ax: plt.Axes, bars: list[plt.Rectangle]) -> None:
    for rect in bars:
        height = rect.get_height()
        ax.annotate(
            f"{height:.1f} nm",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )


def draw_bracket(ax: plt.Axes, x0: float, x1: float, y: float, tick: float = 0.15) -> None:
    # Draw a subtle bracket (┐──┌) connecting two bars at height y
    ax.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick], color="0.3", linewidth=1.0)


def main() -> None:
    # Data (example means and 95% CI, consistent with description)
    labels = [
        "High-Fidelity (HF) Model",
        "Scanning Electron Microscopy (SEM)",
    ]
    means = np.array([15.1, 14.9], dtype=float)
    ci95 = np.array([1.5, 2.0], dtype=float)

    colors = ["#d62728", "#000000"]  # red for HF, black for SEM

    # Prepare figure
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    x = np.arange(len(labels))
    width = 0.6

    bars = ax.bar(
        x,
        means,
        width=width,
        yerr=ci95,
        capsize=8,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.95,
    )

    # Axes and grid
    ax.set_xlabel("Measurement Method")
    ax.set_ylabel("Ni Coarsening, Δd$_{Ni}$ (nm)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # Ensure headroom above the tallest error bar; default to 0–20 nm as suggested
    top = float(np.max(means + ci95))
    y_max = max(20.0, top + 2.0)
    ax.set_ylim(0.0, y_max)

    ax.yaxis.grid(True, color="0.85", linestyle="-", linewidth=0.8)
    ax.set_axisbelow(True)

    # Legend (optional but included for clarity)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[0], edgecolor="black", linewidth=0.8),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[1], edgecolor="black", linewidth=0.8),
    ]
    ax.legend(legend_handles, ["HF Model", "SEM Experiment"], frameon=False, loc="upper left")

    # R² annotation
    r2 = 0.97
    ax.text(
        0.5,
        0.88,
        f"Validation R² = {r2:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
    )

    # Optional bracket emphasizing closeness
    bracket_y = top + 0.8
    draw_bracket(ax, x[0], x[1], bracket_y)

    # Data value labels on bars
    add_value_labels(ax, list(bars))

    # Title
    fig.suptitle("High-Fidelity Model Validation: Predicting Nickel Coarsening", fontsize=12)

    fig.tight_layout(pad=1.2)

    # Outputs
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    for ext in ("png", "svg"):
        out_path = out_dir / f"ni_coarsening_validation.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    main()
