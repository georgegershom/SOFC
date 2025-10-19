#!/usr/bin/env python3
"""
Figure 2: Systematic Underestimation of TPB Thermal Gradients by Low-Fidelity Models
Generates a square scatter plot comparing LF vs. HF predictions of ∇T_TPB with
an experimental reference point and a y=x perfect-agreement line.

Outputs:
- figure2_tpb_gradient_underestimation.png (default)

Optional:
- Set environment variable FIGURE_OUT to change output path
"""

import os
import numpy as np
import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------
# Synthetic data setup
# ---------------------
# HF values chosen so that LF=0.63*HF remains within 40–80 axis limits
hf_values = np.array([64, 66, 68, 70, 72, 74, 76, 78, 80], dtype=float)
underestimation_ratio = 0.63  # ~37% average error
lf_values = np.round(hf_values * underestimation_ratio, 1)

# Experimental reference points (X=Y), key point at (72,72)
exp_values = np.array([72.0])

# ---------------------
# Plot
# ---------------------
fig, ax = plt.subplots(figsize=(6, 6))

# Reference line y=x
ax.plot([40, 80], [40, 80], color="red", linewidth=2.0, label="y = x (Perfect Agreement)")

# LF vs HF scatter (Blue circles)
ax.scatter(
    hf_values,
    lf_values,
    s=70,
    c="tab:blue",
    marker="o",
    edgecolors="white",
    linewidths=0.8,
    alpha=0.95,
    label="LF vs. HF Predictions",
)

# Experimental data (Black circles on the line)
ax.scatter(
    exp_values,
    exp_values,
    s=80,
    c="black",
    marker="o",
    edgecolors="black",
    linewidths=1.2,
    alpha=1.0,
    label="Experimental Data",
)

# Axes, title, labels
ax.set_xlim(40, 80)
ax.set_ylim(40, 80)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Systematic Underestimation of TPB Thermal Gradients by Low-Fidelity Models", pad=12)
ax.set_xlabel(r"HF Prediction of $\nabla T_{\mathrm{TPB}}$ (K/mm)")
ax.set_ylabel(r"LF Prediction of $\nabla T_{\mathrm{TPB}}$ (K/mm)")

# Grid and legend
ax.grid(True, which="both", alpha=0.3)
ax.legend(frameon=True, loc="upper left")

# Critical annotation near the cluster of blue points
ax.annotate(
    "LF Underestimation (~37% avg)",
    xy=(72, 72 * underestimation_ratio),
    xytext=(76.5, 55),
    textcoords="data",
    ha="left",
    va="center",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
)

plt.tight_layout()

# Save
out_path = os.environ.get("FIGURE_OUT", os.path.join(os.getcwd(), "figure2_tpb_gradient_underestimation.png"))
fig.savefig(out_path, dpi=300)
print(f"Saved figure to {out_path}")
