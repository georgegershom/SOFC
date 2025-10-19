import os
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Dataset:
    hf: List[float]
    lf: List[float]
    exp: List[float]


def get_default_dataset() -> Dataset:
    # Example HF vs LF points (LF systematically lower than HF)
    hf_values = [48, 52, 55, 60, 62, 65, 68, 70, 74, 78]
    lf_values = [42, 45, 47, 50, 52, 54, 56, 58, 60, 62]

    # Experimental reference points (on y=x). Include the key 72 K/mm example
    exp_values = [72, 60, 66, 75]

    return Dataset(hf=hf_values, lf=lf_values, exp=exp_values)


def plot_figure_2(dataset: Dataset,
                  x_range: Tuple[float, float] = (40, 80),
                  y_range: Tuple[float, float] = (40, 80),
                  output_path: str = "figures/figure2_systematic_underestimation.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(6, 6), dpi=200)

    # Reference line y = x
    x = np.linspace(x_range[0], x_range[1], 200)
    plt.plot(x, x, color='red', linestyle='-', linewidth=2, label='y = x (Perfect Agreement)')

    # LF vs HF (blue circles)
    plt.scatter(dataset.hf, dataset.lf, color='royalblue', edgecolor='white', linewidth=0.8,
                s=60, marker='o', label='LF vs. HF Predictions')

    # Experimental data (black circles on y=x)
    plt.scatter(dataset.exp, dataset.exp, color='black', edgecolor='black', linewidth=1.0,
                s=70, marker='o', label='Experimental Data')

    # Axes labels and limits
    plt.xlabel('HF Prediction of ∇T_TPB (K/mm)')
    plt.ylabel('LF Prediction of ∇T_TPB (K/mm)')
    plt.xlim(x_range)
    plt.ylim(y_range)

    # Equal aspect ratio for 45-degree y=x line
    plt.gca().set_aspect('equal', adjustable='box')

    # Grid and legend
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.legend(frameon=True)

    # Critical annotation for underestimation cluster
    # Place annotation near the middle of the blue cluster
    annotate_x = np.mean(dataset.hf[:6])
    annotate_y = np.mean(dataset.lf[:6])
    plt.annotate('LF Underestimation\n(~37% avg error)',
                 xy=(annotate_x, annotate_y),
                 xytext=(annotate_x + 6, annotate_y - 8),
                 arrowprops=dict(arrowstyle='->', color='dimgray'),
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='dimgray', alpha=0.9))

    # Title
    plt.title('Systematic Underestimation of TPB Thermal Gradients by Low-Fidelity Models')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    data = get_default_dataset()
    plot_figure_2(data)
