#!/usr/bin/env python3
"""
Sintering profile design and stress–shape trade-off figure generator

Outputs a two-panel figure:
- Panel A: staged sintering temperature profiles T(t) with ramp → soak → symmetric cool-down
- Panel B: Pareto map of residual Lagrangian strain (µε) vs out-of-plane warpage (µm)

Features
- Optional CSV ingestion (auto-detects columns) for profiles and outcomes; else generates realistic synthetic data
- Physics-inspired surrogate to map (ramp_rate, T_soak, soak_time) → (residual_strain, warpage)
- Pareto frontier identification and overlay with confidence shading
- Abaqus-like professional styling (dark-on-light, fine grid, subtle gradients, engineering ticks)
- Reproducible randomness with seed
- CLI flags for customization

CSV schema (auto-detected, case-insensitive, flexible):
- profiles.csv (optional): columns ~ [id,label,ramp_rate_C_per_min,T_soak_C,soak_time_min,ambient_C]
- outcomes.csv (optional): columns ~ [id,residual_strain_microepsilon,warpage_microns]
If provided, outcomes rows are matched by id to profiles. If outcomes missing, they are modeled.

Examples
  python scripts/sintering_tradeoff_figure.py --save fig_sinter_tradeoff.png --dpi 300
  python scripts/sintering_tradeoff_figure.py --profiles data/profiles.csv --outcomes data/outcomes.csv

"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd

# ---------------------------
# Styling: Abaqus-like preset
# ---------------------------

def apply_engineering_style():
    mpl.rcParams.update({
        "figure.figsize": (12, 5.5),
        "figure.dpi": 120,
        "axes.facecolor": "#fbfbfc",
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#111111",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "grid.color": "#cfcfd4",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "font.size": 11.5,
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.facecolor": "#f5f6f7",
        "legend.edgecolor": "#d1d2d6",
        "lines.linewidth": 2.2,
        "errorbar.capsize": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class Profile:
    id: str
    ramp_rate_C_per_min: float  # positive
    T_soak_C: float
    soak_time_min: float
    ambient_C: float = 25.0
    label: Optional[str] = None

    def generate_time_temperature(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate symmetric ramp → soak → cool-down profile."""
        r = max(self.ramp_rate_C_per_min, 1e-6)
        delta_T = max(self.T_soak_C - self.ambient_C, 0.0)
        t_ramp = delta_T / r
        t_soak = max(self.soak_time_min, 0.0)
        # Time grid: resolve ramps and soak
        n_ramp = max(int(t_ramp * 2) + 10, 40)  # ~2 pts/min + base
        n_soak = max(int(t_soak * 2) + 10, 30)
        t_up = np.linspace(0.0, t_ramp, n_ramp, endpoint=False)
        t_hold = np.linspace(t_ramp, t_ramp + t_soak, n_soak, endpoint=False)
        t_down = np.linspace(t_ramp + t_soak, 2 * t_ramp + t_soak, n_ramp)
        T_up = self.ambient_C + r * t_up
        T_hold = np.full_like(t_hold, self.T_soak_C)
        T_down = self.T_soak_C - r * (t_down - (t_ramp + t_soak))
        t = np.concatenate([t_up, t_hold, t_down])
        T = np.concatenate([T_up, T_hold, T_down])
        return t, T


@dataclass
class Outcome:
    residual_strain_microepsilon: float
    warpage_microns: float


# -------------------------------------
# Surrogate physics-inspired outcome map
# -------------------------------------

def simulate_outcome(profile: Profile, 
                     material_params: Optional[Dict[str, float]] = None,
                     rng: Optional[np.random.Generator] = None) -> Outcome:
    """Map process settings to residual strain and warpage.

    Model rationale (surrogate):
    - Residual strain decreases with high soak temperature and with integrated high-T time (more creep/relaxation), but increases with thermal gradients (fast ramps) and insufficient time.
    - Warpage increases with higher peak temperature and faster ramps (curvature + TEC mismatch), and decreases with symmetric cool-down and slower ramps.

    Formulation (dimensionally consistent up to scaling):
      eps_res ~ eps0 * exp(-alpha * T_soak_K) * (1 / (1 + beta * t_eff)) + gamma * (ramp_rate / r_ref)^p
      warp ~ w0 + cT * (T_soak_K - T_ref) + cr * (ramp_rate / r_ref)^q - ct * log(1 + t_eff / t_ref)

    Added small noise to mimic measurement/simulation scatter.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    mp = {
        "eps0": 2200.0,       # µε baseline
        "alpha": 1.2e-3,      # 1/K effect of temperature on creep
        "beta": 5e-3,         # 1/min effect of time at temp
        "gamma": 180.0,       # µε penalty from fast ramps
        "p": 1.1,
        "w0": 15.0,           # µm baseline
        "cT": 0.020,          # µm/°C effect of peak temp beyond reference
        "cr": 8.0,            # µm penalty from fast ramps
        "q": 1.2,
        "ct": 6.0,            # µm benefit from longer effective time
        "T_ref": 900.0,       # °C reference for warpage scaling
        "r_ref": 1.0,         # °C/min reference ramp
        "t_ref": 60.0,        # min reference time scale
        "k_soak_eff": 0.6,    # effective weight of soak time
    }
    if material_params:
        mp.update(material_params)

    T_soak_K = profile.T_soak_C + 273.15
    r = max(profile.ramp_rate_C_per_min, 1e-6)
    delta_T = max(profile.T_soak_C - profile.ambient_C, 0.0)

    t_ramp = delta_T / r
    t_eff = mp["k_soak_eff"] * profile.soak_time_min + 0.25 * t_ramp  # heating contributes less to high-T effect

    eps_relax = mp["eps0"] * math.exp(-mp["alpha"] * T_soak_K) * (1.0 / (1.0 + mp["beta"] * t_eff))
    eps_ramp_penalty = mp["gamma"] * (r / mp["r_ref"]) ** mp["p"]
    residual_strain = max(eps_relax + eps_ramp_penalty, 0.0)

    warp_temp = mp["w0"] + mp["cT"] * (profile.T_soak_C - mp["T_ref"])  # linear sensitivity to peak temp
    warp_ramp = mp["cr"] * (r / mp["r_ref"]) ** mp["q"]
    warp_time_benefit = mp["ct"] * math.log1p(t_eff / mp["t_ref"])  # longer time lowers warpage slightly
    warpage = max(warp_temp + warp_ramp - warp_time_benefit, 0.0)

    # add modest scatter to look realistic
    residual_strain += rng.normal(0.0, 25.0)
    warpage += rng.normal(0.0, 1.2)

    return Outcome(residual_strain_microepsilon=float(residual_strain),
                   warpage_microns=float(warpage))


# ---------------------------
# CSV ingestion (optional)
# ---------------------------

PROFILE_COL_ALIASES = {
    "id": ["id", "profile_id", "name"],
    "label": ["label", "legend", "title"],
    "ramp_rate_C_per_min": ["ramp_rate_c_per_min", "ramp_rate", "ramp", "drdt", "dTdt"],
    "T_soak_C": ["t_soak_c", "tsoak", "t_peak_c", "t_peak", "tmax"],
    "soak_time_min": ["soak_time_min", "soak", "hold", "hold_time_min", "dwell"],
    "ambient_C": ["ambient_c", "ambient", "t_ambient", "t0"],
}

OUTCOME_COL_ALIASES = {
    "id": ["id", "profile_id", "name"],
    "residual_strain_microepsilon": ["residual_strain_microepsilon", "residual_strain", "strain_uE", "uE"],
    "warpage_microns": ["warpage_microns", "warpage", "warp_um", "um"],
}


def _map_columns(df: pd.DataFrame, aliases: Dict[str, List[str]]) -> pd.DataFrame:
    lower_map = {c.lower(): c for c in df.columns}
    mapped = {}
    for target, candidates in aliases.items():
        for cand in candidates:
            if cand.lower() in lower_map:
                mapped[target] = lower_map[cand.lower()]
                break
    return df.rename(columns=mapped)


def load_profiles_csv(path: str) -> List[Profile]:
    df = pd.read_csv(path)
    df = _map_columns(df, PROFILE_COL_ALIASES)
    required = ["id", "ramp_rate_C_per_min", "T_soak_C", "soak_time_min"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"profiles CSV missing required column: {col}")
    profiles: List[Profile] = []
    for _, row in df.iterrows():
        profiles.append(Profile(
            id=str(row["id"]),
            ramp_rate_C_per_min=float(row["ramp_rate_C_per_min"]),
            T_soak_C=float(row["T_soak_C"]),
            soak_time_min=float(row["soak_time_min"]),
            ambient_C=float(row.get("ambient_C", 25.0)),
            label=str(row.get("label", None)) if not pd.isna(row.get("label", None)) else None,
        ))
    return profiles


def load_outcomes_csv(path: str) -> Dict[str, Outcome]:
    df = pd.read_csv(path)
    df = _map_columns(df, OUTCOME_COL_ALIASES)
    required = ["id", "residual_strain_microepsilon", "warpage_microns"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"outcomes CSV missing required column: {col}")
    outcomes: Dict[str, Outcome] = {}
    for _, row in df.iterrows():
        outcomes[str(row["id"]) ] = Outcome(
            residual_strain_microepsilon=float(row["residual_strain_microepsilon"]),
            warpage_microns=float(row["warpage_microns"]),
        )
    return outcomes


# ---------------------------
# Synthetic default dataset
# ---------------------------

def generate_default_profiles() -> List[Profile]:
    # Representative SOFC-like sintering ranges
    defaults = [
        Profile(id="P1", ramp_rate_C_per_min=1.0, T_soak_C=900.0, soak_time_min=60, label="P1: 1.0, 900"),
        Profile(id="P2", ramp_rate_C_per_min=1.5, T_soak_C=1000.0, soak_time_min=90, label="P2: 1.5, 1000"),
        Profile(id="P3", ramp_rate_C_per_min=2.0, T_soak_C=1050.0, soak_time_min=120, label="P3: 2.0, 1050"),
        # extra realistic variants to populate Pareto cloud
        Profile(id="P4", ramp_rate_C_per_min=0.8, T_soak_C=925.0, soak_time_min=80, label="P4: 0.8, 925"),
        Profile(id="P5", ramp_rate_C_per_min=1.2, T_soak_C=975.0, soak_time_min=75, label="P5: 1.2, 975"),
        Profile(id="P6", ramp_rate_C_per_min=1.8, T_soak_C=1025.0, soak_time_min=110, label="P6: 1.8, 1025"),
        Profile(id="P7", ramp_rate_C_per_min=0.6, T_soak_C=880.0, soak_time_min=60, label="P7: 0.6, 880"),
        Profile(id="P8", ramp_rate_C_per_min=2.2, T_soak_C=1075.0, soak_time_min=90, label="P8: 2.2, 1075"),
    ]
    return defaults


# ---------------------------
# Pareto frontier computation
# ---------------------------

def pareto_front(points: np.ndarray) -> np.ndarray:
    """Compute indices of Pareto-efficient points for minimization in 2D."""
    # points: shape (N, 2) for (x,y) = (residual_strain, warpage)
    N = points.shape[0]
    is_efficient = np.ones(N, dtype=bool)
    for i in range(N):
        if not is_efficient[i]:
            continue
        # eliminate points dominated by i
        dominated = (points[:, 0] <= points[i, 0]) & (points[:, 1] <= points[i, 1]) & (
            (points[:, 0] < points[i, 0]) | (points[:, 1] < points[i, 1])
        )
        is_efficient[dominated] = False
        is_efficient[i] = True  # keep self
    return np.nonzero(is_efficient)[0]


# ---------------------------
# Plotting
# ---------------------------

def plot_figure(profiles: List[Profile],
                outcomes: Dict[str, Outcome],
                save: Optional[str] = None,
                dpi: int = 200,
                acceptance_lines: Optional[Tuple[Optional[float], Optional[float]]] = None,
                show_model_curve: bool = True,
                rng_seed: int = 42) -> None:
    apply_engineering_style()

    fig = plt.figure(constrained_layout=True)
    widths = [3.1, 2.4]
    gs = fig.add_gridspec(1, 2, width_ratios=widths)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # Minor ticks and fine grid for engineering look
    for ax in (axA, axB):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which="major", linewidth=0.6)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.5)

    # Panel A: T(t)
    cmap = mpl.colormaps.get_cmap("viridis")
    prof_colors = {}
    n = len(profiles)
    for idx, prof in enumerate(profiles):
        color = cmap(0.15 + 0.7 * idx / max(n - 1, 1))
        prof_colors[prof.id] = color
        t, T = prof.generate_time_temperature()
        axA.plot(t, T, color=color, label=prof.label or prof.id)

    axA.set_title("Panel A — Staged sintering profiles T(t)")
    axA.set_xlabel("Time (min)")
    axA.set_ylabel("Temperature (°C)")
    axA.legend(title="Profile: dT/dt (°C/min), T_soak (°C)", ncols=2, fontsize=10)

    # Panel B: residual strain vs warpage
    # Gather points
    xs, ys, labels = [], [], []
    for prof in profiles:
        oc = outcomes[prof.id]
        xs.append(oc.residual_strain_microepsilon)
        ys.append(oc.warpage_microns)
        labels.append(prof.id)

    xs = np.array(xs)
    ys = np.array(ys)

    # scatter points
    for i, prof in enumerate(profiles):
        axB.scatter(xs[i], ys[i], s=60, color=prof_colors[prof.id], edgecolor="#222", linewidth=0.8, zorder=3)
        axB.text(xs[i] + 8, ys[i] + 0.6, prof.id, color="#222", fontsize=10)

    axB.set_title("Panel B — Pareto: residual strain vs warpage")
    axB.set_xlabel("Residual Lagrangian strain (µε)")
    axB.set_ylabel("Out-of-plane warpage (µm)")

    # Acceptance lines if provided: (vertical strain cap, horizontal warpage cap)
    if acceptance_lines:
        xcap, ycap = acceptance_lines
        if xcap is not None:
            axB.axvline(xcap, color="#9a2e2e", linestyle="--", linewidth=1.5, label=f"strain cap = {xcap:.0f} µε")
        if ycap is not None:
            axB.axhline(ycap, color="#2e6b9a", linestyle=":", linewidth=1.8, label=f"warpage cap = {ycap:.1f} µm")

    # Pareto frontier
    pts = np.c_[xs, ys]
    idx_front = pareto_front(pts)
    pareto_pts = pts[idx_front]
    # Sort by x then y
    order = np.argsort(pareto_pts[:, 0])
    pareto_pts = pareto_pts[order]
    axB.plot(pareto_pts[:, 0], pareto_pts[:, 1], color="#333", linewidth=2.0, label="Pareto frontier", zorder=2)

    # Model curve (smooth param sweep) to "prove" surrogate behavior
    if show_model_curve:
        # Sweep a param ray: increase T_soak while varying ramp rate along typical range
        rng = np.random.default_rng(rng_seed)
        T_grid = np.linspace(min(p.T_soak_C for p in profiles) - 20,
                             max(p.T_soak_C for p in profiles) + 40, 20)
        r_grid = np.linspace(min(p.ramp_rate_C_per_min for p in profiles) - 0.3,
                             max(p.ramp_rate_C_per_min for p in profiles) + 0.3, 20)
        r_grid = np.clip(r_grid, 0.4, None)
        samples: List[Tuple[float, float]] = []
        base = profiles[0]
        for Tval, rval in zip(T_grid, r_grid):
            trial = Profile(id="model", ramp_rate_C_per_min=float(rval), T_soak_C=float(Tval),
                            soak_time_min=float(np.interp(Tval, [T_grid[0], T_grid[-1]], [60, 110])),
                            ambient_C=base.ambient_C)
            oc = simulate_outcome(trial, rng=rng)
            samples.append((oc.residual_strain_microepsilon, oc.warpage_microns))
        samples = np.array(samples)
        # smooth curve fit if SciPy available, else polyline
        try:
            from scipy.interpolate import splprep, splev  # type: ignore
            tck, _ = splprep([samples[:, 0], samples[:, 1]], s=20.0)
            unew = np.linspace(0, 1, 200)
            x_smooth, y_smooth = splev(unew, tck)
            axB.plot(x_smooth, y_smooth, color="#0b5", linewidth=2.0, alpha=0.9, label="Model trend")
        except Exception:
            # fallback: straight polyline
            axB.plot(samples[:, 0], samples[:, 1], color="#0b5", linewidth=1.8, alpha=0.9, label="Model trend")

    axB.legend(loc="best", fontsize=10)

    # Panel labels
    fig.text(0.01, 0.98, "A", fontsize=16, fontweight="bold", va="top", ha="left")
    fig.text(0.52, 0.98, "B", fontsize=16, fontweight="bold", va="top", ha="left")

    # Tighten axes ranges with padding
    xpad = (xs.max() - xs.min()) * 0.15 if len(xs) > 1 else 50
    ypad = (ys.max() - ys.min()) * 0.15 if len(ys) > 1 else 5
    axB.set_xlim(max(0, xs.min() - xpad), xs.max() + xpad)
    axB.set_ylim(max(0, ys.min() - ypad), ys.max() + ypad)

    # Shade Pareto-efficient region (lower-left region under frontier)
    try:
        x0, x1 = axB.get_xlim()
        y0, y1 = axB.get_ylim()
        x_poly = np.concatenate([[x0], pareto_pts[:, 0], [pareto_pts[-1, 0]]])
        y_poly = np.concatenate([[y0], pareto_pts[:, 1], [y0]])
        axB.fill(x_poly, y_poly, facecolor="#00bb55", alpha=0.06, edgecolor="none", zorder=1)
    except Exception:
        pass

    if save:
        plt.savefig(save, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate sintering profile and Pareto trade-off figure.")
    parser.add_argument("--profiles", type=str, default=None, help="Path to profiles CSV")
    parser.add_argument("--outcomes", type=str, default=None, help="Path to outcomes CSV")
    parser.add_argument("--save", type=str, default=None, help="Path to save the figure (e.g., fig.png)")
    parser.add_argument("--dpi", type=int, default=220, help="Output DPI for saved figure")
    parser.add_argument("--xcap", type=float, default=None, help="Residual strain cap (µε) vertical line")
    parser.add_argument("--ycap", type=float, default=None, help="Warpage cap (µm) horizontal line")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-model-curve", action="store_true", help="Disable model trend curve overlay")

    args = parser.parse_args()

    # Load or synthesize profiles
    if args.profiles and os.path.isfile(args.profiles):
        try:
            profiles = load_profiles_csv(args.profiles)
        except Exception as e:
            print(f"Failed to load profiles CSV: {e}. Falling back to synthetic defaults.")
            profiles = generate_default_profiles()
    else:
        profiles = generate_default_profiles()

    # Load or simulate outcomes
    outcomes: Dict[str, Outcome] = {}
    if args.outcomes and os.path.isfile(args.outcomes):
        try:
            outcomes = load_outcomes_csv(args.outcomes)
        except Exception as e:
            print(f"Failed to load outcomes CSV: {e}. Recomputing via model.")
    
    rng = np.random.default_rng(args.seed)
    for prof in profiles:
        if prof.id in outcomes:
            continue
        outcomes[prof.id] = simulate_outcome(prof, rng=rng)

    acceptance = (args.xcap, args.ycap)
    plot_figure(
        profiles=profiles,
        outcomes=outcomes,
        save=args.save,
        dpi=args.dpi,
        acceptance_lines=acceptance,
        show_model_curve=not args.no_model_curve,
        rng_seed=args.seed,
    )


if __name__ == "__main__":
    main()
