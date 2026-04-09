#!/usr/bin/env python3
"""
figure_2_1_sofc.py
==================
Pure-Python / Matplotlib reproduction of **Figure 2.1** from:

  "A Multi-Fidelity SOFC Dataset and Transformer Neural Operator Framework
   for Cross-Scale Stress, Damage, and Durability Digital Twins"

Figure 2.1 — Reference SOFC Architecture, Coupled-Physics Envelope,
              and Target-Output Hierarchy

Five panels:
  (a) Layered planar SOFC architecture with layer labels, thickness
      notation, interface names, and canonical domain symbols.
  (b) Coupled-physics dependency graph: electrochemical heat generation,
      temperature gradients, thermal expansion mismatch, creep relaxation,
      microstructural coarsening, and interface damage.
  (c) Target-output hierarchy: operating conditions → full-field outputs
      → damage surrogates → life-relevant outputs.
  (d) Constitutive dependency graph: T, μ, r_Ni, λ_TPB, d_int →
      effective constitutive response C(T,μ)·σ, with symbol glossary.
  (e) Electrolyte stress comparison: linear elastic vs. viscoelastic
      model with fracture threshold (publication-quality quantitative plot).

Dependencies (all standard):
    matplotlib  >=  3.5
    numpy       >=  1.21

Run:
    python figure_2_1_sofc.py

Output:
    figure_2_1_sofc.png  (300 dpi, publication quality)
    Also displays interactively via plt.show()
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as path_effects
import numpy as np

# ─── Publication-quality global settings ──────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset":   "stix",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "axes.linewidth":     1.0,          # 1 pt solid black axes
    "xtick.major.width":  1.0,
    "ytick.major.width":  1.0,
    "xtick.minor.width":  0.6,
    "ytick.minor.width":  0.6,
    "xtick.direction":    "in",         # ticks face inward
    "ytick.direction":    "in",
    "xtick.major.size":   5,
    "ytick.major.size":   5,
    "xtick.minor.size":   3,
    "ytick.minor.size":   3,
    "axes.edgecolor":     "black",
    "axes.grid":          False,
    "legend.frameon":     False,        # legend has no outer border
    "legend.fontsize":    8,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "figure.dpi":         150,
})


# ─── Colour palette ──────────────────────────────────────────────────────────
C_AN   = "#c8a96e"   # warm tan   – Ni-YSZ anode support
C_ELY  = "#7fb3d8"   # medium blue – 8YSZ electrolyte
C_GDC  = "#7bc07a"   # green      – GDC interlayer
C_CAT  = "#d97373"   # rose       – LSCF cathode
C_IFACE = "#4a5568"  # interface line colour

# Brighter, more saturated palette for physics nodes
C_PHYS = {
    "echem": "#1565c0",
    "temp":  "#c62828",
    "mech":  "#2e7d32",
    "creep": "#e65100",
    "micro": "#6a1b9a",
    "dam":   "#ad1457",
}
C_HIER = {
    "op":    "#1565c0",
    "field": "#2e7d32",
    "dmg":   "#c62828",
    "life":  "#6a1b9a",
}
C_CONST = {
    "T":     "#c62828",
    "mu":    "#1565c0",
    "rNi":   "#6a1b9a",
    "tpb":   "#2e7d32",
    "dint":  "#ad1457",
    "sigma": "#263238",
    "C":     "#e65100",
}


# ─── Helper: bold panel label ────────────────────────────────────────────────
def panel_label(ax, txt, x=-0.05, y=1.05):
    """Bold panel tag placed outside the top-left corner of an axis."""
    t = ax.text(x, y, txt, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left",
                color="#1a1a2e", fontfamily="serif")
    t.set_path_effects([
        path_effects.withStroke(linewidth=2.5, foreground="white"),
    ])


# ─── Helper: draw a rounded-rectangle node ───────────────────────────────────
def draw_node(ax, x, y, w, h, text, facecolor, textcolor="white",
              fontsize=7.0, edgecolor=None, linewidth=1.2, alpha=0.92,
              zorder=5, shadow=True):
    """Draw a publication-quality rounded-rectangle node with optional shadow."""
    if edgecolor is None:
        edgecolor = facecolor
    if shadow:
        shadow_rect = FancyBboxPatch(
            (x - w / 2 + 0.06, y - h / 2 - 0.06), w, h,
            boxstyle="round,pad=0.12",
            linewidth=0, facecolor="#00000018", zorder=zorder - 1,
        )
        ax.add_patch(shadow_rect)
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.12",
        linewidth=linewidth, edgecolor=edgecolor,
        facecolor=facecolor, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(rect)
    ax.text(x, y, text,
            ha="center", va="center",
            fontsize=fontsize, color=textcolor, fontweight="bold",
            zorder=zorder + 1, multialignment="center",
            fontfamily="serif")


# ─── Helper: curved arrow between two points ─────────────────────────────────
def curved_arrow(ax, x0, y0, x1, y1, color="#666666", lw=1.4,
                 rad=0.12, zorder=3, headwidth=6, headlength=5):
    """Draw a curved arrow between two points."""
    ax.annotate("",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color, lw=lw,
                    connectionstyle=f"arc3,rad={rad}",
                    mutation_scale=12,
                ),
                zorder=zorder)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure layout
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14), facecolor="white")

# Subtle background gradient effect via a rectangle
bg_rect = mpatches.FancyBboxPatch(
    (0.005, 0.005), 0.99, 0.99,
    boxstyle="round,pad=0.005",
    transform=fig.transFigure, facecolor="#fafbfd",
    edgecolor="#e0e0e0", linewidth=0.8, zorder=0,
)
fig.patches.append(bg_rect)

fig.subplots_adjust(left=0.04, right=0.97,
                    bottom=0.04, top=0.91,
                    hspace=0.40, wspace=0.32)

# Title with professional styling
fig.text(
    0.5, 0.965,
    "Figure 2.1  ·  Reference SOFC Architecture, "
    "Coupled-Physics Envelope, and Target-Output Hierarchy",
    ha="center", va="top", fontsize=14,
    fontweight="bold", color="#1a1a2e",
    fontfamily="serif",
)
fig.text(
    0.5, 0.945,
    "A Multi-Fidelity SOFC Dataset and Transformer Neural Operator "
    "Framework for Cross-Scale Digital Twins",
    ha="center", va="top", fontsize=9,
    color="#6b7280", style="italic", fontfamily="serif",
)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL (a) – Layered planar SOFC cross-section
# ═══════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_axes([0.04, 0.52, 0.27, 0.38])
ax_a.set_xlim(-0.5, 10.5)
ax_a.set_ylim(-0.5, 10.5)
ax_a.axis("off")

# Panel title
ax_a.text(0.5, 1.04,
          "(a)  Planar SOFC Architecture & Domain $\\Omega$",
          transform=ax_a.transAxes,
          fontsize=11, fontweight="bold", ha="center", va="bottom",
          color="#1a1a2e", fontfamily="serif")

# Visual heights (not to true scale; thin layers made legible)
vis = {"an": 4.6, "ely": 1.3, "gdc": 0.9, "cat": 2.6}
gap = 0.15   # inter-layer gap for interface markers

layers = [
    ("an",  "Anode support\n(Ni–YSZ)",  "500 μm", C_AN,  "$\\Omega_{\\mathrm{an}}$"),
    ("ely", "Electrolyte\n(8YSZ)",       "10 μm",  C_ELY, "$\\Omega_{\\mathrm{ely}}$"),
    ("gdc", "GDC interlayer",            "5 μm",   C_GDC, "$\\Omega_{\\mathrm{gdc}}$"),
    ("cat", "Cathode\n(LSCF)",           "40 μm",  C_CAT, "$\\Omega_{\\mathrm{cat}}$"),
]
interface_syms = [
    "$\\Gamma_{\\mathrm{an|ely}}$",
    "$\\Gamma_{\\mathrm{ely|gdc}}$",
    "$\\Gamma_{\\mathrm{gdc|cat}}$",
]

y_cursor = 0.5
for i, (key, label, thick, color, domain) in enumerate(layers):
    h = vis[key]
    # Shadow
    shadow = FancyBboxPatch(
        (1.08, y_cursor - 0.05), 7.0, h,
        boxstyle="round,pad=0.04",
        linewidth=0, facecolor="#00000012", zorder=2,
    )
    ax_a.add_patch(shadow)
    # Layer rectangle
    rect = FancyBboxPatch(
        (1.0, y_cursor), 7.0, h,
        boxstyle="round,pad=0.04",
        linewidth=1.4, edgecolor="#3a3a3a",
        facecolor=color, zorder=3,
    )
    ax_a.add_patch(rect)
    # Subtle hatching for texture
    if key == "an":
        for yy in np.arange(y_cursor + 0.3, y_cursor + h - 0.1, 0.5):
            ax_a.plot([1.3, 7.7], [yy, yy],
                      color="#00000008", linewidth=0.6, zorder=3)
    # Layer label
    t = ax_a.text(4.5, y_cursor + h / 2, label,
                  ha="center", va="center",
                  fontsize=9, color="#1a1a2e", fontweight="bold", zorder=4,
                  fontfamily="serif")
    t.set_path_effects([
        path_effects.withStroke(linewidth=2, foreground="white"),
    ])
    # Thickness annotation
    ax_a.text(8.35, y_cursor + h / 2, thick,
              ha="left", va="center",
              fontsize=7.5, color="#555", style="italic", zorder=4,
              fontfamily="serif")
    # Domain symbol
    ax_a.text(0.65, y_cursor + h / 2, domain,
              ha="right", va="center",
              fontsize=8.5, color="#444", zorder=4)
    # Interface dashed line + label
    if i < len(layers) - 1:
        iy = y_cursor + h + gap / 2
        ax_a.plot([0.8, 8.2], [iy, iy],
                  color=C_IFACE, linewidth=1.6, linestyle=(0, (5, 3)),
                  zorder=5, alpha=0.85)
        ax_a.text(4.5, iy + 0.22, interface_syms[i],
                  ha="center", va="bottom",
                  fontsize=7.5, color=C_IFACE,
                  fontstyle="italic", zorder=6,
                  fontfamily="serif",
                  bbox=dict(facecolor="white", edgecolor="none",
                            alpha=0.7, pad=0.8))
    y_cursor += h + gap

# Axis arrows (through-thickness and in-plane)
ax_a.annotate("", xy=(9.5, 9.8), xytext=(9.5, 0.2),
              arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5,
                              mutation_scale=14))
ax_a.text(9.85, 5.0, "$x_2$\n(through-\nthickness)",
          ha="center", va="center", fontsize=7.5, color="#333",
          fontfamily="serif", fontweight="bold")
ax_a.annotate("", xy=(8.3, -0.1), xytext=(0.7, -0.1),
              arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5,
                              mutation_scale=14))
ax_a.text(4.5, -0.35, "$x_1$ (in-plane)",
          ha="center", va="top", fontsize=7.5, color="#333",
          fontfamily="serif", fontweight="bold")

# Constrained edges annotation
ax_a.text(0.03, 0.01, "* constrained edges",
          transform=ax_a.transAxes,
          fontsize=7, color="#c62828", va="bottom",
          fontweight="bold", fontfamily="serif")

panel_label(ax_a, "(a)")


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL (b) – Coupled-physics dependency graph
# ═══════════════════════════════════════════════════════════════════════════════
ax_b = fig.add_axes([0.35, 0.52, 0.30, 0.38])
ax_b.set_xlim(-0.5, 10.5)
ax_b.set_ylim(-0.5, 10.5)
ax_b.axis("off")

ax_b.text(0.5, 1.04,
          "(b)  Coupled-Physics Dependency Graph",
          transform=ax_b.transAxes,
          fontsize=11, fontweight="bold", ha="center", va="bottom",
          color="#1a1a2e", fontfamily="serif")

# Node positions: (x, y, colour, label)
nodes_b = {
    "echem": (5.0, 9.0, C_PHYS["echem"],
              "Electrochemical\nheat gen.\n$q_{\\mathrm{act}},\\,q_{\\mathrm{ohm}}$"),
    "temp":  (1.8, 6.5, C_PHYS["temp"],
              "Temperature\ngradients\n$\\nabla T$"),
    "mech":  (8.2, 6.5, C_PHYS["mech"],
              "Thermal expansion\nmismatch\n$\\varepsilon_{\\mathrm{th}}$"),
    "creep": (2.0, 3.2, C_PHYS["creep"],
              "Creep\nrelaxation\n$\\varepsilon_{\\mathrm{cr}}$"),
    "micro": (8.0, 3.2, C_PHYS["micro"],
              "Microstructural\ncoarsening\n$r_{\\mathrm{Ni}}$"),
    "dam":   (5.0, 0.8, C_PHYS["dam"],
              "Interface\ndamage\n$d_{\\mathrm{int}}$"),
}

edges_b = [
    ("echem", "temp",  0.15),
    ("echem", "mech",  -0.15),
    ("temp",  "mech",  0.0),
    ("temp",  "creep", 0.10),
    ("mech",  "dam",   -0.12),
    ("creep", "dam",   0.12),
    ("micro", "echem", 0.15),
    ("micro", "dam",   -0.10),
    ("dam",   "micro", -0.10),
]

# Draw edges first
for src, tgt, rad in edges_b:
    x0, y0 = nodes_b[src][0], nodes_b[src][1]
    x1, y1 = nodes_b[tgt][0], nodes_b[tgt][1]
    curved_arrow(ax_b, x0, y0, x1, y1,
                 color="#8899aa", lw=1.6, rad=rad, zorder=2)

# Draw nodes (rounded rectangles with shadows)
nw, nh = 2.6, 1.7
for key, (x, y, c, label) in nodes_b.items():
    draw_node(ax_b, x, y, nw, nh, label,
              facecolor=c, fontsize=6.8, shadow=True)

panel_label(ax_b, "(b)")


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL (c) – Target-output hierarchy
# ═══════════════════════════════════════════════════════════════════════════════
ax_c = fig.add_axes([0.68, 0.52, 0.29, 0.38])
ax_c.set_xlim(-0.5, 10.5)
ax_c.set_ylim(-0.5, 10.5)
ax_c.axis("off")

ax_c.text(0.5, 1.04,
          "(c)  Target-Output Hierarchy",
          transform=ax_c.transAxes,
          fontsize=11, fontweight="bold", ha="center", va="bottom",
          color="#1a1a2e", fontfamily="serif")

hier = [
    ("Operating Conditions",
     "$T_{\\mathrm{op}}$,  $j$,  $f_u$,  $p$,  cycling protocol",
     C_HIER["op"],    9.0),
    ("Full-Field Outputs",
     "$T(\\mathbf{x},t)$,  $\\sigma(\\mathbf{x},t)$,  $j(\\mathbf{x})$",
     C_HIER["field"], 6.6),
    ("Damage Surrogates",
     "$d_b$,  $d_{\\mathrm{int}}$,  $L_{\\mathrm{cr}}$,  $A_{\\mathrm{delam}}$",
     C_HIER["dmg"],   4.2),
    ("Life-Relevant Outputs",
     "$t_f$,  $J(t)$,  remaining-life %,  failure risk",
     C_HIER["life"],  1.8),
]

bw, bh = 8.6, 1.65
bx = 0.7
for idx, (label, sub, color, cy) in enumerate(hier):
    # Shadow
    s_rect = FancyBboxPatch(
        (bx + 0.06, cy - bh / 2 - 0.05), bw, bh,
        boxstyle="round,pad=0.14",
        linewidth=0, facecolor="#00000015", zorder=2,
    )
    ax_c.add_patch(s_rect)
    # Box with gradient-like fill
    rect = FancyBboxPatch(
        (bx, cy - bh / 2), bw, bh,
        boxstyle="round,pad=0.14",
        linewidth=1.8, edgecolor=color,
        facecolor=color + "18", zorder=3,
    )
    ax_c.add_patch(rect)
    # Left accent bar
    accent = FancyBboxPatch(
        (bx, cy - bh / 2), 0.25, bh,
        boxstyle="round,pad=0.04",
        linewidth=0, facecolor=color, alpha=0.7, zorder=4,
    )
    ax_c.add_patch(accent)
    # Step number
    ax_c.text(bx + 0.13, cy, str(idx + 1),
              ha="center", va="center",
              fontsize=8, fontweight="bold", color="white", zorder=5)
    # Title and subtitle
    ax_c.text(5.3, cy + 0.28, label,
              ha="center", va="center",
              fontsize=9.5, fontweight="bold", color=color, zorder=4,
              fontfamily="serif")
    ax_c.text(5.3, cy - 0.38, sub,
              ha="center", va="center",
              fontsize=7.5, color="#444", zorder=4, style="italic",
              fontfamily="serif")

# Arrows between hierarchy levels
for i in range(len(hier) - 1):
    y_top = hier[i][3] - bh / 2 - 0.08
    y_bot = hier[i + 1][3] + bh / 2 + 0.08
    ax_c.annotate("",
                  xy=(5.3, y_bot),
                  xytext=(5.3, y_top),
                  arrowprops=dict(arrowstyle="-|>", color="#555",
                                  lw=2.0, mutation_scale=14))
    # Small arrow label
    mid_y = (y_top + y_bot) / 2
    arrow_labels = ["feeds", "maps to", "predicts"]
    ax_c.text(6.0, mid_y, arrow_labels[i],
              ha="left", va="center",
              fontsize=6.5, color="#888", style="italic",
              fontfamily="serif")

panel_label(ax_c, "(c)")


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL (d) – Constitutive dependency graph + Symbol glossary
# ═══════════════════════════════════════════════════════════════════════════════
ax_d = fig.add_axes([0.04, 0.06, 0.55, 0.40])
ax_d.set_xlim(-0.5, 14.5)
ax_d.set_ylim(-0.5, 8.5)
ax_d.axis("off")

ax_d.text(0.5, 1.04,
          "(d)  Constitutive Dependency Graph:  "
          "$T,\\,\\mu,\\,r_{\\mathrm{Ni}},\\,"
          "\\lambda_{\\mathrm{TPB}},\\,d_{\\mathrm{int}}$"
          "  $\\rightarrow$  "
          "$\\mathbf{C}(T,\\mu)\\cdot\\boldsymbol{\\sigma}$",
          transform=ax_d.transAxes,
          fontsize=10, fontweight="bold", ha="center", va="bottom",
          color="#1a1a2e", fontfamily="serif")

# Input variables (left column)
inputs_d = [
    ("$T$\nTemperature",                 2.0, 7.0, C_CONST["T"]),
    ("$\\mu$\nPorosity",                 2.0, 5.2, C_CONST["mu"]),
    ("$r_{\\mathrm{Ni}}$\nNi radius",   2.0, 3.4, C_CONST["rNi"]),
    ("$\\lambda_{\\mathrm{TPB}}$\nTPB density", 2.0, 1.6, C_CONST["tpb"]),
]

# Central process node
proc_x, proc_y = 7.0, 4.3
proc_label = ("Effective\nConstitutive\nResponse\n"
              "$\\mathbf{C}(T,\\mu)\\cdot\\boldsymbol{\\sigma}$")

# Output and damage nodes
dint_pos = (7.0, 1.0, C_CONST["dint"],
            "$d_{\\mathrm{int}}$\nInterface\ndamage")
sigma_pos = (11.5, 4.3, C_CONST["sigma"],
             "$\\boldsymbol{\\sigma}$\nStress\ntensor")

# Draw input nodes
input_nw, input_nh = 2.8, 1.3
for label, x, y, c in inputs_d:
    draw_node(ax_d, x, y, input_nw, input_nh, label,
              facecolor=c, fontsize=7.0, shadow=True)

# Draw central process node (larger, distinct)
draw_node(ax_d, proc_x, proc_y, 3.2, 2.8, proc_label,
          facecolor=C_CONST["C"], fontsize=7.5,
          edgecolor="#bf6000", linewidth=2.0, shadow=True)

# Draw damage node
dx, dy, dc, dl = dint_pos
draw_node(ax_d, dx, dy, 2.6, 1.3, dl,
          facecolor=dc, fontsize=7.0, shadow=True)

# Draw output node
sx, sy, sc, sl = sigma_pos
draw_node(ax_d, sx, sy, 2.6, 1.8, sl,
          facecolor=sc, fontsize=7.5, shadow=True)

# Arrows: inputs → central process
for label, x, y, c in inputs_d:
    curved_arrow(ax_d, x + input_nw / 2 + 0.1, y,
                 proc_x - 1.6, proc_y + (y - proc_y) * 0.2,
                 color=c, lw=1.8, rad=0.05)

# Arrow: d_int → central process
curved_arrow(ax_d, dx, dy + 0.7,
             proc_x, proc_y - 1.45,
             color=dc, lw=1.8, rad=0.0)

# Arrow: central process → sigma
curved_arrow(ax_d, proc_x + 1.6, proc_y,
             sx - 1.35, sy,
             color=C_CONST["C"], lw=2.0, rad=0.0)

# Feedback arrow: sigma → d_int
curved_arrow(ax_d, sx, sy - 0.95,
             dx + 1.35, dy,
             color="#777", lw=1.4, rad=-0.25)
ax_d.text(10.2, 2.2, "feedback",
          fontsize=6.5, color="#888", style="italic",
          rotation=-40, fontfamily="serif")

# ─── Symbol glossary (right side of panel d) ─────────────────────────────────
gx = 13.0
gy_start = 7.8
glossary = [
    ("$T$",                         "Temperature / K",            C_CONST["T"]),
    ("$\\mu$",                      "Porosity / –",               C_CONST["mu"]),
    ("$r_{\\mathrm{Ni}}$",         "Ni particle radius / μm",    C_CONST["rNi"]),
    ("$\\lambda_{\\mathrm{TPB}}$", "TPB density / m$\\cdot$m$^{-3}$", C_CONST["tpb"]),
    ("$d_{\\mathrm{int}}$",        "Interface damage / –",       C_CONST["dint"]),
    ("$\\boldsymbol{\\sigma}$",    "Cauchy stress / MPa",        C_CONST["sigma"]),
    ("$\\mathbf{C}$",              "Stiffness tensor / GPa",     C_CONST["C"]),
]

# Glossary title
ax_d.text(gx, gy_start + 0.3, "Symbol Glossary",
          ha="center", va="bottom",
          fontsize=8.5, fontweight="bold", color="#1a1a2e",
          fontfamily="serif")
ax_d.plot([gx - 1.2, gx + 1.2], [gy_start + 0.2, gy_start + 0.2],
          color="#ccc", linewidth=0.8)

for j, (sym, desc, c) in enumerate(glossary):
    yy = gy_start - j * 0.95 - 0.3
    # Colour dot
    ax_d.plot(gx - 1.1, yy, "o", color=c, markersize=5, zorder=5)
    # Symbol
    ax_d.text(gx - 0.8, yy, sym,
              ha="left", va="center",
              fontsize=7.5, color=c, fontweight="bold", zorder=5)
    # Description
    ax_d.text(gx - 0.0, yy, desc,
              ha="left", va="center",
              fontsize=6.5, color="#555", zorder=5,
              fontfamily="serif")

panel_label(ax_d, "(d)")


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL (e) – BONUS: Stress vs. Temperature comparison plot
#   (Publication-quality quantitative subplot demonstrating axis standards)
# ═══════════════════════════════════════════════════════════════════════════════
ax_e = fig.add_axes([0.65, 0.06, 0.30, 0.38])

# Simulated data: Von Mises stress in electrolyte vs. temperature
T = np.linspace(25, 900, 200)
# Linear elastic model (conservative, higher stress)
sigma_elastic = 150 - 0.02 * (T - 25) + 12 * np.sin(0.005 * T)
# Viscoelastic model (stress relaxation at high T)
sigma_visco = 150 - 0.02 * (T - 25) - 0.00008 * (T - 25)**2 + \
              8 * np.sin(0.005 * T)
# Fracture threshold
sigma_frac = np.full_like(T, 165.0)

# Plot with coloured lines, 1pt width
line1, = ax_e.plot(T, sigma_elastic, color="#c62828", linewidth=1.0,
                   linestyle="-", label="Linear elastic model")
line2, = ax_e.plot(T, sigma_visco, color="#1565c0", linewidth=1.0,
                   linestyle="--", label="Viscoelastic model")
line3, = ax_e.plot(T, sigma_frac, color="#333333", linewidth=1.0,
                   linestyle=":", label="Fracture threshold (8YSZ)")

# Markers at selected points for distinction
marker_idx = np.arange(0, len(T), 25)
ax_e.plot(T[marker_idx], sigma_elastic[marker_idx], "o",
          color="#c62828", markersize=3.5, zorder=5)
ax_e.plot(T[marker_idx], sigma_visco[marker_idx], "s",
          color="#1565c0", markersize=3.5, zorder=5)

# Fill between to highlight stress relaxation region
ax_e.fill_between(T, sigma_visco, sigma_elastic,
                  alpha=0.08, color="#6a1b9a",
                  label="Stress relaxation (up to 20%)")

# Axes formatting per publication standards
ax_e.set_xlabel("Temperature, $T$ / °C", fontsize=10, fontfamily="serif",
                labelpad=6)
ax_e.set_ylabel("Von Mises stress, $\\sigma_{\\mathrm{VM}}$ / MPa",
                fontsize=10, fontfamily="serif", labelpad=6,
                rotation=90)

# Proper tick formatting
ax_e.set_xlim(0, 950)
ax_e.set_ylim(80, 180)
ax_e.set_xticks(np.arange(0, 1001, 200))
ax_e.set_yticks(np.arange(80, 181, 20))
ax_e.tick_params(axis="both", which="both", direction="in",
                 top=True, right=True, width=1.0)
ax_e.minorticks_on()
ax_e.tick_params(axis="both", which="minor", direction="in",
                 top=True, right=True, width=0.6)

# Black 1pt axes
for spine in ax_e.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

# Legend without border
ax_e.legend(loc="upper right", fontsize=7, frameon=False,
            labelspacing=0.4)

# Panel title
ax_e.text(0.5, 1.06,
          "(e)  Electrolyte Stress: Elastic vs. Viscoelastic Model",
          transform=ax_e.transAxes,
          fontsize=10, fontweight="bold", ha="center", va="bottom",
          color="#1a1a2e", fontfamily="serif")

# Annotation for key insight
ax_e.annotate(
    "~20% stress\nrelaxation\nat 800 °C",
    xy=(800, sigma_visco[np.argmin(np.abs(T - 800))]),
    xytext=(600, 100),
    fontsize=7, color="#6a1b9a", fontweight="bold",
    fontfamily="serif",
    arrowprops=dict(arrowstyle="-|>", color="#6a1b9a", lw=1.0),
    bbox=dict(facecolor="white", edgecolor="#6a1b9a",
              boxstyle="round,pad=0.3", alpha=0.9),
)

panel_label(ax_e, "(e)")


# ═══════════════════════════════════════════════════════════════════════════════
# Save and display
# ═══════════════════════════════════════════════════════════════════════════════
output_file = "figure_2_1_sofc.png"
fig.savefig(output_file, dpi=300, facecolor="white",
            bbox_inches="tight", pad_inches=0.15)
print(f"✓  Saved: {output_file}  "
      f"({fig.get_size_inches()[0]:.0f}×{fig.get_size_inches()[1]:.0f} in "
      f"@ 300 dpi)")

# Display interactively (works in GUI / Jupyter / IDE)
plt.show()

print("✓  Done.")
