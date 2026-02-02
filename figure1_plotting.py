"""Figure 1 plotting helpers."""

def add_panel_label(ax, label, cfg, x=-0.08, y=1.06):
    """Add a panel label that works for 2D and 3D axes."""
    text_kwargs = dict(
        fontsize=cfg.TITLE_SIZE + 2,
        fontweight="bold",
        color=cfg.COLORS["text_dark"],
        verticalalignment="top",
        horizontalalignment="left",
    )
    label_text = f"({label})"

    # Axes3D.text expects x, y, z, s. Use text2D to keep axes coords.
    if hasattr(ax, "text2D"):
        ax.text2D(x, y, label_text, transform=ax.transAxes, **text_kwargs)
    else:
        ax.text(x, y, label_text, transform=ax.transAxes, **text_kwargs)
