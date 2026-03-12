# =============================================================================
# figures/sem_diagrams.R
#
# Publication-quality Structural Equation Modeling (SEM) diagrams.
# These are illustrative methodology figures included in the research article
# "A Comparative Analysis of Constitutive Models for Predicting the
# Electrolyte's Fracture Risk in Planar SOFCs".
#
# The diagrams depict generic SEM measurement model types:
#   Panel A — Reflective latent variable model  (η → x1, x2, x3)
#   Panel B — Formative composite model         (x1, x2, x3 → ξ;  FIN-SAV)
#
# Output files (written to the same directory as this script):
#   panel_A_reflective.pdf / panel_A_reflective.png
#   panel_B_formative.pdf  / panel_B_formative.png
#   sem_combined.pdf       / sem_combined.png         (double-column)
#
# How to run
#   Rscript figures/sem_diagrams.R
#   — or —
#   source("figures/sem_diagrams.R")   # from an interactive R session
#
# Requirements: ggplot2, ggforce (optional), patchwork, grid, showtext, sysfonts
# =============================================================================

# ---------------------------------------------------------------------------
# 0.  Package installation and loading
# ---------------------------------------------------------------------------
required_pkgs <- c("ggplot2", "patchwork", "grid", "showtext", "sysfonts")

for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing missing package: ", pkg)
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

library(ggplot2)
library(patchwork)
library(grid)
library(showtext)
library(sysfonts)

# ---------------------------------------------------------------------------
# 1.  Font setup — Times New Roman (via Tinos, metrically identical)
# ---------------------------------------------------------------------------
tryCatch({
  sysfonts::font_add_google("Tinos", "TimesNR")
  message("Loaded 'Tinos' (Times New Roman equivalent) via Google Fonts.")
}, error = function(e) {
  message("Google Fonts unavailable; falling back to built-in 'serif'.")
  sysfonts::font_add("TimesNR", regular = "serif")
})
showtext_auto(enable = TRUE)
showtext_opts(dpi = 300)

FONT <- "TimesNR"

# ---------------------------------------------------------------------------
# 2.  Design constants
# ---------------------------------------------------------------------------
COL_LATENT  <- "#2F6CB5"   # steel-blue  (latent variable fill)
COL_OBS     <- "#F5F5F5"   # warm light grey (observed indicator fill)
COL_ERR     <- "#FFFBE6"   # cream (error / disturbance circle fill)
COL_BORDER  <- "#1A1A1A"   # near-black (node borders, text)
COL_ARROW   <- "#1A1A1A"   # arrow stroke
COL_LABEL   <- "#B22222"   # firebrick  (edge λ / γ italic labels)
COL_WHITE   <- "white"

LW_BORDER   <- 1.5         # node border line-width (pts equivalent in ggplot)
LW_ARROW    <- 0.8         # arrow segment line-width

# Journal column widths (inches)
FIG_W_SINGLE <- 3.35       # single-column ≈ 8.5 cm
FIG_W_DOUBLE <- 6.70       # double-column ≈ 17 cm
FIG_H        <- 3.60       # panel height

# ---------------------------------------------------------------------------
# 3.  Low-level drawing helpers
#     Each returns a list of ggplot2 layer objects that can be added with '+'
# ---------------------------------------------------------------------------

# Helper: unit circle / ellipse polygon points
.ellipse_pts <- function(cx, cy, a, b, n = 300) {
  theta <- seq(0, 2 * pi, length.out = n)
  list(x = cx + a * cos(theta), y = cy + b * sin(theta))
}

#' Filled ellipse node (draw fill polygon THEN border path so border is visible)
ellipse_node <- function(cx, cy, a = 0.60, b = 0.28,
                         fill = COL_LATENT, border = COL_BORDER,
                         lw = LW_BORDER) {
  pts <- .ellipse_pts(cx, cy, a, b)
  list(
    # 1. filled polygon (no outline so it does not mask the explicit border)
    annotate("polygon", x = pts$x, y = pts$y,
             fill = fill, colour = NA),
    # 2. border path drawn on top of fill
    annotate("path", x = pts$x, y = pts$y,
             colour = border, linewidth = lw)
  )
}

#' Filled rectangle (observed indicator) node
rect_node <- function(cx, cy, w = 0.80, h = 0.42,
                      fill = COL_OBS, border = COL_BORDER,
                      lw = LW_BORDER) {
  list(
    annotate("rect",
             xmin = cx - w / 2, xmax = cx + w / 2,
             ymin = cy - h / 2, ymax = cy + h / 2,
             fill = fill, colour = border, linewidth = lw)
  )
}

#' Filled circle (error / disturbance term)
circle_node <- function(cx, cy, r = 0.15,
                        fill = COL_ERR, border = COL_BORDER,
                        lw = LW_BORDER) {
  pts <- .ellipse_pts(cx, cy, r, r)
  list(
    annotate("polygon", x = pts$x, y = pts$y,
             fill = fill, colour = NA),
    annotate("path", x = pts$x, y = pts$y,
             colour = border, linewidth = lw)
  )
}

#' Arrow segment — pads both ends so the line starts/ends at node boundaries
seg_arrow <- function(x0, y0, x1, y1,
                      pad0 = 0.0, pad1 = 0.0,
                      col = COL_ARROW, lw = LW_ARROW) {
  dx  <- x1 - x0
  dy  <- y1 - y0
  len <- sqrt(dx^2 + dy^2)
  if (len < 1e-9) return(list())
  ux  <- dx / len;  uy <- dy / len
  list(
    annotate("segment",
             x    = x0 + ux * pad0,  y    = y0 + uy * pad0,
             xend = x1 - ux * pad1,  yend = y1 - uy * pad1,
             colour = col, linewidth = lw, lineend = "butt",
             arrow = arrow(length = unit(9, "pt"),
                           type = "closed", ends = "last"))
  )
}

#' Plain line — no arrowhead (used for x → ε connections)
seg_plain <- function(x0, y0, x1, y1,
                      pad0 = 0.0, pad1 = 0.0,
                      col = COL_ARROW, lw = LW_ARROW) {
  dx  <- x1 - x0
  dy  <- y1 - y0
  len <- sqrt(dx^2 + dy^2)
  if (len < 1e-9) return(list())
  ux  <- dx / len;  uy <- dy / len
  list(
    annotate("segment",
             x    = x0 + ux * pad0,  y    = y0 + uy * pad0,
             xend = x1 - ux * pad1,  yend = y1 - uy * pad1,
             colour = col, linewidth = lw, lineend = "butt")
  )
}

#' Text label helper
lbl <- function(x, y, txt, size = 5.5,
                fontface = "plain", colour = COL_BORDER,
                family = FONT, hjust = 0.5, vjust = 0.5) {
  list(
    annotate("text", x = x, y = y, label = txt,
             size = size, fontface = fontface,
             colour = colour, family = family,
             hjust = hjust, vjust = vjust)
  )
}

#' Point on the boundary of an ellipse (cx, cy, a, b) in the direction of
#' the vector from (cx, cy) towards (tx, ty).  Used to clip arrow endpoints.
ellipse_edge <- function(cx, cy, a, b, tx, ty) {
  dx <- tx - cx;  dy <- ty - cy
  # parametric t: (dx*t/a)^2 + (dy*t/b)^2 = 1  => t = 1 / sqrt((dx/a)^2 + (dy/b)^2)
  denom <- sqrt((dx / a)^2 + (dy / b)^2)
  if (denom < 1e-12) return(c(cx, cy))
  t <- 1 / denom
  c(cx + dx * t, cy + dy * t)
}

# Base ggplot theme — pure white canvas, no axes/grid
theme_sem <- function() {
  theme_void() +
    theme(
      plot.background  = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.title       = element_text(family = FONT, size = 14,
                                      face = "bold", colour = COL_BORDER,
                                      hjust = 0.5,
                                      margin = margin(b = 8)),
      plot.caption     = element_text(family = FONT, size = 7.5,
                                      colour = COL_BORDER, hjust = 0.5,
                                      margin = margin(t = 6)),
      plot.margin      = margin(10, 10, 8, 10)
    )
}

# ---------------------------------------------------------------------------
# 4.  Panel A — Reflective model  (η → x1, x2, x3)
# ---------------------------------------------------------------------------
#
#  Canvas:  x ∈ [0, 4.2],  y ∈ [0.0, 3.2]
#
#  [η ellipse @ (0.9, 1.6)]  ---λ1--->  [x1 box @ (2.6, 2.4)]  ---> (ε1)
#                             ---λ2--->  [x2 box @ (2.6, 1.6)]  ---> (ε2)
#                             ---λ3--->  [x3 box @ (2.6, 0.8)]  ---> (ε3)

make_panel_A <- function() {

  # Node centres
  eta_cx <- 0.90;  eta_cy <- 1.60      # latent construct
  eta_a  <- 0.60;  eta_b  <- 0.28      # ellipse semi-axes

  obs_x  <- 2.60                        # x of all observed boxes
  obs_y  <- c(2.40, 1.60, 0.80)        # y of x1, x2, x3
  box_hw <- 0.40;  box_hh <- 0.21      # half-width, half-height of boxes

  err_x  <- 3.58                        # x of error circles
  err_r  <- 0.15                        # error circle radius

  # Arrow endpoints (clipped to node boundaries)
  # η → x2  is horizontal (same y)
  # η → x1  and η → x3  are diagonal

  .eta_to_box <- function(bx, by) {
    # start: point on ellipse edge towards box
    ep  <- ellipse_edge(eta_cx, eta_cy, eta_a, eta_b, bx, by)
    # end:   left edge of box
    ex  <- bx - box_hw
    ey  <- by
    list(x0 = ep[1], y0 = ep[2], x1 = ex, y1 = ey)
  }

  aa <- .eta_to_box(obs_x, obs_y[1])
  ab <- .eta_to_box(obs_x, obs_y[2])
  ac <- .eta_to_box(obs_x, obs_y[3])

  # midpoint of each arrow (for λ label placement)
  lam_x1 <- (aa$x0 + aa$x1) / 2;  lam_y1 <- (aa$y0 + aa$y1) / 2
  lam_x2 <- (ab$x0 + ab$x1) / 2;  lam_y2 <- (ab$y0 + ab$y1) / 2
  lam_x3 <- (ac$x0 + ac$x1) / 2;  lam_y3 <- (ac$y0 + ac$y1) / 2

  ggplot() +
    coord_fixed(xlim = c(0.0, 4.2), ylim = c(0.2, 3.2), expand = FALSE) +

    # ---- Latent ellipse η ----
    ellipse_node(eta_cx, eta_cy, a = eta_a, b = eta_b) +
    lbl(eta_cx, eta_cy + 0.06, "\u03b7",       # η
        size = 8.0, fontface = "bolditalic", colour = COL_WHITE) +
    lbl(eta_cx, eta_cy - 0.11, "Latent",
        size = 4.0, fontface = "bold",         colour = COL_WHITE) +

    # ---- Observed indicator boxes ----
    rect_node(obs_x, obs_y[1]) +
    rect_node(obs_x, obs_y[2]) +
    rect_node(obs_x, obs_y[3]) +
    lbl(obs_x, obs_y[1], "x\u2081", size = 6.5, fontface = "bold") +   # x₁
    lbl(obs_x, obs_y[2], "x\u2082", size = 6.5, fontface = "bold") +   # x₂
    lbl(obs_x, obs_y[3], "x\u2083", size = 6.5, fontface = "bold") +   # x₃

    # ---- Error circles ε1, ε2, ε3 ----
    circle_node(err_x, obs_y[1], r = err_r) +
    circle_node(err_x, obs_y[2], r = err_r) +
    circle_node(err_x, obs_y[3], r = err_r) +
    lbl(err_x, obs_y[1], "\u03b51", size = 4.5, fontface = "italic") +  # ε1
    lbl(err_x, obs_y[2], "\u03b52", size = 4.5, fontface = "italic") +  # ε2
    lbl(err_x, obs_y[3], "\u03b53", size = 4.5, fontface = "italic") +  # ε3

    # ---- Arrows η → xi ----
    seg_arrow(aa$x0, aa$y0, aa$x1, aa$y1) +
    seg_arrow(ab$x0, ab$y0, ab$x1, ab$y1) +
    seg_arrow(ac$x0, ac$y0, ac$x1, ac$y1) +

    # ---- λ edge labels ----
    lbl(lam_x1, lam_y1 + 0.13, "\u03bb\u2081",       # λ₁
        size = 5.0, fontface = "italic", colour = COL_LABEL) +
    lbl(lam_x2, lam_y2 + 0.12, "\u03bb\u2082",       # λ₂
        size = 5.0, fontface = "italic", colour = COL_LABEL) +
    lbl(lam_x3, lam_y3 - 0.13, "\u03bb\u2083",       # λ₃
        size = 5.0, fontface = "italic", colour = COL_LABEL) +

    # ---- Plain lines xi → εi ----
    seg_plain(obs_x + box_hw, obs_y[1], err_x - err_r, obs_y[1]) +
    seg_plain(obs_x + box_hw, obs_y[2], err_x - err_r, obs_y[2]) +
    seg_plain(obs_x + box_hw, obs_y[3], err_x - err_r, obs_y[3]) +

    # ---- Title & caption ----
    labs(
      title   = "Panel A: Reflective Latent Variable (\u03b7 \u2192 x)",
      caption = paste0(
        "Panel A: Reflective measurement model. The latent construct \u03b7 causes\n",
        "observed indicators x\u2081\u2013x\u2083 via factor loadings \u03bb\u2081\u2013\u03bb\u2083.\n",
        "\u03b5\u1d62 = measurement error term for each indicator x\u1d62 (i = 1, 2, 3)."
      )
    ) +
    theme_sem()
}

# ---------------------------------------------------------------------------
# 5.  Panel B — Formative composite  (x1, x2, x3 → ξ)
# ---------------------------------------------------------------------------
#
#  Canvas:  x ∈ [0, 4.2],  y ∈ [0.0, 3.2]
#
#  [x1 box @ (1.0, 2.4)] ---γ1--->  [ξ ellipse @ (3.1, 1.6)]
#  [x2 box @ (1.0, 1.6)] ---γ2--->
#  [x3 box @ (1.0, 0.8)] ---γ3--->
#                          <---ζ--- (ζ circle @ (3.1, 2.90))

make_panel_B <- function() {

  # Node centres
  xi_cx  <- 3.10;  xi_cy  <- 1.60      # composite latent ξ
  xi_a   <- 0.60;  xi_b   <- 0.28

  obs_x  <- 1.00                        # x of indicator boxes
  obs_y  <- c(2.40, 1.60, 0.80)
  box_hw <- 0.40;  box_hh <- 0.21

  zeta_cx <- 3.10;  zeta_cy <- 2.88     # disturbance circle ζ
  zeta_r  <- 0.15

  # Arrow endpoints: box right edge → ξ ellipse edge
  .box_to_xi <- function(bx, by) {
    ex  <- bx + box_hw
    ey  <- by
    ep  <- ellipse_edge(xi_cx, xi_cy, xi_a, xi_b, bx, by)
    list(x0 = ex, y0 = ey, x1 = ep[1], y1 = ep[2])
  }

  ba <- .box_to_xi(obs_x, obs_y[1])
  bb <- .box_to_xi(obs_x, obs_y[2])
  bc <- .box_to_xi(obs_x, obs_y[3])

  # midpoints for γ labels
  gam_x1 <- (ba$x0 + ba$x1) / 2;  gam_y1 <- (ba$y0 + ba$y1) / 2
  gam_x2 <- (bb$x0 + bb$x1) / 2;  gam_y2 <- (bb$y0 + bb$y1) / 2
  gam_x3 <- (bc$x0 + bc$x1) / 2;  gam_y3 <- (bc$y0 + bc$y1) / 2

  # ζ → ξ arrow endpoints
  zeta_ep <- ellipse_edge(xi_cx, xi_cy, xi_a, xi_b, zeta_cx, zeta_cy)

  ggplot() +
    coord_fixed(xlim = c(0.0, 4.2), ylim = c(0.2, 3.2), expand = FALSE) +

    # ---- Observed indicator boxes ----
    rect_node(obs_x, obs_y[1]) +
    rect_node(obs_x, obs_y[2]) +
    rect_node(obs_x, obs_y[3]) +
    # Primary symbol (bold, larger)
    lbl(obs_x, obs_y[1] + 0.06, "x\u2081",
        size = 6.0, fontface = "bold") +
    lbl(obs_x, obs_y[1] - 0.12, "(Literacy)",
        size = 3.4, fontface = "plain") +
    lbl(obs_x, obs_y[2] + 0.06, "x\u2082",
        size = 6.0, fontface = "bold") +
    lbl(obs_x, obs_y[2] - 0.12, "(Tenure)",
        size = 3.4, fontface = "plain") +
    lbl(obs_x, obs_y[3] + 0.06, "x\u2083",
        size = 6.0, fontface = "bold") +
    lbl(obs_x, obs_y[3] - 0.12, "(Digital Use)",
        size = 3.4, fontface = "plain") +

    # ---- Composite ellipse ξ (FIN-SAV) ----
    ellipse_node(xi_cx, xi_cy, a = xi_a, b = xi_b) +
    lbl(xi_cx, xi_cy + 0.06, "\u03be",          # ξ
        size = 8.0, fontface = "bolditalic", colour = COL_WHITE) +
    lbl(xi_cx, xi_cy - 0.11, "FIN-SAV",
        size = 3.8, fontface = "bold",           colour = COL_WHITE) +

    # ---- Disturbance circle ζ ----
    circle_node(zeta_cx, zeta_cy, r = zeta_r) +
    lbl(zeta_cx, zeta_cy, "\u03b6",              # ζ
        size = 4.5, fontface = "italic") +

    # ---- Arrows xi → ξ ----
    seg_arrow(ba$x0, ba$y0, ba$x1, ba$y1) +
    seg_arrow(bb$x0, bb$y0, bb$x1, bb$y1) +
    seg_arrow(bc$x0, bc$y0, bc$x1, bc$y1) +

    # ---- γ edge labels ----
    lbl(gam_x1, gam_y1 + 0.13, "\u03b3\u2081",  # γ₁
        size = 5.0, fontface = "italic", colour = COL_LABEL) +
    lbl(gam_x2, gam_y2 + 0.12, "\u03b3\u2082",  # γ₂
        size = 5.0, fontface = "italic", colour = COL_LABEL) +
    lbl(gam_x3, gam_y3 - 0.13, "\u03b3\u2083",  # γ₃
        size = 5.0, fontface = "italic", colour = COL_LABEL) +

    # ---- Arrow ζ → ξ (disturbance enters composite) ----
    seg_arrow(zeta_cx, zeta_cy - zeta_r,
              zeta_ep[1], zeta_ep[2]) +
    lbl(zeta_cx + 0.18,
        (zeta_cy - zeta_r + zeta_ep[2]) / 2 + 0.03,
        "\u03b6", size = 5.0, fontface = "italic", colour = COL_LABEL) +

    # ---- Title & caption ----
    labs(
      title   = "Panel B: Formative Composite (x \u2192 \u03be; FIN-SAV)",
      caption = paste0(
        "Panel B: Formative composite model. Observed indicators x\u2081 (Literacy),\n",
        "x\u2082 (Tenure), x\u2083 (Digital Use) form composite \u03be (FIN-SAV)\n",
        "via weights \u03b3\u2081\u2013\u03b3\u2083; \u03b6 = disturbance term."
      )
    ) +
    theme_sem()
}

# ---------------------------------------------------------------------------
# 6.  Build plots
# ---------------------------------------------------------------------------
message("Building Panel A (Reflective model) ...")
pA <- make_panel_A()

message("Building Panel B (Formative composite) ...")
pB <- make_panel_B()

# Combined two-panel figure (side-by-side)
combined <- (pA | pB) +
  plot_annotation(
    title   = paste0(
      "Structural Equation Modeling Diagrams\u2014",
      "Reflective vs. Formative Measurement Models"
    ),
    caption = paste0(
      "Note: \u03b7 = reflective latent construct; \u03be = formative composite (FIN-SAV); ",
      "\u03bb = factor loadings; \u03b3 = formative weights;\n",
      "\u03b51\u2013\u03b53 = measurement error terms; \u03b6 = disturbance term. ",
      "Font: Times New Roman (Tinos). Export resolution: 300 DPI."
    ),
    theme = theme(
      plot.title      = element_text(family = FONT, size = 15, face = "bold",
                                     hjust = 0.5, colour = COL_BORDER,
                                     margin = margin(b = 4)),
      plot.caption    = element_text(family = FONT, size = 7.5,
                                     colour = COL_BORDER, hjust = 0.5,
                                     margin = margin(t = 6)),
      plot.background = element_rect(fill = "white", colour = NA)
    )
  )

# ---------------------------------------------------------------------------
# 7.  Output directory detection (works with both Rscript and source())
# ---------------------------------------------------------------------------
# Strategy:
#   1. If launched via Rscript, commandArgs() contains "--file=<path>".
#   2. If source()'d interactively, search all active frames for an $ofile
#      attribute (set by source() since R 3.x).
#   3. Fall back to "figures" relative to the working directory.

out_dir <- tryCatch({
  # --- Rscript path ---
  args     <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    normalizePath(dirname(sub("^--file=", "", file_arg[1])))
  } else {
    # --- source() path: search all frames for $ofile ---
    ofile <- NULL
    for (i in seq_along(sys.frames())) {
      env <- sys.frames()[[i]]
      if (exists("ofile", envir = env, inherits = FALSE)) {
        ofile <- get("ofile", envir = env, inherits = FALSE)
        if (!is.null(ofile) && nchar(ofile) > 0) break
      }
    }
    if (!is.null(ofile)) {
      normalizePath(dirname(ofile))
    } else {
      "figures"   # safe fallback: write next to a 'figures/' sub-dir in cwd
    }
  }
}, error = function(e) "figures")

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
message("Output directory: ", out_dir)

# ---------------------------------------------------------------------------
# 8.  Export helper — saves PDF (vector) + PNG (300 DPI)
# ---------------------------------------------------------------------------
save_fig <- function(plot_obj, stem, w = FIG_W_SINGLE, h = FIG_H, dpi = 300) {
  pdf_path <- file.path(out_dir, paste0(stem, ".pdf"))
  png_path <- file.path(out_dir, paste0(stem, ".png"))

  # PDF — vector format, preferred for journal submission
  tryCatch(
    ggsave(pdf_path, plot = plot_obj, width = w, height = h, units = "in",
           device = cairo_pdf),
    error = function(e) {
      message("  cairo_pdf unavailable, falling back to pdf device.")
      ggsave(pdf_path, plot = plot_obj, width = w, height = h, units = "in",
             device = "pdf")
    }
  )
  message("  Saved: ", pdf_path)

  # PNG — raster, 300 DPI, for online/review submission
  ggsave(png_path, plot = plot_obj, width = w, height = h, units = "in",
         dpi = dpi, device = "png")
  message("  Saved: ", png_path)
}

# Individual panels — single-column width (3.35 in)
save_fig(pA,       "panel_A_reflective", w = FIG_W_SINGLE, h = FIG_H)
save_fig(pB,       "panel_B_formative",  w = FIG_W_SINGLE, h = FIG_H)

# Combined figure — double-column width (6.70 in)
save_fig(combined, "sem_combined",       w = FIG_W_DOUBLE, h = FIG_H + 0.4)

message("\nAll figures saved successfully.")
saved_files <- list.files(out_dir, pattern = "\\.(pdf|png)$", full.names = TRUE)
invisible(lapply(saved_files, message))
