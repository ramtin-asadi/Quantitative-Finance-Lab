from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from .curves import LAB_COLORS, set_plot_style


def plot_computation_dag(
    ax=None,
    title: str = "BSM Call Price - Computational Graph (DAG)",
    figsize: tuple[float, float] = (22, 11),
    save_path: str | None = None,
    dpi: int = 200,
    node_radius: float = 0.38,
    font_size: float = 12,
    edge_lw: float = 1.4,
    colormap: str = "Set3"):
    """
    Draw the original Notebook 4 BSM computation DAG.

    This restores the old Quantitative-Finance-Lab visual language: rounded
    colored nodes, labeled arrows, and explicit forward/reverse pass guides.
    """
    del colormap
    set_plot_style()

    bg_colour = "#fafafa"
    text_colour = "black"
    edge_colour = "black"
    arrow_colour = LAB_COLORS[2]
    input_fc = LAB_COLORS[0]
    output_fc = LAB_COLORS[1]
    op_fc = LAB_COLORS[8]
    intermed_fc = LAB_COLORS[10]
    legend_edge = "#bbbbbb"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor=bg_colour)
    else:
        fig = ax.get_figure()
    ax.set_facecolor(bg_colour)
    ax.set_aspect("equal")
    ax.axis("off")

    r = float(node_radius)

    nodes = [
        ("S", 0.0, 9.0, r"$S$", "input"),
        ("K", 0.0, 7.2, r"$K$", "input"),
        ("tau", 0.0, 5.4, r"$\tau$", "input"),
        ("r", 0.0, 3.6, r"$r$", "input"),
        ("q", 0.0, 1.8, r"$q$", "input"),
        ("sigma", 0.0, 0.0, r"$\sigma$", "input"),
        ("ln", 3.0, 8.5, r"$\ln$", "op"),
        ("sig2", 3.0, 0.5, r"$(\cdot)^2$", "op"),
        ("sqrt", 3.0, 4.2, r"$\sqrt{\,}$", "op"),
        ("ln_sk", 5.5, 8.5, r"$\ln\!\frac{S}{K}$", "intermed"),
        ("halfsig", 5.5, 0.5, r"$\frac{\sigma^2}{2}$", "intermed"),
        ("sqrttau", 5.5, 4.2, r"$\sqrt{\tau}$", "intermed"),
        ("drift", 8.0, 3.0, r"$r - q + \frac{\sigma^2}{2}$", "op"),
        ("num", 10.5, 6.5, r"$\ln\!\frac{S}{K} + \mu\tau$", "intermed"),
        ("den", 10.5, 2.0, r"$\sigma\sqrt{\tau}$", "intermed"),
        ("d1", 13.5, 6.5, r"$d_1$", "intermed"),
        ("d2", 13.5, 2.0, r"$d_2$", "intermed"),
        ("Nd1", 16.5, 7.8, r"$\Phi(d_1)$", "op"),
        ("Nd2", 16.5, 3.2, r"$\Phi(d_2)$", "op"),
        ("discq", 16.5, 9.5, r"$e^{-q\tau}$", "op"),
        ("discr", 16.5, 0.8, r"$e^{-r\tau}$", "op"),
        ("term1", 19.5, 8.0, r"$S e^{-q\tau}\Phi(d_1)$", "intermed"),
        ("term2", 19.5, 2.0, r"$K e^{-r\tau}\Phi(d_2)$", "intermed"),
        ("C", 22.5, 5.0, r"$C$", "output"),
    ]

    edges = [
        ("S", "ln", ""),
        ("K", "ln", ""),
        ("sigma", "sig2", ""),
        ("tau", "sqrt", ""),
        ("ln", "ln_sk", ""),
        ("sig2", "halfsig", ""),
        ("sqrt", "sqrttau", ""),
        ("r", "drift", ""),
        ("q", "drift", ""),
        ("halfsig", "drift", ""),
        ("ln_sk", "num", ""),
        ("drift", "num", r"$\times\tau$"),
        ("tau", "num", ""),
        ("sigma", "den", ""),
        ("sqrttau", "den", ""),
        ("num", "d1", r"$\div$"),
        ("den", "d1", ""),
        ("d1", "d2", r"$-\sigma\sqrt{\tau}$"),
        ("den", "d2", ""),
        ("d1", "Nd1", ""),
        ("d2", "Nd2", ""),
        ("q", "discq", ""),
        ("tau", "discq", ""),
        ("r", "discr", ""),
        ("tau", "discr", ""),
        ("S", "term1", ""),
        ("discq", "term1", ""),
        ("Nd1", "term1", ""),
        ("K", "term2", ""),
        ("discr", "term2", ""),
        ("Nd2", "term2", ""),
        ("term1", "C", r"$+$"),
        ("term2", "C", r"$-$"),
    ]

    pos = {n[0]: (n[1], n[2]) for n in nodes}
    labels = {n[0]: n[3] for n in nodes}
    cat_colours = {
        "input": input_fc,
        "op": op_fc,
        "intermed": intermed_fc,
        "output": output_fc,
    }

    def _node_radius(nid):
        txt = labels[nid]
        if len(txt) > 18:
            return r * 2.2, r * 1.0
        if len(txt) > 10:
            return r * 1.7, r * 0.9
        return r, r

    for src, dst, lbl in edges:
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        dx, dy = x1 - x0, y1 - y0
        dist = np.hypot(dx, dy)
        if dist == 0:
            continue
        ux, uy = dx / dist, dy / dist
        rx0, ry0 = _node_radius(src)
        rx1, ry1 = _node_radius(dst)
        xs, ys = x0 + ux * (max(rx0, ry0) + 0.05), y0 + uy * (max(rx0, ry0) + 0.05)
        xe, ye = x1 - ux * (max(rx1, ry1) + 0.05), y1 - uy * (max(rx1, ry1) + 0.05)
        arrow = FancyArrowPatch(
            (xs, ys),
            (xe, ye),
            arrowstyle="-|>",
            mutation_scale=14,
            lw=edge_lw,
            color=arrow_colour,
            connectionstyle="arc3,rad=0.06",
            zorder=1,
        )
        ax.add_patch(arrow)
        if lbl:
            ax.text(
                0.5 * (xs + xe),
                0.5 * (ys + ye) + 0.25,
                lbl,
                ha="center",
                va="center",
                fontsize=font_size - 2,
                color=text_colour,
                bbox={"boxstyle": "round,pad=0.18", "fc": bg_colour, "ec": legend_edge},
                zorder=5,
            )

    for nid, x, y, label, cat in nodes:
        rx, ry = _node_radius(nid)
        shadow = mpatches.FancyBboxPatch(
            (x - rx + 0.06, y - ry - 0.06),
            2 * rx,
            2 * ry,
            boxstyle=f"round,pad=0.0,rounding_size={min(rx, ry) * 0.9}",
            facecolor="#00000015",
            edgecolor="none",
            linewidth=0,
            zorder=2,
        )
        ax.add_patch(shadow)
        node = mpatches.FancyBboxPatch(
            (x - rx, y - ry),
            2 * rx,
            2 * ry,
            boxstyle=f"round,pad=0.0,rounding_size={min(rx, ry) * 0.9}",
            facecolor=cat_colours[cat],
            edgecolor=edge_colour,
            linewidth=1.6,
            zorder=3,
        )
        ax.add_patch(node)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=font_size if len(label) < 14 else font_size - 2,
            fontweight="bold",
            color=text_colour,
            zorder=4,
        )

    y_ann = -1.8
    ax.annotate(
        "",
        xy=(8, y_ann),
        xytext=(1, y_ann),
        arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": LAB_COLORS[0]},
    )
    ax.text(
        4.5,
        y_ann - 0.55,
        "Forward pass  (evaluate price)",
        ha="center",
        fontsize=font_size,
        fontstyle="italic",
        color=LAB_COLORS[0],
    )
    ax.annotate(
        "",
        xy=(15, y_ann),
        xytext=(22, y_ann),
        arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": LAB_COLORS[1]},
    )
    ax.text(
        18.5,
        y_ann - 0.55,
        "Reverse pass  (chain-rule -> Greeks)",
        ha="center",
        fontsize=font_size,
        fontstyle="italic",
        color=LAB_COLORS[1],
    )

    lx, ly = 0.5, 10.8
    for i, (cat, name) in enumerate(
        [("input", "Input"), ("op", "Operation"), ("intermed", "Intermediate"), ("output", "Output")]
    ):
        rect = mpatches.FancyBboxPatch(
            (lx + i * 4.2, ly),
            0.55,
            0.55,
            boxstyle="round,pad=0.0,rounding_size=0.2",
            facecolor=cat_colours[cat],
            edgecolor=edge_colour,
            lw=1.2,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            lx + i * 4.2 + 0.85,
            ly + 0.27,
            name,
            fontsize=font_size - 1,
            va="center",
            color=text_colour,
        )

    ax.set_title(title, fontsize=font_size + 5, pad=20)
    ax.set_xlim(-2.0, 24.5)
    ax.set_ylim(-3.5, 12.0)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, ax


def plot_straddle_payoff(
    ax=None,
    *,
    spot: float = 100.0,
    strike: float | None = None,
    call_premium: float = 6.0,
    put_premium: float = 5.0,
    side: str = "long",
    price_range: tuple[float, float] | None = None,
    n_points: int = 301,
    title: str | None = None,
    figsize: tuple[float, float] = (10.5, 5.8),
):
    """
    Plot a plain-vanilla straddle payoff diagram.

    The default is a long ATM straddle: buy one call and one put with the same
    strike and expiry. Set ``side="short"`` to show the short-straddle mirror.
    """
    set_plot_style()
    side_norm = str(side).lower().strip()
    if side_norm not in {"long", "short"}:
        raise ValueError("side must be 'long' or 'short'.")

    k = float(spot if strike is None else strike)
    premium = float(call_premium) + float(put_premium)
    if price_range is None:
        lo = max(0.0, k - 4.0 * max(premium, 0.15 * k))
        hi = k + 4.0 * max(premium, 0.15 * k)
    else:
        lo, hi = map(float, price_range)
    s_grid = np.linspace(lo, hi, int(n_points))

    call_payoff = np.maximum(s_grid - k, 0.0) - float(call_premium)
    put_payoff = np.maximum(k - s_grid, 0.0) - float(put_premium)
    long_straddle = call_payoff + put_payoff
    net_payoff = long_straddle if side_norm == "long" else -long_straddle

    left_be = k - premium
    right_be = k + premium

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.axhline(0.0, color="#222222", lw=1.0)
    ax.axvline(k, color="#444444", lw=1.0, ls="--", label="strike")
    if left_be > lo:
        ax.axvline(left_be, color=LAB_COLORS[3], lw=0.9, ls=":", label="break-even")
    if right_be < hi:
        ax.axvline(right_be, color=LAB_COLORS[3], lw=0.9, ls=":")

    ax.plot(s_grid, call_payoff, lw=1.0, ls="--", alpha=0.65, color=LAB_COLORS[0], label="call leg")
    ax.plot(s_grid, put_payoff, lw=1.0, ls="--", alpha=0.65, color=LAB_COLORS[1], label="put leg")
    ax.plot(
        s_grid,
        net_payoff,
        lw=2.4,
        color=LAB_COLORS[2],
        label=f"{side_norm} straddle net payoff",
    )

    if side_norm == "long":
        ax.fill_between(
            s_grid,
            net_payoff,
            0.0,
            where=net_payoff < 0,
            color=LAB_COLORS[3],
            alpha=0.16,
            label="premium at risk",
        )
        ax.fill_between(s_grid, net_payoff, 0.0, where=net_payoff > 0, color=LAB_COLORS[4], alpha=0.12)
        ax.annotate(
            f"max loss = premium paid\n{premium:,.2f}",
            xy=(k, -premium),
            xytext=(k, -premium - 0.22 * max(premium, 1.0)),
            ha="center",
            va="top",
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#333333"},
        )
    else:
        ax.fill_between(
            s_grid,
            net_payoff,
            0.0,
            where=net_payoff > 0,
            color=LAB_COLORS[4],
            alpha=0.16,
            label="premium received",
        )
        ax.fill_between(s_grid, net_payoff, 0.0, where=net_payoff < 0, color=LAB_COLORS[3], alpha=0.12)
        ax.annotate(
            f"max gain = premium received\n{premium:,.2f}",
            xy=(k, premium),
            xytext=(k, premium + 0.18 * max(premium, 1.0)),
            ha="center",
            va="bottom",
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#333333"},
        )

    ax.text(left_be, 0.0, f"  {left_be:,.1f}", va="bottom", fontsize=8, color=LAB_COLORS[3])
    ax.text(right_be, 0.0, f"  {right_be:,.1f}", va="bottom", fontsize=8, color=LAB_COLORS[3])
    ax.text(k, 0.0, f"  K={k:,.1f}", va="top", fontsize=8, color="#333333")
    ax.set_xlabel("Underlying price at expiry")
    ax.set_ylabel("Profit / loss per straddle")
    ax.set_title(title or f"{side_norm.title()} straddle payoff: call + put, same strike and expiry")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    return fig, ax


plot_bsm_comp_graph = plot_computation_dag

__all__ = ["plot_bsm_comp_graph", "plot_computation_dag", "plot_straddle_payoff"]
