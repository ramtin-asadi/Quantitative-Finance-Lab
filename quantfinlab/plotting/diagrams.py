from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
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
        text_x = k + 0.18 * (hi - lo)
        text_y = -premium - 0.18 * max(hi - lo, premium)
        ax.annotate(
            f"Max loss\npremium paid = {premium:,.2f}",
            xy=(k, -premium),
            xytext=(text_x, text_y),
            ha="left",
            va="top",
            linespacing=1.35,
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#333333"},
            bbox={"boxstyle": "round,pad=0.30", "fc": "white", "ec": "#bbbbbb", "alpha": 0.92},
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


def tree_structure(ax=None, *, steps: int = 3, spot: float = 100.0, up: float = 1.12, down: float = 0.90, strike: float = 100.0, rate: float = 0.04, p: float = 0.52, title: str | None = None):
    set_plot_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
    else:
        fig = ax.get_figure()
    ax.axis("off")
    n = int(max(2, steps))
    disc = np.exp(-float(rate) / max(n, 1))
    stock = {}
    value = {}
    exercise = {}
    for i in range(n + 1):
        for j in range(i + 1):
            stock[(i, j)] = float(spot) * (float(up) ** j) * (float(down) ** (i - j))
    for j in range(n + 1):
        payoff = max(float(strike) - stock[(n, j)], 0.0)
        value[(n, j)] = payoff
        exercise[(n, j)] = payoff > 0.0
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            payoff = max(float(strike) - stock[(i, j)], 0.0)
            continuation = disc * (float(p) * value[(i + 1, j + 1)] + (1.0 - float(p)) * value[(i + 1, j)])
            value[(i, j)] = max(payoff, continuation)
            exercise[(i, j)] = payoff > continuation and payoff > 0.0
    positions = {}
    for i in range(n + 1):
        for j in range(i + 1):
            positions[(i, j)] = (i, j - 0.5 * i)
    for i in range(n):
        for j in range(i + 1):
            x0, y0 = positions[(i, j)]
            for jj, label in ((j, "d"), (j + 1, "u")):
                x1, y1 = positions[(i + 1, jj)]
                ax.add_patch(FancyArrowPatch((x0 + 0.13, y0), (x1 - 0.13, y1), arrowstyle="-|>", mutation_scale=10, lw=1.0, color="#6b7280", alpha=0.75))
                ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.06, label, ha="center", va="center", fontsize=8, color="#374151")
    for i in range(n + 1):
        for j in range(i + 1):
            x, y = positions[(i, j)]
            terminal = i == n
            early = bool(exercise[(i, j)]) and not terminal
            face = "#dbeafe" if not terminal else "#fef3c7"
            if early:
                face = "#fecaca"
            box = mpatches.FancyBboxPatch((x - 0.34, y - 0.18), 0.68, 0.36, boxstyle="round,pad=0.035,rounding_size=0.04", fc=face, ec="#111827", lw=1.0)
            ax.add_patch(box)
            if i == 0:
                label = f"S0={stock[(i, j)]:.0f}\nV={value[(i, j)]:.2f}"
            elif terminal:
                label = f"S={stock[(i, j)]:.0f}\npayoff={value[(i, j)]:.2f}"
            else:
                payoff = max(float(strike) - stock[(i, j)], 0.0)
                label = f"S={stock[(i, j)]:.0f}\nmax({payoff:.2f}, C)"
            ax.text(x, y, label, ha="center", va="center", fontsize=8)
    ax.text(n + 0.35, 0.58 * n, "terminal payoff", ha="left", va="center", fontsize=9, color="#92400e")
    ax.text(0.35, -0.58 * n, "backward induction:\nV = max(exercise, continuation)", ha="left", va="center", fontsize=9, color="#991b1b")
    handles = [
        mpatches.Patch(fc="#dbeafe", ec="#111827", label="continuation node"),
        mpatches.Patch(fc="#fef3c7", ec="#111827", label="terminal payoff"),
        mpatches.Patch(fc="#fecaca", ec="#111827", label="early exercise"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=8)
    ax.set_xlim(-0.65, n + 1.45)
    ax.set_ylim(-n / 2 - 0.85, n / 2 + 0.85)
    ax.set_title(title or "American binomial tree: stock moves, terminal payoff, and backward max step")
    fig.tight_layout()
    return fig, ax


def overlay_payoffs(axs=None, *, spot: float = 100.0, call_strike: float = 105.0, put_strike: float = 95.0, call_premium: float = 2.5, put_premium: float = 2.0):
    set_plot_style()
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
    else:
        axs = np.asarray(axs).ravel()
        fig = axs[0].get_figure()
    s = np.linspace(0.65 * spot, 1.35 * spot, 301)
    stock = s - spot
    short_call = -np.maximum(s - call_strike, 0.0) + call_premium
    long_put = np.maximum(put_strike - s, 0.0) - put_premium
    covered = stock + short_call
    protective = stock + long_put
    collar = stock + short_call + long_put
    specs = [
        (covered, [("stock", stock), ("short call", short_call)], "covered call", [call_strike], spot - call_premium),
        (protective, [("stock", stock), ("long put", long_put)], "protective put", [put_strike], spot + put_premium),
        (collar, [("stock", stock), ("short call", short_call), ("long put", long_put)], "collar", [put_strike, call_strike], spot - call_premium + put_premium),
    ]
    for ax, (combined, legs, label, strikes, breakeven) in zip(axs[:3], specs):
        ax.axhline(0.0, color="#222222", lw=0.8)
        ax.axvline(spot, color="#666666", lw=0.8, ls="--")
        for name, leg in legs:
            ax.plot(s, leg, lw=1.1, alpha=0.62, ls="--", label=name)
        ax.plot(s, combined, lw=2.2, color="#111827", label="combined")
        for k in strikes:
            ax.axvline(k, color="#9ca3af", lw=0.9)
            ax.text(k, ax.get_ylim()[0], f"K={k:.0f}", ha="center", va="bottom", fontsize=7)
        ax.axvline(breakeven, color="#ef4444", lw=0.9, ls=":")
        ax.fill_between(s, combined, 0.0, where=combined < 0, color="#fecaca", alpha=0.18)
        ax.fill_between(s, combined, 0.0, where=combined > 0, color="#bbf7d0", alpha=0.15)
        ax.set_xlabel("final underlying price")
        ax.set_ylabel("P&L per share")
        ax.set_title(label)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig, axs


def plot_fixed_float_swap_diagram(
    ax=None,
    *,
    title: str = "Fixed-for-floating swap: payer and receiver views",
    figsize: tuple[float, float] = (13.0, 6.3),
):
    set_plot_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)

    bg = "#fbfbfb"
    panel_edge = "#d5d9df"
    fixed_color = "#1f77b4"
    float_color = "#c44e52"
    receive_color = "#eaf3ff"
    pay_color = "#fff1ed"
    dealer_color = "#f6f7f9"

    ax.set_facecolor(bg)
    panel = mpatches.FancyBboxPatch(
        (0.35, 0.45),
        13.3,
        5.75,
        boxstyle="round,pad=0.22,rounding_size=0.18",
        fc="white",
        ec=panel_edge,
        lw=1.2,
    )
    ax.add_patch(panel)

    boxes = {
        "receiver": (0.95, 3.75, 3.1, 1.25, receive_color, "Receiver fixed\nadds duration"),
        "dealer_top": (5.45, 3.75, 3.1, 1.25, dealer_color, "Swap counterparty\nclears net cashflows"),
        "payer": (9.95, 3.75, 3.1, 1.25, pay_color, "Payer fixed\nremoves duration"),
        "dealer_bottom": (5.45, 1.55, 3.1, 1.25, dealer_color, "Same swap,\nopposite side"),
    }
    for x, y, w, h, color, text in boxes.values():
        patch = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.18,rounding_size=0.16",
            fc=color,
            ec="#30343b",
            lw=1.1,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, fontweight="bold")

    def arrow(start, end, color, rad=0.0):
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            lw=2.1,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(patch)
        return patch

    arrow((4.05, 4.72), (5.45, 4.72), fixed_color)
    arrow((5.45, 4.08), (4.05, 4.08), float_color)
    ax.text(4.75, 5.02, "receives fixed", ha="center", va="bottom", fontsize=9, color=fixed_color)
    ax.text(4.75, 3.75, "pays floating", ha="center", va="top", fontsize=9, color=float_color)

    arrow((8.55, 4.72), (9.95, 4.72), float_color)
    arrow((9.95, 4.08), (8.55, 4.08), fixed_color)
    ax.text(9.25, 5.02, "receives floating", ha="center", va="bottom", fontsize=9, color=float_color)
    ax.text(9.25, 3.75, "pays fixed", ha="center", va="top", fontsize=9, color=fixed_color)

    arrow((5.45, 2.42), (4.05, 2.42), fixed_color)
    arrow((4.05, 1.78), (5.45, 1.78), float_color)
    ax.text(4.75, 2.72, "fixed leg", ha="center", va="bottom", fontsize=9, color=fixed_color)
    ax.text(4.75, 1.45, "floating leg", ha="center", va="top", fontsize=9, color=float_color)

    arrow((8.55, 2.42), (9.95, 2.42), fixed_color)
    arrow((9.95, 1.78), (8.55, 1.78), float_color)
    ax.text(9.25, 2.72, "fixed leg", ha="center", va="bottom", fontsize=9, color=fixed_color)
    ax.text(9.25, 1.45, "floating leg", ha="center", va="top", fontsize=9, color=float_color)

    ax.text(2.50, 5.75, "Receiver swap", ha="center", fontsize=11, fontweight="bold", color="#20242a")
    ax.text(11.50, 5.75, "Payer swap", ha="center", fontsize=11, fontweight="bold", color="#20242a")
    ax.plot([7.0, 7.0], [0.8, 5.95], color=panel_edge, lw=1.0, ls="--")
    ax.text(
        7.0,
        0.85,
        "Synthetic overlay notional is set from target DV01; the figure shows cashflow direction, not a market OIS curve.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555b63",
    )
    ax.set_title(title, fontsize=14, pad=14)
    fig.tight_layout()
    return fig, ax


def _course_canvas(ax=None, *, figsize=(12.0, 6.2), title: str | None = None, xlim=(0, 12), ylim=(0, 6)):
    set_plot_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    else:
        fig = ax.get_figure()
        fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, fontweight="normal", loc="left", pad=16, color=LAB_COLORS[2])
        ax.plot([0.0, 0.16], [1.02, 1.02], transform=ax.transAxes, color=LAB_COLORS[0], lw=2.5, clip_on=False)
    return fig, ax


def _round_box(
    ax,
    xy,
    wh,
    text,
    *,
    fc="white",
    ec=LAB_COLORS[2],
    lw=1.2,
    fontsize=10,
    weight="normal",
    text_color=LAB_COLORS[2],
    alpha=1.0,
):
    x, y = xy
    w, h = wh
    shadow = mpatches.FancyBboxPatch(
        (x + 0.055, y - 0.055),
        w,
        h,
        boxstyle="round,pad=0.10,rounding_size=0.10",
        fc=LAB_COLORS[8],
        alpha=0.15,
        ec="none",
        zorder=1,
    )
    box = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.10,rounding_size=0.10",
        fc=fc,
        ec=ec,
        lw=lw,
        alpha=alpha,
        zorder=2,
    )
    ax.add_patch(shadow)
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=text_color,
        linespacing=1.25,
        zorder=3,
    )
    return box


def _arrow(ax, start, end, *, color=LAB_COLORS[2], lw=1.7, rad=0.0, scale=14):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=scale,
        lw=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def _soft_label(ax, xy, text, *, fontsize=9, color=LAB_COLORS[2], ha="center"):
    ax.text(
        xy[0],
        xy[1],
        text,
        ha=ha,
        va="center",
        fontsize=fontsize,
        color=color,
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": LAB_COLORS[8], "alpha": 0.35},
        zorder=4,
    )


def _course_header(fig, title):
    fig.suptitle(title, fontsize=14, fontweight="normal", x=0.06, ha="left", color=LAB_COLORS[2])
    fig.add_artist(Line2D([0.06, 0.16], [0.91, 0.91], transform=fig.transFigure, color=LAB_COLORS[0], lw=2.5))


def ml_pipeline(ax=None, *, title: str = "Regime learning pipeline"):
    fig, ax = _course_canvas(ax, figsize=(13.5, 5.2), title=title, xlim=(0, 13.5), ylim=(0, 5.0))
    boxes = [
        (0.45, 2.65, "Daily ETF\nprices", LAB_COLORS[0]),
        (2.30, 2.65, "Returns and\nrolling features", LAB_COLORS[3]),
        (4.35, 2.65, "Diagnostics\nVIF, PCA, RF", LAB_COLORS[4]),
        (6.35, 2.65, "Models\nclusters, HMM,\nclassifiers", LAB_COLORS[6]),
        (8.65, 2.65, "Month-end\nprobabilities", LAB_COLORS[5]),
        (10.75, 2.65, "Regime-aware\nweights", LAB_COLORS[9]),
        (12.20, 0.95, "Costed\nbacktest", LAB_COLORS[2]),
    ]
    centers = []
    for x, y, text, color in boxes:
        w = 1.45 if x < 12 else 1.05
        h = 1.0
        _round_box(ax, (x, y), (w, h), text, fc=color, ec=color, fontsize=9, text_color="white")
        centers.append((x + w / 2, y + h / 2))
    for a, b in zip(centers[:6], centers[1:6], strict=False):
        _arrow(ax, (a[0] + 0.78, a[1]), (b[0] - 0.78, b[1]), color=LAB_COLORS[0])
    _arrow(ax, (11.50, 2.65), (12.30, 1.95), color=LAB_COLORS[0], rad=-0.10)
    _soft_label(ax, (3.30, 1.05), "Daily model, monthly trading: information is sampled at rebalance dates")
    ax.plot([0.55, 11.55], [2.18, 2.18], color=LAB_COLORS[8], lw=1.0, ls="--")
    ax.text(5.9, 1.90, "training signal available at date t", ha="center", fontsize=8, color=LAB_COLORS[2])
    fig.tight_layout()
    return fig, ax


def unsupervised_supervised(ax=None, *, title: str = "Unsupervised discovery vs supervised prediction"):
    fig, ax = _course_canvas(ax, figsize=(13.2, 5.8), title=title, xlim=(0, 13.2), ylim=(0, 5.4))
    ax.text(0.55, 3.9, "Discovery", color=LAB_COLORS[3], fontsize=11)
    ax.text(0.55, 1.55, "Prediction", color=LAB_COLORS[6], fontsize=11)
    top = [("X(t)\nfeatures", 2.0, LAB_COLORS[0]), ("Find latent\nstates", 5.0, LAB_COLORS[3]), ("Profile and\norder regimes", 8.0, LAB_COLORS[9]), ("State at t", 11.0, LAB_COLORS[2])]
    bottom = [("X(t)\nfeatures", 2.0, LAB_COLORS[0]), ("Future 21d\nlabel", 5.0, LAB_COLORS[5]), ("Fit classifier", 8.0, LAB_COLORS[6]), ("P(state at t+1)", 11.0, LAB_COLORS[4])]
    for lane, y in [(top, 3.42), (bottom, 1.10)]:
        for text, x, color in lane:
            _round_box(ax, (x, y), (2.0, 0.82), text, fc=color, ec=color, fontsize=9, text_color="white")
        for left, right in zip(lane[:-1], lane[1:], strict=False):
            _arrow(ax, (left[1] + 2.08, y + 0.41), (right[1] - 0.08, y + 0.41), color=LAB_COLORS[2])
    ax.plot([0.5, 12.8], [2.66, 2.66], color=LAB_COLORS[8], lw=1.0, ls="--")
    _soft_label(ax, (6.7, 0.36), "Discovery interprets today's state; supervised learning predicts the next realized state")
    fig.tight_layout()
    return fig, ax


def kmeans_geometry(ax=None, *, title: str = "KMeans: nearest centroid geometry"):
    fig, ax = _course_canvas(ax, figsize=(8.5, 6.2), title=title, xlim=(-3.5, 3.8), ylim=(-3.0, 3.4))
    rng = np.random.default_rng(11)
    centers = np.array([[-0.92, 0.62], [0.92, 0.58], [0.0, -0.86]])
    colors = [LAB_COLORS[0], LAB_COLORS[3], LAB_COLORS[1]]
    for c, color in zip(centers, colors, strict=False):
        pts = rng.normal(c, [0.48, 0.42], size=(50, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=26, alpha=0.62, color=color, edgecolor="white", linewidth=0.4)
        ax.scatter(c[0], c[1], s=210, marker="X", color=LAB_COLORS[2], edgecolor="white", linewidth=1.4, zorder=5)
    xx = np.linspace(-3.4, 3.6, 250)
    yy = np.linspace(-2.9, 3.2, 250)
    X, Y = np.meshgrid(xx, yy)
    D = np.stack([(X - c[0]) ** 2 + (Y - c[1]) ** 2 for c in centers])
    lab = np.argmin(D, axis=0)
    ax.contourf(X, Y, lab, levels=[-0.5, 0.5, 1.5, 2.5], colors=colors, alpha=0.12)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig, ax


def agglomerative_tree(ax=None, *, title: str = "Agglomerative clustering: merge closest groups"):
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.5), facecolor="white")
    _course_header(fig, title)
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.3)
    ax.axis("off")
    leaves = [(0.9, 0.78), (2.0, 0.78), (3.35, 0.78), (5.9, 0.78), (7.15, 0.78), (8.55, 0.78)]
    for i, (x, y) in enumerate(leaves):
        ax.scatter(x, y, s=120, color=LAB_COLORS[[0, 0, 3, 3, 1, 1][i]], edgecolor="white", linewidth=1.0, zorder=4)
        ax.text(x, y - 0.38, f"x{i+1}", ha="center", fontsize=8)
    def merge(left, right, h):
        x1, y1 = left
        x2, y2 = right
        ax.plot([x1, x1, x2, x2], [y1, h, h, y2], color=LAB_COLORS[2], lw=1.7)
        return ((x1 + x2) / 2, h)
    a = merge(leaves[0], leaves[1], 1.62)
    b = merge(a, leaves[2], 2.45)
    c = merge(leaves[3], leaves[4], 1.72)
    d = merge(c, leaves[5], 2.55)
    merge(b, d, 3.72)
    ax.axhline(2.90, color=LAB_COLORS[1], lw=1.2, ls="--")
    ax.text(0.70, 3.10, "cut here: two clusters", fontsize=8.5, color=LAB_COLORS[1], va="bottom")
    ax.text(4.75, 4.45, "Linkage tree", ha="center", fontsize=10, color=LAB_COLORS[2])
    ax = axes[1]
    ax.set_xlim(-0.5, 5.4)
    ax.set_ylim(-0.5, 3.6)
    ax.axis("off")
    pts = np.array([[0.7, 1.45], [1.45, 1.45], [2.70, 1.45], [3.45, 1.45], [4.72, 1.45]])
    for xy, label in zip(pts, list("ABCDE"), strict=False):
        ax.scatter(*xy, s=260, color=LAB_COLORS[4], edgecolor="white", linewidth=1.0, zorder=4)
        ax.text(*xy, label, ha="center", va="center", fontsize=9, color="white")
    for center, width, height, color in [
        ((1.08, 1.45), 1.35, 0.80, LAB_COLORS[0]),
        ((3.08, 1.45), 1.35, 0.80, LAB_COLORS[3]),
        ((2.02, 1.45), 3.55, 1.45, LAB_COLORS[6]),
        ((2.70, 1.45), 5.38, 2.10, LAB_COLORS[2]),
    ]:
        ax.add_patch(mpatches.Ellipse(center, width, height, fill=False, ec=color, lw=1.5, ls="--"))
    ax.text(2.45, 3.02, "Nested cluster memberships", ha="center", fontsize=10, color=LAB_COLORS[2])
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig, axes


def gmm_mixture(ax=None, *, title: str = "Gaussian mixtures: soft regime membership"):
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), facecolor="white")
    _course_header(fig, title)
    ax = axes[0]
    rng = np.random.default_rng(22)
    specs = [
        ([-1.05, 0.65], [[0.50, 0.20], [0.20, 0.30]], LAB_COLORS[0]),
        ([1.00, 0.75], [[0.36, -0.17], [-0.17, 0.45]], LAB_COLORS[3]),
        ([0.10, -0.90], [[0.68, 0.0], [0.0, 0.24]], LAB_COLORS[1]),
    ]
    for mean, cov, color in specs:
        pts = rng.multivariate_normal(mean, cov, size=70)
        ax.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.38, color=color)
        vals, vecs = np.linalg.eigh(np.asarray(cov))
        angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        for scale, alpha in [(1.3, 0.32), (2.2, 0.16)]:
            ell = mpatches.Ellipse(mean, scale * np.sqrt(vals[1]) * 2, scale * np.sqrt(vals[0]) * 2, angle=angle, fc=color, ec=color, alpha=alpha, lw=1.2)
            ax.add_patch(ell)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.grid(True, alpha=0.2)
    ax.set_title("Feature space", fontsize=11, fontweight="normal", color=LAB_COLORS[2])
    ax = axes[1]
    x = np.linspace(-4, 4, 400)
    params = [(-1.45, 0.48, LAB_COLORS[0]), (0.0, 0.38, LAB_COLORS[3]), (1.65, 0.62, LAB_COLORS[1])]
    mix = np.zeros_like(x)
    for mu, sigma, color in params:
        density = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        mix += density / 3.0
        ax.plot(x, density / 3.0, color=color, lw=2.1)
        ax.axvline(mu, color=color, lw=1.0, ls="--")
    ax.plot(x, mix, color=LAB_COLORS[2], lw=2.3, label="mixture")
    dots = np.array([-2.0, -1.62, -1.33, -1.05, -0.43, -0.12, 0.18, 0.48, 1.08, 1.42, 1.72, 2.15])
    ax.scatter(dots, np.zeros_like(dots), color=LAB_COLORS[2], s=36, zorder=4)
    ax.set_title("Overlapping component densities", fontsize=11, fontweight="normal", color=LAB_COLORS[2])
    ax.set_xlabel("one regime feature")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="x", alpha=0.18)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig, axes


def bayesian_mixture(ax=None, *, title: str = "Bayesian mixture: unused components shrink away"):
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), facecolor="white")
    _course_header(fig, title)
    x = np.linspace(-4.2, 4.2, 500)
    params = [
        (-2.15, 0.48, 0.30, LAB_COLORS[0]),
        (-0.65, 0.54, 0.26, LAB_COLORS[3]),
        (0.85, 0.47, 0.21, LAB_COLORS[1]),
        (2.20, 0.56, 0.14, LAB_COLORS[6]),
        (3.05, 0.32, 0.05, LAB_COLORS[8]),
        (-3.05, 0.34, 0.04, LAB_COLORS[4]),
    ]
    mixture = np.zeros_like(x)
    for mu, sigma, weight, color in params:
        density = weight * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        mixture += density
        alpha = 1.0 if weight >= 0.08 else 0.38
        axes[0].plot(x, density, color=color, lw=2.0, alpha=alpha)
        axes[0].fill_between(x, 0, density, color=color, alpha=0.11 if weight >= 0.08 else 0.05)
    axes[0].plot(x, mixture, color=LAB_COLORS[2], lw=2.3)
    axes[0].set_title("Upper bound: six candidate components", fontsize=11, fontweight="normal", color=LAB_COLORS[2])
    axes[0].set_xlabel("regime feature")
    axes[0].set_ylabel("density")
    axes[0].grid(True, alpha=0.18)
    weights = np.array([p[2] for p in params])
    labels = [f"component {i}" for i in range(1, 7)]
    colors = [p[3] for p in params]
    ypos = np.arange(6)[::-1]
    axes[1].barh(ypos, weights, color=colors, edgecolor="white", height=0.62)
    axes[1].axvline(0.08, color=LAB_COLORS[2], lw=1.1, ls="--")
    for y_pos, weight in zip(ypos, weights, strict=False):
        text = "active" if weight >= 0.08 else "shrunk"
        axes[1].text(weight + 0.012, y_pos, text, va="center", fontsize=9, color=LAB_COLORS[2])
    axes[1].set_yticks(ypos)
    axes[1].set_yticklabels(labels)
    axes[1].set_xlim(0, 0.37)
    axes[1].set_xlabel("posterior component weight")
    axes[1].set_title("Posterior keeps only supported states", fontsize=11, fontweight="normal", color=LAB_COLORS[2])
    axes[1].grid(True, axis="x", alpha=0.18)
    axes[1].annotate(
        "prior shrinks unsupported states",
        xy=(0.05, 0),
        xytext=(0.13, 0.52),
        fontsize=8.5,
        color=LAB_COLORS[2],
        arrowprops={"arrowstyle": "->", "color": LAB_COLORS[4], "lw": 1.2},
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": LAB_COLORS[8], "alpha": 0.55},
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig, axes


def markov_chain(ax=None, *, title: str = "Markov chain: regimes persist and transition"):
    fig, ax = _course_canvas(ax, figsize=(11.0, 5.8), title=title, xlim=(0, 11.0), ylim=(0, 5.4))
    states = [
        ("Risk-on", 1.9, LAB_COLORS[3]),
        ("Neutral", 5.5, LAB_COLORS[0]),
        ("Stress", 9.1, LAB_COLORS[1]),
    ]
    for name, x, color in states:
        _round_box(ax, (x - 0.88, 2.55), (1.76, 0.84), name, fc=color, ec=color, fontsize=10, text_color="white")
        _arrow(ax, (x - 0.52, 3.46), (x + 0.52, 3.46), color=color, rad=-0.72, scale=12)
    ax.text(1.9, 4.32, "0.82", ha="center", fontsize=9, color=LAB_COLORS[3])
    ax.text(5.5, 4.32, "0.67", ha="center", fontsize=9, color=LAB_COLORS[0])
    ax.text(9.1, 4.32, "0.76", ha="center", fontsize=9, color=LAB_COLORS[1])
    _arrow(ax, (2.83, 3.05), (4.55, 3.05), color=LAB_COLORS[0])
    _arrow(ax, (6.43, 3.05), (8.15, 3.05), color=LAB_COLORS[1])
    _arrow(ax, (4.55, 2.74), (2.83, 2.74), color=LAB_COLORS[3], rad=-0.12)
    _arrow(ax, (8.15, 2.74), (6.43, 2.74), color=LAB_COLORS[4], rad=-0.12)
    ax.text(3.68, 3.30, "0.12", ha="center", fontsize=9, color=LAB_COLORS[0])
    ax.text(7.28, 3.30, "0.20", ha="center", fontsize=9, color=LAB_COLORS[1])
    ax.text(3.68, 2.33, "0.09", ha="center", fontsize=9, color=LAB_COLORS[3])
    ax.text(7.28, 2.33, "0.17", ha="center", fontsize=9, color=LAB_COLORS[4])
    _arrow(ax, (8.62, 2.18), (2.40, 2.18), color=LAB_COLORS[4], rad=-0.18)
    ax.text(5.5, 1.55, "tail transition: 0.07", ha="center", fontsize=9, color=LAB_COLORS[4])
    _soft_label(ax, (5.5, 0.68), "Every row of the transition matrix gives P(S_t | S_{t-1})")
    fig.tight_layout()
    return fig, ax


def hidden_markov_model(ax=None, *, title: str = "Hidden Markov model: latent states emit observed features"):
    fig, ax = _course_canvas(ax, figsize=(12.8, 6.0), title=title, xlim=(0, 12.8), ylim=(0, 5.8))
    xs = [3.05, 6.65, 10.25]
    states = [("Risk-on\nS(t-1)", LAB_COLORS[3]), ("Neutral\nS(t)", LAB_COLORS[0]), ("Stress\nS(t+1)", LAB_COLORS[1])]
    observed = ["positive return\nlow volatility", "mixed signals\nmoderate volatility", "negative return\nhigh volatility"]
    ax.plot([1.55, 1.55], [1.30, 4.65], color=LAB_COLORS[8], lw=1.0, ls="--")
    ax.text(0.35, 4.05, "Hidden\nstate", ha="left", va="center", fontsize=10, color=LAB_COLORS[2], linespacing=1.25)
    ax.text(0.35, 1.90, "Observed\nfeatures", ha="left", va="center", fontsize=10, color=LAB_COLORS[2], linespacing=1.25)
    for i, (x, (state, color), obs) in enumerate(zip(xs, states, observed, strict=False)):
        _round_box(ax, (x - 0.90, 3.62), (1.80, 0.86), state, fc=color, ec=color, fontsize=9, text_color="white")
        _round_box(ax, (x - 1.02, 1.46), (2.04, 0.82), obs, fc="white", ec=color, fontsize=8, text_color=LAB_COLORS[2])
        _arrow(ax, (x, 3.52), (x, 2.34), color=color)
        ax.text(x + 0.12, 2.88, "emits", fontsize=8, color=color, ha="left", va="center")
        if i < len(xs) - 1:
            _arrow(ax, (x + 0.98, 4.05), (xs[i + 1] - 0.98, 4.05), color=LAB_COLORS[2])
    ax.text(4.85, 4.38, "transition", ha="center", fontsize=8, color=LAB_COLORS[2])
    ax.text(8.45, 4.38, "transition", ha="center", fontsize=8, color=LAB_COLORS[2])
    _soft_label(ax, (6.75, 0.62), "Filtering infers state probabilities from the feature sequence")
    fig.tight_layout()
    return fig, ax


def logistic_boundary(ax=None, *, title: str = "Logistic regression: probability instead of an unbounded score"):
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), facecolor="white")
    _course_header(fig, title)
    x = np.linspace(-3.4, 3.4, 240)
    obs_x = np.array([-3.0, -2.6, -2.1, -1.6, -1.2, 0.65, 1.05, 1.45, 1.95, 2.5, 2.9])
    obs_y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    axes[0].scatter(obs_x, obs_y, color=LAB_COLORS[1], s=34, zorder=4)
    axes[0].plot(x, 0.34 * x + 0.52, color=LAB_COLORS[2], lw=2.0)
    axes[0].set_title("Linear score", fontsize=11, fontweight="normal", color=LAB_COLORS[2])
    axes[1].scatter(obs_x, obs_y, color=LAB_COLORS[1], s=34, zorder=4)
    axes[1].plot(x, 1.0 / (1.0 + np.exp(-2.0 * x)), color=LAB_COLORS[0], lw=2.5)
    axes[1].axhline(0.5, color=LAB_COLORS[3], lw=1.1, ls="--")
    axes[1].set_title("Logistic probability", fontsize=11, fontweight="normal", color=LAB_COLORS[2])
    for a in axes:
        a.set_xlabel("feature score")
        a.set_yticks([0, 0.5, 1])
        a.set_yticklabels(["state 0", "0.5", "state 1"])
        a.set_ylim(-0.12, 1.15)
        a.grid(True, alpha=0.18)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig, axes


def svm_margin(ax=None, *, title: str = "Linear boundary versus maximum margin"):
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.3), facecolor="white")
    _course_header(fig, title)
    rng = np.random.default_rng(8)
    a = rng.normal([-1.0, -0.4], [0.55, 0.45], size=(48, 2))
    b = rng.normal([1.1, 0.5], [0.55, 0.45], size=(48, 2))
    x = np.linspace(-3, 3, 200)
    y = -0.75 * x + 0.20
    for panel in axes:
        panel.scatter(a[:, 0], a[:, 1], s=25, color=LAB_COLORS[0], alpha=0.65, edgecolor="white", linewidth=0.4)
        panel.scatter(b[:, 0], b[:, 1], s=25, color=LAB_COLORS[1], alpha=0.65, edgecolor="white", linewidth=0.4)
        panel.plot(x, y, color=LAB_COLORS[2], lw=2.0)
        panel.set_xlim(-3, 3)
        panel.set_ylim(-2.6, 2.8)
        panel.set_xlabel("feature 1")
        panel.grid(True, alpha=0.18)
    axes[0].set_ylabel("feature 2")
    axes[0].set_title("Logistic boundary", fontsize=11, fontweight="normal", color=LAB_COLORS[2])
    axes[1].plot(x, y + 0.52, color=LAB_COLORS[3], lw=1.4, ls="--")
    axes[1].plot(x, y - 0.52, color=LAB_COLORS[3], lw=1.4, ls="--")
    axes[1].fill_between(x, y - 0.52, y + 0.52, color=LAB_COLORS[3], alpha=0.10)
    axes[1].set_title("SVM margin", fontsize=11, fontweight="normal", color=LAB_COLORS[2])
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig, axes


def lda_projection(ax=None, *, title: str = "LDA: project onto the most separating direction"):
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), facecolor="white")
    _course_header(fig, title)
    rng = np.random.default_rng(18)
    a = rng.multivariate_normal([0.1, 1.1], [[0.35, 0.15], [0.15, 0.34]], size=38)
    b = rng.multivariate_normal([1.2, -0.25], [[0.38, 0.14], [0.14, 0.34]], size=38)
    axes[0].scatter(a[:, 0], a[:, 1], s=32, color=LAB_COLORS[3], alpha=0.78, label="state 0")
    axes[0].scatter(b[:, 0], b[:, 1], s=32, marker="^", color=LAB_COLORS[0], alpha=0.78, label="state 1")
    line_x = np.linspace(-1.1, 2.4, 100)
    axes[0].plot(line_x, -0.85 * line_x + 1.15, color=LAB_COLORS[4], lw=2.0)
    axes[0].set_title("Original features", fontsize=11, fontweight="normal", color=LAB_COLORS[2])
    axes[0].legend(frameon=False, fontsize=8)
    proj_a = a @ np.array([-0.65, 0.76])
    proj_b = b @ np.array([-0.65, 0.76])
    axes[1].scatter(proj_a, np.full_like(proj_a, 0.62), s=32, color=LAB_COLORS[3], alpha=0.78)
    axes[1].scatter(proj_b, np.full_like(proj_b, 0.38), s=32, marker="^", color=LAB_COLORS[0], alpha=0.78)
    cut = 0.5 * (np.mean(proj_a) + np.mean(proj_b))
    axes[1].axvline(cut, color=LAB_COLORS[4], lw=2.0)
    axes[1].set_title("One discriminant axis", fontsize=11, fontweight="normal", color=LAB_COLORS[2])
    axes[1].set_yticks([0.38, 0.62])
    axes[1].set_yticklabels(["state 1", "state 0"])
    for panel in axes:
        panel.grid(True, alpha=0.18)
        panel.set_xlabel("discriminant score" if panel is axes[1] else "feature 1")
    axes[0].set_ylabel("feature 2")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig, axes


def knn_neighbors(ax=None, *, title: str = "KNN: local vote around the query point"):
    fig, ax = _course_canvas(ax, figsize=(8.3, 6.5), title=title, xlim=(-3.0, 3.2), ylim=(-2.65, 2.8))
    rng = np.random.default_rng(19)
    groups = [
        (rng.normal([-1.25, 0.70], [0.48, 0.39], size=(22, 2)), LAB_COLORS[0], "o", "State A"),
        (rng.normal([1.22, 0.72], [0.47, 0.42], size=(22, 2)), LAB_COLORS[3], "s", "State B"),
        (rng.normal([-0.25, -1.05], [0.50, 0.36], size=(22, 2)), LAB_COLORS[1], "D", "State C"),
    ]
    pts = np.vstack([group[0] for group in groups])
    lab = np.concatenate([np.full(len(group[0]), i) for i, group in enumerate(groups)])
    xx, yy = np.meshgrid(np.linspace(-2.9, 3.1, 160), np.linspace(-2.55, 2.65, 160))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    dist = ((grid[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argpartition(dist, kth=6, axis=1)[:, :7]
    pred = np.array([np.bincount(lab[row], minlength=3).argmax() for row in nearest]).reshape(xx.shape)
    ax.contourf(xx, yy, pred, levels=[-0.5, 0.5, 1.5, 2.5], colors=[LAB_COLORS[0], LAB_COLORS[3], LAB_COLORS[1]], alpha=0.09)
    ax.contour(xx, yy, pred, levels=[0.5, 1.5], colors=[LAB_COLORS[0], LAB_COLORS[3]], linewidths=1.25, linestyles="--", alpha=0.75)
    q = np.array([0.25, 0.10])
    qdist = np.sqrt(((pts - q) ** 2).sum(axis=1))
    neighbors = np.argsort(qdist)[:7]
    radius = qdist[neighbors[-1]]
    for p in pts[neighbors]:
        ax.plot([q[0], p[0]], [q[1], p[1]], color=LAB_COLORS[2], lw=0.9, alpha=0.65)
    for points, color, marker, label in groups:
        ax.scatter(points[:, 0], points[:, 1], s=42, color=color, marker=marker, alpha=0.9, edgecolor="white", linewidth=0.7, label=label, zorder=3)
    ax.add_patch(mpatches.Circle(q, radius, fill=False, ec=LAB_COLORS[4], lw=1.7, ls="--", zorder=4))
    ax.scatter(q[0], q[1], s=185, marker="*", color=LAB_COLORS[2], edgecolor="white", linewidth=0.9, zorder=6)
    ax.text(q[0] + 0.14, q[1] - 0.19, "query", fontsize=9, color=LAB_COLORS[2])
    _soft_label(ax, (1.58, -1.83), "k = 7 local vote", ha="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=3)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.grid(True, alpha=0.16)
    fig.tight_layout()
    return fig, ax


def decision_tree_split(ax=None, *, title: str = "Decision tree: axis-aligned rules"):
    fig, ax = _course_canvas(ax, figsize=(8.8, 5.8), title=title, xlim=(-3, 3), ylim=(-2.6, 2.8))
    rng = np.random.default_rng(33)
    a = rng.normal([-1.0, -0.2], [0.65, 0.55], size=(45, 2))
    b = rng.normal([1.1, 0.4], [0.65, 0.55], size=(45, 2))
    ax.scatter(a[:, 0], a[:, 1], s=28, color=LAB_COLORS[0], alpha=0.62, edgecolor="white", linewidth=0.4)
    ax.scatter(b[:, 0], b[:, 1], s=28, color=LAB_COLORS[1], alpha=0.62, edgecolor="white", linewidth=0.4)
    ax.axvline(-0.15, color=LAB_COLORS[2], lw=1.8)
    ax.plot([-0.15, 3], [0.85, 0.85], color=LAB_COLORS[2], lw=1.8)
    ax.plot([-3, -0.15], [-0.85, -0.85], color=LAB_COLORS[2], lw=1.8)
    _soft_label(ax, (-1.6, 2.1), "if x1 <= c")
    _soft_label(ax, (1.4, 2.1), "then split x2")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig, ax


def ensemble_bagging(ax=None, *, title: str = "Ensembles: parallel bagging versus sequential boosting"):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), facecolor="white")
    _course_header(fig, title)
    def tree(panel, x, y, color, scale=1.0):
        nodes = [
            (x, y), (x - 0.27 * scale, y - 0.34 * scale), (x + 0.27 * scale, y - 0.34 * scale),
            (x - 0.40 * scale, y - 0.68 * scale), (x - 0.12 * scale, y - 0.68 * scale),
            (x + 0.15 * scale, y - 0.68 * scale), (x + 0.42 * scale, y - 0.68 * scale),
        ]
        for start, end in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]:
            panel.plot([nodes[start][0], nodes[end][0]], [nodes[start][1], nodes[end][1]], color=color, lw=1.0)
        for px, py in nodes:
            panel.add_patch(mpatches.Circle((px, py), 0.065 * scale, fc="white", ec=color, lw=1.1, zorder=3))
    for panel, name in zip(axes, ["Bagging", "Boosting"], strict=False):
        panel.set_xlim(0, 6.2)
        panel.set_ylim(0, 6.2)
        panel.axis("off")
        panel.set_title(name, fontsize=12, fontweight="normal", color=LAB_COLORS[2])
    ax = axes[0]
    _round_box(ax, (2.12, 5.30), (1.86, 0.52), "Training data", fc=LAB_COLORS[0], ec=LAB_COLORS[0], text_color="white")
    for x, text in [(0.78, "bootstrap 1"), (2.37, "bootstrap 2"), (3.96, "bootstrap M")]:
        _arrow(ax, (3.05, 5.27), (x + 0.67, 4.76), color=LAB_COLORS[2])
        _round_box(ax, (x, 4.18), (1.34, 0.44), text, fc="white", ec=LAB_COLORS[3], fontsize=8)
        tree(ax, x + 0.67, 3.72, LAB_COLORS[3], scale=0.83)
        _arrow(ax, (x + 0.67, 2.90), (3.05, 1.48), color=LAB_COLORS[2])
    _round_box(ax, (2.02, 0.84), (2.06, 0.52), "Average / vote", fc=LAB_COLORS[4], ec=LAB_COLORS[4], text_color="white")
    ax = axes[1]
    _round_box(ax, (2.02, 5.34), (2.12, 0.46), "Weighted data", fc=LAB_COLORS[0], ec=LAB_COLORS[0], fontsize=9, text_color="white")
    learners = [(0.32, LAB_COLORS[3], "Weak 1"), (2.40, LAB_COLORS[6], "Weak 2"), (4.48, LAB_COLORS[1], "Weak M")]
    _arrow(ax, (3.08, 5.29), (1.05, 4.65), color=LAB_COLORS[2], rad=0.12)
    for x, color, text in learners:
        _round_box(ax, (x, 4.10), (1.42, 0.46), text, fc=color, ec=color, fontsize=8.5, text_color="white")
        tree(ax, x + 0.72, 3.60, color, scale=0.78)
        _arrow(ax, (x + 0.72, 2.68), (3.08, 1.28), color=LAB_COLORS[2], lw=1.35, rad=0.04)
    _arrow(ax, (1.80, 4.34), (2.33, 4.34), color=LAB_COLORS[1], lw=1.5)
    _arrow(ax, (3.88, 4.34), (4.41, 4.34), color=LAB_COLORS[1], lw=1.5)
    ax.text(2.06, 4.66, "reweight", ha="center", va="bottom", fontsize=8, color=LAB_COLORS[1])
    ax.text(4.14, 4.66, "reweight", ha="center", va="bottom", fontsize=8, color=LAB_COLORS[1])
    _round_box(ax, (1.82, 0.66), (2.52, 0.50), "Weighted prediction", fc=LAB_COLORS[4], ec=LAB_COLORS[4], fontsize=9, text_color="white")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig, axes


def walkforward_split(ax=None, *, title: str = "Walk-forward split: train only on the past"):
    fig, ax = _course_canvas(ax, figsize=(12.5, 4.7), title=title, xlim=(0, 12.5), ylim=(0, 4.4))
    y0 = 2.7
    for i, start in enumerate([0.8, 2.6, 4.4, 6.2]):
        train_w = 3.0
        test_w = 0.72
        y = y0 - i * 0.55
        ax.add_patch(mpatches.Rectangle((start, y), train_w, 0.32, fc=LAB_COLORS[0], ec=LAB_COLORS[0], alpha=0.55, lw=1.0))
        ax.add_patch(mpatches.Rectangle((start + train_w, y), test_w, 0.32, fc=LAB_COLORS[1], ec=LAB_COLORS[1], alpha=0.70, lw=1.0))
        ax.text(start - 0.15, y + 0.16, f"fold {i+1}", ha="right", va="center", fontsize=8, color=LAB_COLORS[2])
    ax.text(2.0, 3.35, "training window", ha="center", fontsize=9, color=LAB_COLORS[0])
    ax.text(4.15, 3.35, "next month", ha="center", fontsize=9, color=LAB_COLORS[1])
    ax.arrow(0.8, 0.65, 9.9, 0.0, head_width=0.12, head_length=0.22, fc=LAB_COLORS[2], ec=LAB_COLORS[2], lw=1.0)
    ax.text(5.75, 0.35, "calendar time", ha="center", fontsize=9, color=LAB_COLORS[2])
    _soft_label(ax, (9.4, 2.6), "fit scaler and model\ninside each fold", ha="left")
    fig.tight_layout()
    return fig, ax


plot_bsm_comp_graph = plot_computation_dag

__all__ = [
    "agglomerative_tree",
    "bayesian_mixture",
    "decision_tree_split",
    "ensemble_bagging",
    "gmm_mixture",
    "hidden_markov_model",
    "kmeans_geometry",
    "knn_neighbors",
    "lda_projection",
    "logistic_boundary",
    "markov_chain",
    "ml_pipeline",
    "plot_bsm_comp_graph",
    "plot_computation_dag",
    "plot_fixed_float_swap_diagram",
    "plot_straddle_payoff",
    "svm_margin",
    "tree_structure",
    "overlay_payoffs",
    "unsupervised_supervised",
    "walkforward_split",
]
