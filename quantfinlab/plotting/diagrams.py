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


plot_bsm_comp_graph = plot_computation_dag

__all__ = [
    "plot_bsm_comp_graph",
    "plot_computation_dag",
    "plot_fixed_float_swap_diagram",
    "plot_straddle_payoff",
    "tree_structure",
    "overlay_payoffs",
]
