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
    n = int(max(2, steps))
    fig, ax = _course_canvas(
        ax,
        figsize=(10.5, 5.8),
        title=title or "American binomial tree: stock moves and backward induction",
        xlim=(-0.65, n + 1.45),
        ylim=(-n / 2 - 0.85, n / 2 + 0.85),
    )
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
                ax.add_patch(FancyArrowPatch((x0 + 0.13, y0), (x1 - 0.13, y1), arrowstyle="-|>", mutation_scale=10, lw=1.0, color=LAB_COLORS[8], alpha=0.95))
                ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.06, rf"${label}$", ha="center", va="center", fontsize=8, color=LAB_COLORS[2])
    for i in range(n + 1):
        for j in range(i + 1):
            x, y = positions[(i, j)]
            terminal = i == n
            early = bool(exercise[(i, j)]) and not terminal
            face = "#eaf3ff" if not terminal else "#fff7ed"
            if early:
                face = "#fff1f2"
            box = mpatches.FancyBboxPatch((x - 0.34, y - 0.18), 0.68, 0.36, boxstyle="round,pad=0.035,rounding_size=0.04", fc=face, ec=LAB_COLORS[2], lw=1.0)
            ax.add_patch(box)
            if i == 0:
                label = rf"$S_0={stock[(i, j)]:.0f}$" + "\n" + rf"$V={value[(i, j)]:.2f}$"
            elif terminal:
                label = rf"$S={stock[(i, j)]:.0f}$" + "\n" + rf"payoff $={value[(i, j)]:.2f}$"
            else:
                payoff = max(float(strike) - stock[(i, j)], 0.0)
                label = rf"$S={stock[(i, j)]:.0f}$" + "\n" + rf"$\max({payoff:.2f}, C)$"
            ax.text(x, y, label, ha="center", va="center", fontsize=8, color=LAB_COLORS[2])
    ax.text(n + 0.35, 0.58 * n, "terminal payoff", ha="left", va="center", fontsize=9, color="#9a3412")
    ax.text(0.35, -0.58 * n, "backward induction:\n" + r"$V=\max(\mathrm{exercise},\mathrm{continuation})$", ha="left", va="center", fontsize=9, color=LAB_COLORS[1])
    handles = [
        mpatches.Patch(fc="#eaf3ff", ec=LAB_COLORS[2], label="continuation node"),
        mpatches.Patch(fc="#fff7ed", ec=LAB_COLORS[2], label="terminal payoff"),
        mpatches.Patch(fc="#fff1f2", ec=LAB_COLORS[2], label="early exercise"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=8)
    fig.tight_layout()
    return fig, ax


def overlay_payoffs(axs=None, *, spot: float = 100.0, call_strike: float = 105.0, put_strike: float = 95.0, call_premium: float = 2.5, put_premium: float = 2.0, title: str | None = "Covered call, protective put, and collar"):
    set_plot_style()
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor="white")
    else:
        axs = np.asarray(axs).ravel()
        fig = axs[0].get_figure()
        fig.patch.set_facecolor("white")
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
    leg_colors = {
        "stock": LAB_COLORS[0],
        "short call": LAB_COLORS[1],
        "long put": LAB_COLORS[6],
    }
    if title:
        _course_header(fig, title)
    for ax, (combined, legs, label, strikes, breakeven) in zip(axs[:3], specs, strict=False):
        ax.set_facecolor("white")
        ax.axhline(0.0, color=LAB_COLORS[2], lw=0.8)
        ax.axvline(spot, color=LAB_COLORS[8], lw=0.8, ls="--")
        for name, leg in legs:
            ax.plot(s, leg, lw=1.1, alpha=0.72, ls="--", color=leg_colors.get(name, LAB_COLORS[8]), label=name)
        ax.plot(s, combined, lw=2.2, color=LAB_COLORS[2], label="combined")
        for k in strikes:
            ax.axvline(k, color=LAB_COLORS[8], lw=0.9)
            ax.text(k, ax.get_ylim()[0], rf"$K={k:.0f}$", ha="center", va="bottom", fontsize=7, color=LAB_COLORS[2])
        ax.axvline(breakeven, color=LAB_COLORS[1], lw=0.9, ls=":")
        ax.fill_between(s, combined, 0.0, where=combined < 0, color=LAB_COLORS[1], alpha=0.12)
        ax.fill_between(s, combined, 0.0, where=combined > 0, color=LAB_COLORS[4], alpha=0.12)
        ax.set_xlabel("Final underlying price")
        ax.set_ylabel("P&L per share")
        ax.set_title(label.title(), fontsize=11, color=LAB_COLORS[2], pad=8)
        ax.grid(True, alpha=0.16)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout(rect=(0, 0, 1, 0.88) if title else None)
    return fig, axs


def plot_fixed_float_swap_diagram(
    ax=None,
    *,
    title: str = "Fixed-for-floating swap: payer and receiver views",
    figsize: tuple[float, float] = (13.0, 6.3),
):
    fig, ax = _course_canvas(ax, figsize=figsize, title=title, xlim=(0, 14), ylim=(0, 7))

    panel_edge = "#d7e3f5"
    fixed_color = LAB_COLORS[0]
    float_color = LAB_COLORS[1]
    receive_color = "#f0f8ff"
    pay_color = "#fff8f0"
    dealer_color = "#f8fafc"

    panel = mpatches.FancyBboxPatch(
        (0.35, 0.45),
        13.3,
        5.75,
        boxstyle="round,pad=0.22,rounding_size=0.18",
        fc="white",
        ec=panel_edge,
        lw=1.2,
        zorder=0,
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
            ec=LAB_COLORS[8],
            lw=1.1,
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, fontweight="bold", color=LAB_COLORS[2], zorder=3)

    def arrow(start, end, color, rad=0.0):
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            lw=2.1,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=3,
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

    ax.text(2.50, 5.75, "Receiver swap", ha="center", fontsize=11, fontweight="bold", color=LAB_COLORS[2])
    ax.text(11.50, 5.75, "Payer swap", ha="center", fontsize=11, fontweight="bold", color=LAB_COLORS[2])
    ax.plot([7.0, 7.0], [0.8, 5.95], color=panel_edge, lw=1.0, ls="--")
    ax.text(
        7.0,
        0.85,
        r"Synthetic overlay notional is set from target $DV01$; the figure shows cashflow direction, not a market OIS curve.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=LAB_COLORS[2],
    )
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


def quantile_forecast(ax=None, *, title: str = "Quantile forecasts: median and prediction interval"):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor="white") if ax is None else (ax.get_figure(), ax)
    _course_header(fig, title)
    x = np.linspace(-3.2, 3.2, 500)
    y = np.exp(-0.5 * x**2)
    y /= y.max()
    q10, q50, q90 = -1.28, 0.0, 1.28
    ax.fill_between(x, 0, y, where=(x >= q10) & (x <= q90), color=LAB_COLORS[0], alpha=0.22, label="central interval")
    ax.plot(x, y, color=LAB_COLORS[2], lw=2.0)
    for q, label, color in [(q10, "q10", LAB_COLORS[1]), (q50, "q50", LAB_COLORS[3]), (q90, "q90", LAB_COLORS[1])]:
        ax.axvline(q, color=color, lw=1.8, ls="--" if label != "q50" else "-")
        ax.text(q, 1.03, label, ha="center", va="bottom", fontsize=10, color=color)
    ax.annotate("width measures uncertainty", xy=(q90, 0.30), xytext=(1.65, 0.68), color=LAB_COLORS[2], arrowprops={"arrowstyle": "->", "color": LAB_COLORS[2]})
    ax.set_yticks([])
    ax.set_xlabel("future normalized excess return")
    ax.set_ylim(0, 1.16)
    ax.grid(True, axis="x", alpha=0.16)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig, ax


def linear_regularization_comparison(ax=None, *, title: str = "Ridge, Lasso, and Elastic Net"):
    fig, ax = _course_canvas(ax, figsize=(14.2, 5.7), title=title, xlim=(0, 14.2), ylim=(0, 5.5))
    starts = [0.70, 5.00, 9.30]
    specs = [
        ("Ridge", LAB_COLORS[0], "L2 penalty", "smooth shrinkage\nkeeps grouped signals", "circle constraint"),
        ("Lasso", LAB_COLORS[1], "L1 penalty", "sparse solution\ncan zero features", "diamond constraint"),
        ("Elastic Net", LAB_COLORS[3], "L1 + L2 penalty", "sparse but stable\nwith correlated features", "rounded diamond"),
    ]

    theta = np.linspace(0, 2 * np.pi, 240)
    for x, (name, color, penalty, effect, shape_label) in zip(starts, specs, strict=False):
        ax.add_patch(mpatches.FancyBboxPatch((x, 0.70), 3.55, 3.95, boxstyle="round,pad=0.13,rounding_size=0.16", fc="white", ec=LAB_COLORS[8], lw=1.1))
        _round_box(ax, (x + 0.38, 4.05), (1.15, 0.42), name, fc=color, ec=color, text_color="white", fontsize=9.5)
        ax.text(x + 2.20, 4.26, penalty, ha="center", va="center", fontsize=8.6, color=LAB_COLORS[2])
        cx, cy = x + 1.72, 2.48
        ax.plot([cx - 1.08, cx + 1.10], [cy, cy], color=LAB_COLORS[8], lw=1.0)
        ax.plot([cx, cx], [cy - 1.05, cy + 1.08], color=LAB_COLORS[8], lw=1.0)
        ax.text(cx + 1.18, cy - 0.08, "b1", fontsize=7.8, color=LAB_COLORS[2])
        ax.text(cx + 0.08, cy + 1.10, "b2", fontsize=7.8, color=LAB_COLORS[2])

        for scale, lw in [(1.18, 0.9), (0.82, 0.9)]:
            ell_x = cx + 1.18 * scale * np.cos(theta)
            ell_y = cy + 0.62 * scale * np.sin(theta)
            rot_x = cx + 0.86 * (ell_x - cx) - 0.42 * (ell_y - cy)
            rot_y = cy + 0.42 * (ell_x - cx) + 0.86 * (ell_y - cy)
            ax.plot(rot_x, rot_y, color="#98a2b3", lw=lw, alpha=0.55, zorder=1)

        if name == "Ridge":
            pts_x = cx + 0.72 * np.cos(theta)
            pts_y = cy + 0.72 * np.sin(theta)
            opt = (cx + 0.42, cy + 0.37)
        elif name == "Lasso":
            pts_x = [cx, cx + 0.78, cx, cx - 0.78, cx]
            pts_y = [cy + 0.78, cy, cy - 0.78, cy, cy + 0.78]
            opt = (cx + 0.73, cy + 0.02)
        else:
            p = 1.32
            pts_x = cx + 0.82 * np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** (2 / p)
            pts_y = cy + 0.82 * np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** (2 / p)
            opt = (cx + 0.58, cy + 0.18)

        ax.plot(pts_x, pts_y, color=color, lw=2.0, zorder=2)
        ax.scatter([cx + 0.98], [cy + 0.72], s=46, color="#475467", edgecolor="white", linewidth=0.7, zorder=4)
        ax.text(cx + 1.05, cy + 0.76, "unregularized", fontsize=7.4, color=LAB_COLORS[2], ha="left")
        _arrow(ax, (cx + 0.90, cy + 0.65), opt, color=color, lw=1.2, scale=8)
        ax.scatter([opt[0]], [opt[1]], s=62, color=color, edgecolor="#111827", linewidth=0.8, zorder=5)
        ax.text(cx, 1.08, shape_label, ha="center", fontsize=8.0, color=LAB_COLORS[2])
        ax.text(cx, 0.50, effect, ha="center", va="top", fontsize=8.2, color=LAB_COLORS[2], linespacing=1.16)

    _soft_label(ax, (7.10, 5.05), "All three are linear forecasters; the penalty decides how aggressively noisy coefficients are shrunk.")
    fig.tight_layout()
    return fig, ax


def hist_gradient_boosting_diagram(ax=None, *, title: str = "HistGradientBoosting: binned residual learning"):
    fig, ax = _course_canvas(ax, figsize=(16.4, 6.2), title=title, xlim=(0, 16.4), ylim=(0, 6.0))

    def mini_tree(cx, cy, color, label):
        ax.text(cx - 0.88, cy + 0.05, label, ha="right", va="center", fontsize=8, color=LAB_COLORS[2])
        ax.add_patch(mpatches.Circle((cx, cy + 0.28), 0.18, fc=color, ec="#111827", lw=1.0, zorder=4))
        for dx in [-0.44, 0.44]:
            ax.add_patch(mpatches.Circle((cx + dx, cy - 0.22), 0.15, fc="white", ec=color, lw=1.4, zorder=4))
            ax.plot([cx, cx + dx], [cy + 0.10, cy - 0.08], color=color, lw=1.45)

    _round_box(ax, (0.62, 3.35), (1.80, 0.72), "daily\nfeatures", fc=LAB_COLORS[0], ec=LAB_COLORS[0], text_color="white", fontsize=9)
    _arrow(ax, (2.48, 3.71), (2.90, 3.71), color=LAB_COLORS[2], lw=1.55, scale=10)

    bar_x = 3.05
    heights = [0.32, 0.68, 1.14, 0.88, 0.52, 0.28]
    for i, h in enumerate(heights):
        fc = ["#a7d8fa", "#7bc8f6", LAB_COLORS[0], LAB_COLORS[3], "#9ddbd4", "#ccd6e0"][i]
        ax.add_patch(mpatches.Rectangle((bar_x + i * 0.26, 3.10), 0.20, h, fc=fc, ec="#ffffff", lw=0.7, zorder=3))
    _round_box(ax, (2.82, 2.55), (1.82, 0.38), "histogram bins", fc="white", ec=LAB_COLORS[0], fontsize=8.2)
    _arrow(ax, (4.72, 3.50), (5.10, 3.50), color=LAB_COLORS[2], lw=1.55, scale=10)

    grid_x, grid_y = 5.28, 2.82
    for r in range(4):
        for c in range(5):
            val = (r * 3 + c) % 6
            color = ["#d6ebff", "#a7d8fa", LAB_COLORS[0], "#c4ebe7", LAB_COLORS[3], "#edf2f7"][val]
            ax.add_patch(mpatches.Rectangle((grid_x + c * 0.31, grid_y + r * 0.31), 0.28, 0.28, fc=color, ec="white", lw=0.65))
    _round_box(ax, (5.02, 2.22), (1.95, 0.38), "integer bin matrix", fc="white", ec=LAB_COLORS[3], fontsize=8.2)
    _arrow(ax, (7.05, 3.50), (7.45, 3.50), color=LAB_COLORS[2], lw=1.55, scale=10)

    tree_box = mpatches.FancyBboxPatch((7.55, 1.30), 2.20, 3.75, boxstyle="round,pad=0.12,rounding_size=0.13", fc="white", ec=LAB_COLORS[8], lw=1.1, zorder=0)
    ax.add_patch(tree_box)
    mini_tree(8.65, 4.42, LAB_COLORS[3], "tree 1")
    mini_tree(8.65, 3.20, LAB_COLORS[6], "tree 2")
    mini_tree(8.65, 1.98, LAB_COLORS[1], "tree 3")
    _round_box(ax, (8.02, 0.76), (1.28, 0.38), "fit gradients", fc="white", ec=LAB_COLORS[6], fontsize=8.0)
    _arrow(ax, (9.82, 3.50), (10.30, 3.50), color=LAB_COLORS[2], lw=1.55, scale=10)

    eq_y = 3.28
    eq_boxes = [
        (10.45, "F0", LAB_COLORS[2], 0.62),
        (11.25, "+ eta T1", LAB_COLORS[3], 0.92),
        (12.34, "+ eta T2", LAB_COLORS[6], 0.92),
        (13.43, "+ eta T3", LAB_COLORS[1], 0.92),
    ]
    for i, (x, text, color, width) in enumerate(eq_boxes):
        _round_box(ax, (x, eq_y), (width, 0.46), text, fc="white", ec=color, fontsize=7.8)
        if i > 0:
            prev_x, _, _, prev_w = eq_boxes[i - 1]
            _arrow(ax, (prev_x + prev_w + 0.05, eq_y + 0.23), (x - 0.05, eq_y + 0.23), color=LAB_COLORS[2], lw=1.15, scale=8)
    _round_box(ax, (14.80, 3.10), (1.05, 0.74), "forecast\nscore", fc=LAB_COLORS[1], ec=LAB_COLORS[1], text_color="white", fontsize=8.2)
    _arrow(ax, (14.40, eq_y + 0.23), (14.78, 3.47), color=LAB_COLORS[2], lw=1.35, scale=9)

    x_curve = np.linspace(10.55, 15.50, 160)
    y_curve = 1.05 + 0.26 * np.tanh((x_curve - 12.20) * 2.4) + 0.045 * np.sin(x_curve * 6.5)
    ax.plot(x_curve, y_curve, color=LAB_COLORS[2], lw=1.85)
    ax.step([10.55, 11.05, 11.70, 12.45, 13.15, 14.05, 15.50], [0.76, 0.92, 1.14, 1.28, 1.18, 1.34, 1.43], where="post", color=LAB_COLORS[1], lw=1.8)
    ax.text(13.00, 0.42, "leaf values add piecewise corrections", ha="center", fontsize=8.4, color=LAB_COLORS[2])
    _soft_label(ax, (8.20, 5.32), "Histogram binning speeds split search; boosting adds small regularized trees one after another.")
    fig.tight_layout()
    return fig, ax


def mlp_architecture(ax=None, *, title: str = "MLP median-neutral forecast network"):
    fig, ax = _course_canvas(ax, figsize=(14.8, 7.3), title=title, xlim=(0, 14.8), ylim=(0, 7.2))
    edge_light = "#374151"
    edge_dark = LAB_COLORS[0]
    node_edge = "#1f2937"
    layer_x = [1.15, 3.65, 5.95, 8.25, 10.45, 12.75]
    layers = [
        ("Input\nfeatures", ["rank r63", "rel mom", "vol z", "drawdown", "regime", "asset emb"]),
        ("Hidden 1", ["", "", "", "", "", ""]),
        ("Hidden 2", ["", "", "", "", "", ""]),
        ("Hidden 3", ["", "", "", "", ""]),
        ("Hidden 4", ["", "", "", ""]),
        ("Output", ["relative\nscore"]),
    ]
    y_by_count = {
        1: [3.72],
        4: [5.02, 4.28, 3.54, 2.80],
        5: [5.25, 4.55, 3.85, 3.15, 2.45],
        6: [5.42, 4.80, 4.18, 3.56, 2.94, 2.32],
    }
    positions = []
    for x, (layer_name, labels) in zip(layer_x, layers, strict=False):
        ys = y_by_count[len(labels)]
        layer_pos = [(x, y) for y in ys]
        positions.append(layer_pos)
        ax.text(x, 6.25, layer_name, ha="center", va="bottom", fontsize=10, color=LAB_COLORS[2], linespacing=1.15)
        for j, ((px, py), label) in enumerate(zip(layer_pos, labels, strict=False)):
            if len(positions) == 1:
                color = LAB_COLORS[5] if j == len(labels) - 1 else LAB_COLORS[0]
            elif len(positions) == len(layers):
                color = LAB_COLORS[1]
            else:
                color = LAB_COLORS[6] if len(positions) % 2 else LAB_COLORS[3]
            ax.add_patch(mpatches.Circle((px + 0.035, py - 0.035), 0.23, fc="#00000018", ec="none", zorder=2))
            ax.add_patch(mpatches.Circle((px, py), 0.23, fc=color, ec=node_edge, lw=1.35, zorder=4))
            if label:
                if len(positions) == 1:
                    ax.text(px - 0.42, py, label, ha="right", va="center", fontsize=8.4, color=LAB_COLORS[2])
                    ax.add_patch(FancyArrowPatch((px - 0.35, py), (px - 0.23, py), arrowstyle="-|>", mutation_scale=8, lw=1.15, color=edge_light, alpha=1.0))
                else:
                    ax.text(px + 0.62, py, label, ha="left", va="center", fontsize=9, color=LAB_COLORS[2], linespacing=1.1)

    for left, right in zip(positions[:-1], positions[1:], strict=False):
        for x0, y0 in left:
            for x1, y1 in right:
                ax.plot([x0 + 0.23, x1 - 0.23], [y0, y1], color=edge_light, lw=1.20, alpha=0.88, zorder=1)
    highlight = [
        (positions[0][1], positions[1][2]),
        (positions[0][-1], positions[1][4]),
        (positions[1][2], positions[2][1]),
        (positions[2][1], positions[3][2]),
        (positions[3][2], positions[4][1]),
        (positions[4][1], positions[5][0]),
    ]
    for (x0, y0), (x1, y1) in highlight:
        ax.add_patch(FancyArrowPatch((x0 + 0.24, y0), (x1 - 0.24, y1), arrowstyle="-|>", mutation_scale=11, lw=2.05, color=edge_dark, alpha=0.98, zorder=3))

    _round_box(ax, (1.10, 0.92), (2.00, 0.58), "daily date-asset row", fc="white", ec=LAB_COLORS[0], fontsize=8.5)
    _round_box(ax, (4.08, 0.92), (2.06, 0.58), "weighted sum", fc="white", ec=LAB_COLORS[3], fontsize=8.5)
    ax.add_patch(mpatches.Circle((7.08, 1.21), 0.34, fc="white", ec=LAB_COLORS[2], lw=1.3))
    ax.text(7.08, 1.21, "sum", ha="center", va="center", fontsize=8.5, color=LAB_COLORS[2])
    _round_box(ax, (8.05, 0.92), (2.30, 0.58), "SiLU(sum + b)", fc="white", ec=LAB_COLORS[6], fontsize=8.5)
    _round_box(ax, (11.08, 0.92), (1.65, 0.58), "score head", fc="white", ec=LAB_COLORS[1], fontsize=8.5)
    _arrow(ax, (3.15, 1.21), (4.02, 1.21), color=LAB_COLORS[2], scale=10)
    _arrow(ax, (6.18, 1.21), (6.72, 1.21), color=LAB_COLORS[2], scale=10)
    _arrow(ax, (7.43, 1.21), (7.98, 1.21), color=LAB_COLORS[2], scale=10)
    _arrow(ax, (10.40, 1.21), (11.02, 1.21), color=LAB_COLORS[2], scale=10)
    ax.text(7.35, 0.42, "Target: volatility-scaled forward excess return minus same-date cross-sectional median.", ha="center", va="center", fontsize=8.8, color=LAB_COLORS[2])
    ax.text(6.85, 5.92, "all nodes are connected; the blue path illustrates one learned high-impact route", ha="center", fontsize=8.6, color=LAB_COLORS[2], alpha=0.72)
    fig.tight_layout()
    return fig, ax


def activation_loss(ax=None, *, title: str = "Activation and loss functions"):
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), facecolor="white")
    _course_header(fig, title)
    x = np.linspace(-4, 4, 400)
    silu = x / (1.0 + np.exp(-x))
    relu = np.maximum(x, 0.0)
    axes[0].plot(x, silu, color=LAB_COLORS[0], lw=2.1, label="SiLU")
    axes[0].plot(x, relu, color=LAB_COLORS[1], lw=1.6, ls="--", label="ReLU")
    axes[0].axhline(0.0, color=LAB_COLORS[2], lw=0.8)
    axes[0].set_title("Smooth activation", fontsize=11, fontweight="normal", color=LAB_COLORS[2])
    err = np.linspace(-3, 3, 400)
    huber = np.where(np.abs(err) <= 1.0, 0.5 * err**2, np.abs(err) - 0.5)
    pinball = np.maximum(0.1 * err, -0.9 * err)
    axes[1].plot(err, huber, color=LAB_COLORS[3], lw=2.1, label="Huber")
    axes[1].plot(err, pinball, color=LAB_COLORS[4], lw=1.8, label="pinball q10")
    axes[1].set_title("Robust point and quantile losses", fontsize=11, fontweight="normal", color=LAB_COLORS[2])
    for panel in axes:
        panel.grid(True, alpha=0.18)
        panel.legend(frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig, axes


def lstm_architecture(ax=None, *, title: str = "LSTM cell: gates control memory and hidden state"):
    fig, ax = _course_canvas(ax, figsize=(14.2, 7.0), title=title, xlim=(0, 14.2), ylim=(0, 6.9))
    edge = "#111827"
    pale = "#eef5ff"
    cell = mpatches.FancyBboxPatch((2.65, 1.38), 9.70, 4.70, boxstyle="round,pad=0.20,rounding_size=0.18", fc="white", ec=LAB_COLORS[8], lw=1.15)
    ax.add_patch(cell)

    def gate(x, y, label, color):
        _round_box(ax, (x - 0.47, y - 0.30), (0.94, 0.60), label, fc=color, ec=color, text_color="white", fontsize=8.5)

    def op(x, y, label, fc=pale):
        ax.add_patch(mpatches.Circle((x, y), 0.26, fc=fc, ec=LAB_COLORS[2], lw=1.1, zorder=4))
        ax.text(x, y, label, ha="center", va="center", fontsize=9, color=LAB_COLORS[2], zorder=5)

    ax.text(1.10, 5.28, "cell memory\nc(t-1)", ha="right", va="center", fontsize=10, color=LAB_COLORS[2], linespacing=1.15)
    ax.text(1.10, 2.22, "hidden state\nh(t-1)", ha="right", va="center", fontsize=10, color=LAB_COLORS[2], linespacing=1.15)
    ax.text(3.00, 0.88, "input x(t)", ha="center", va="center", fontsize=10, color=LAB_COLORS[2])
    _arrow(ax, (1.25, 5.28), (3.30, 5.28), color=edge, lw=1.25, scale=12)
    _arrow(ax, (2.20, 2.22), (11.45, 2.22), color=edge, lw=1.05, scale=11)
    _arrow(ax, (3.00, 0.96), (3.00, 2.22), color=edge, lw=1.05, scale=11)

    gate(3.65, 3.05, "f(t)\nsigmoid", LAB_COLORS[1])
    gate(5.35, 3.05, "i(t)\nsigmoid", LAB_COLORS[0])
    gate(7.05, 3.05, "g(t)\ntanh", LAB_COLORS[3])
    gate(9.25, 3.05, "o(t)\nsigmoid", LAB_COLORS[6])
    for x in [3.65, 5.35, 7.05, 9.25]:
        _arrow(ax, (x, 2.28), (x, 2.72), color=edge, lw=0.95, scale=9)

    op(3.65, 5.28, "*")
    op(6.10, 4.36, "*")
    op(7.62, 5.28, "+")
    op(9.82, 4.20, "tanh")
    op(10.42, 3.42, "*")
    _arrow(ax, (3.65, 3.38), (3.65, 5.02), color=edge, lw=1.0, scale=10)
    _arrow(ax, (5.35, 3.38), (5.92, 4.18), color=edge, lw=1.0, scale=10)
    _arrow(ax, (7.05, 3.38), (6.24, 4.18), color=edge, lw=1.0, scale=10)
    _arrow(ax, (6.36, 4.36), (7.35, 5.16), color=edge, lw=1.0, scale=10)
    _arrow(ax, (3.92, 5.28), (7.35, 5.28), color=edge, lw=1.0, scale=10)
    _arrow(ax, (7.89, 5.28), (12.95, 5.28), color=edge, lw=1.25, scale=12)
    _arrow(ax, (8.02, 5.16), (9.70, 4.34), color=edge, lw=0.95, scale=9)
    _arrow(ax, (9.25, 3.38), (10.17, 3.42), color=edge, lw=1.0, scale=10)
    _arrow(ax, (9.98, 4.00), (10.28, 3.62), color=edge, lw=1.0, scale=10)
    _arrow(ax, (10.68, 3.42), (12.90, 2.60), color=edge, lw=1.25, scale=12)
    ax.text(13.10, 5.28, "c(t)", ha="left", va="center", fontsize=10.5, fontweight="bold", color=LAB_COLORS[2])
    ax.text(13.05, 2.55, "h(t)", ha="left", va="center", fontsize=10.5, fontweight="bold", color=LAB_COLORS[2])

    ax.text(3.65, 3.78, "forget", ha="center", fontsize=8, color=LAB_COLORS[1])
    ax.text(5.35, 3.78, "input", ha="center", fontsize=8, color=LAB_COLORS[0])
    ax.text(7.05, 3.78, "candidate", ha="center", fontsize=8, color=LAB_COLORS[3])
    ax.text(9.25, 3.78, "output", ha="center", fontsize=8, color=LAB_COLORS[6])
    _soft_label(ax, (6.25, 0.55), "The top lane preserves long-term cell memory; gates decide what to forget, write, and expose.")
    fig.tight_layout()
    return fig, ax


def tcn_receptive_field(ax=None, *, title: str = "TCN daily sequence model: causal dilated receptive field"):
    fig, ax = _course_canvas(ax, figsize=(15.0, 7.0), title=title, xlim=(0, 15.0), ylim=(0, 6.9))
    x0 = 1.05
    step = 1.16
    n_time = 10
    layer_y = [1.05, 2.15, 3.25, 4.35]
    labels = ["daily\nfeatures", "conv d=1", "conv d=2", "conv d=4"]
    dilations = [1, 2, 4]
    node_edge = "#1f2937"
    highlight_nodes = {0: {2, 3, 4, 5, 6, 7, 8, 9}, 1: {3, 5, 7, 9}, 2: {5, 9}, 3: {9}}
    highlight_edges = set()
    frontier = {9}
    for layer_idx in range(3, 0, -1):
        dilation = dilations[layer_idx - 1]
        previous = set()
        for j in frontier:
            previous.add(j)
            if j - dilation >= 0:
                previous.add(j - dilation)
                highlight_edges.add((layer_idx - 1, j - dilation, j))
            highlight_edges.add((layer_idx - 1, j, j))
        frontier = previous

    for layer_idx, y in enumerate(layer_y):
        ax.text(0.25, y, labels[layer_idx], ha="left", va="center", fontsize=9, color=LAB_COLORS[2])
        for j in range(n_time):
            px = x0 + j * step
            active = j in highlight_nodes.get(layer_idx, set())
            if layer_idx == 0:
                fc = LAB_COLORS[0] if active else "#bde3ff"
                ec = node_edge if active else "#5bb8ee"
            elif layer_idx == 3:
                fc = LAB_COLORS[1] if active else "#ffd5c3"
                ec = node_edge if active else "#ff8759"
            else:
                fc = LAB_COLORS[3] if active else "#c7ebe7"
                ec = node_edge if active else "#4fbab0"
            ax.add_patch(mpatches.Circle((px, y), 0.20, fc=fc, ec=ec, lw=1.25, zorder=4))
            if layer_idx == 0:
                ax.text(px, 0.52, f"t-{n_time - 1 - j}" if j < n_time - 1 else "t", ha="center", fontsize=8, color=LAB_COLORS[2])

    for layer_idx, dilation in enumerate(dilations):
        y0, y1 = layer_y[layer_idx], layer_y[layer_idx + 1]
        for j in range(n_time):
            for src in [j, j - dilation]:
                if src < 0:
                    continue
                is_hi = (layer_idx, src, j) in highlight_edges
                color = LAB_COLORS[0] if is_hi else "#374151"
                lw = 2.05 if is_hi else 1.42
                alpha = 0.98 if is_hi else 0.88
                ax.add_patch(FancyArrowPatch((x0 + src * step, y0 + 0.22), (x0 + j * step, y1 - 0.22), arrowstyle="-|>", mutation_scale=8 if is_hi else 7, lw=lw, color=color, alpha=alpha, zorder=2))
        ax.text(12.05, 0.5 * (y0 + y1), f"dilation {dilation}", fontsize=9, color=[LAB_COLORS[3], LAB_COLORS[5], LAB_COLORS[1]][layer_idx], va="center", fontweight="medium")

    _round_box(ax, (12.25, 4.95), (1.65, 0.68), "quantile /\nscore head", fc=LAB_COLORS[1], ec=LAB_COLORS[1], text_color="white", fontsize=8.5)
    _arrow(ax, (x0 + 9 * step + 0.24, layer_y[-1]), (12.20, 5.28), color=LAB_COLORS[0], lw=1.8)
    ax.plot([x0 + 2 * step, x0 + 9 * step], [0.28, 0.28], color=LAB_COLORS[2], lw=1.5)
    ax.plot([x0 + 2 * step, x0 + 2 * step], [0.24, 0.38], color=LAB_COLORS[2], lw=1.5)
    ax.plot([x0 + 9 * step, x0 + 9 * step], [0.24, 0.38], color=LAB_COLORS[2], lw=1.5)
    ax.text(x0 + 5.5 * step, 0.08, "lookback window used for the relative-return forecast", ha="center", fontsize=8.7, color=LAB_COLORS[2])
    _soft_label(ax, (6.55, 6.08), "No recurrence: the model sees a fixed daily window and forecasts median-neutral relative return.")
    fig.tight_layout()
    return fig, ax


def sequence_memory_comparison(ax=None, *, title: str = "How forecasting models remember a sequence"):
    fig, ax = _course_canvas(ax, figsize=(16.4, 7.2), title=title, xlim=(0, 16.4), ylim=(0, 7.0))
    card_w = 3.38
    card_h = 5.35
    starts = [0.55, 4.45, 8.35, 12.25]
    specs = [
        ("MLP", LAB_COLORS[0], "single date row\nno internal state"),
        ("RNN", LAB_COLORS[3], "hidden state\nworking memory"),
        ("LSTM", LAB_COLORS[1], "cell state + hidden\nlonger memory"),
        ("TCN", LAB_COLORS[6], "fixed receptive field\nparallel convolutions"),
    ]
    for x, (name, color, subtitle) in zip(starts, specs, strict=False):
        ax.add_patch(mpatches.FancyBboxPatch((x, 0.78), card_w, card_h, boxstyle="round,pad=0.14,rounding_size=0.20", fc="white", ec=LAB_COLORS[8], lw=1.1))
        _round_box(ax, (x + 0.35, 5.42), (card_w - 0.70, 0.42), name, fc=color, ec=color, text_color="white", fontsize=10)
        ax.text(x + card_w / 2, 1.10, subtitle, ha="center", va="center", fontsize=8.2, color=LAB_COLORS[2], linespacing=1.18)
        xs = np.linspace(x + 0.55, x + card_w - 0.55, 5)
        for i, px in enumerate(xs):
            active = i == 4 or name in {"RNN", "LSTM", "TCN"}
            ax.add_patch(mpatches.Circle((px, 2.08), 0.13, fc=color if active else "#bfc5cf", ec="white", lw=0.7, zorder=3))
            ax.text(px, 1.70, f"t-{4-i}" if i < 4 else "t", ha="center", fontsize=7.2, color=LAB_COLORS[2])
        if name == "MLP":
            _round_box(ax, (x + 1.10, 3.35), (1.05, 0.62), "dense", fc=color, ec=color, text_color="white", fontsize=8)
            _arrow(ax, (xs[-1], 2.24), (x + 1.62, 3.30), color=color, lw=2.0, scale=10)
            _arrow(ax, (x + 1.62, 3.99), (x + 1.62, 4.72), color=color, lw=2.0, scale=10)
        elif name == "RNN":
            _round_box(ax, (x + 1.03, 3.20), (1.18, 0.72), "h(t)", fc=color, ec=color, text_color="white", fontsize=8)
            for px in xs:
                _arrow(ax, (px, 2.22), (x + 1.62, 3.14), color=color, lw=1.30, scale=8)
            ax.add_patch(FancyArrowPatch((x + 2.18, 3.55), (x + 2.18, 3.55), connectionstyle="arc3,rad=1.15", arrowstyle="-|>", mutation_scale=13, lw=1.80, color=color))
            _arrow(ax, (x + 1.62, 3.96), (x + 1.62, 4.72), color=color, lw=2.0, scale=10)
        elif name == "LSTM":
            center = x + card_w / 2
            lane_y = 4.24
            bus_y = 3.05
            gate_y = 3.48
            gate_xs = [x + 0.86, center, x + 2.52]
            ax.plot([x + 0.58, x + 2.86], [lane_y, lane_y], color=color, lw=2.35, zorder=2)
            ax.text(x + 0.62, 4.50, "cell state", ha="left", fontsize=7.5, color=color)
            _round_box(ax, (center - 0.56, 2.44), (1.12, 0.42), "h(t)", fc="white", ec=color, fontsize=8)
            _arrow(ax, (center, 2.90), (center, bus_y - 0.03), color=color, lw=1.35, scale=7)
            ax.plot([gate_xs[0], gate_xs[-1]], [bus_y, bus_y], color=color, lw=1.65, zorder=2)
            for gx, label in zip(gate_xs, ["f", "i", "o"], strict=False):
                _arrow(ax, (gx, bus_y + 0.03), (gx, gate_y - 0.24), color=color, lw=1.25, scale=7)
                ax.add_patch(mpatches.FancyBboxPatch((gx - 0.21, gate_y - 0.18), 0.42, 0.36, boxstyle="round,pad=0.03,rounding_size=0.06", fc=color, ec=color, lw=1.2, zorder=5))
                ax.text(gx, gate_y, label, ha="center", va="center", fontsize=8.0, color="white", zorder=6)
                _arrow(ax, (gx, gate_y + 0.23), (gx, lane_y - 0.10), color=color, lw=1.45, scale=8)
            _arrow(ax, (center, lane_y + 0.02), (center, 4.72), color=color, lw=2.0, scale=10)
        else:
            layer_ys = [2.78, 3.46, 4.14]
            prev_ys = [2.08] + layer_ys[:-1]
            node_r = 0.095
            input_r = 0.13

            def edge_points(src_x, src_y, dst_x, dst_y, src_r, dst_r):
                dx = dst_x - src_x
                dy = dst_y - src_y
                dist = float(np.hypot(dx, dy))
                if dist == 0.0:
                    return (src_x, src_y + src_r), (dst_x, dst_y - dst_r)
                ux = dx / dist
                uy = dy / dist
                return (src_x + ux * src_r, src_y + uy * src_r), (dst_x - ux * dst_r, dst_y - uy * dst_r)

            for prev_y, yy, dilation in zip(prev_ys, layer_ys, [1, 2, 4], strict=False):
                for j in range(len(xs)):
                    for src in [j, j - dilation]:
                        if src < 0:
                            continue
                        start, end = edge_points(xs[src], prev_y, xs[j], yy, input_r if prev_y == 2.08 else node_r, node_r)
                        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7, lw=1.45, color=color, alpha=0.96, zorder=3))
                for px in xs:
                    ax.add_patch(mpatches.Circle((px, yy), node_r, fc="#ead4ff", ec=color, lw=1.25, zorder=4))
                ax.text(x + 0.44, yy, f"d={dilation}", ha="right", va="center", fontsize=7.2, color=color)
            _arrow(ax, (xs[-1], layer_ys[-1] + 0.14), (x + 1.70, 4.72), color=color, lw=2.10, scale=10)
        _round_box(ax, (x + 1.08, 4.76), (1.08, 0.42), "forecast", fc="#f8fafc", ec=color, fontsize=7.8)

    _soft_label(ax, (7.60, 0.34), "The project uses tabular MLPs, LSTM memory, and TCN receptive fields as different ways to summarize recent daily features.")
    fig.tight_layout()
    return fig, ax


def _rl_panel(ax, xy, wh, label, *, fc="#f8fafc", ec="#d7e3f5", fontsize=10, ls="-", lw=1.35):
    x, y = xy
    w, h = wh
    box = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.16,rounding_size=0.18",
        fc=fc,
        ec=ec,
        lw=lw,
        ls=ls,
        zorder=0,
    )
    ax.add_patch(box)
    ax.text(x + 0.22, y + h - 0.32, label, ha="left", va="top", fontsize=fontsize, fontweight="bold", color=LAB_COLORS[2], zorder=3)
    return box


def _rl_chip(ax, xy, text, *, fc="white", ec="#d4dff0", color=LAB_COLORS[2], fontsize=8.2):
    x, y = xy
    w = max(0.70, 0.082 * len(text) + 0.32)
    h = 0.30
    patch = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.05,rounding_size=0.09",
        fc=fc,
        ec=ec,
        lw=0.9,
        zorder=4,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=color, zorder=5)
    return patch


def _network_icon(ax, xy, wh, *, color=LAB_COLORS[0], title="network", layers=(4, 5, 4), node_fc="white", fontsize=8.5):
    x, y = xy
    w, h = wh
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.12,rounding_size=0.18",
            fc="#ffffff",
            ec=color,
            lw=1.35,
            zorder=3,
        )
    )
    xs = np.linspace(x + 0.36, x + w - 0.36, len(layers))
    layer_nodes = []
    for lx, n in zip(xs, layers, strict=False):
        ys = np.linspace(y + 0.44, y + h - 0.58, n)
        layer_nodes.append([(lx, yy) for yy in ys])
    for src_layer, dst_layer in zip(layer_nodes, layer_nodes[1:], strict=False):
        for sx, sy in src_layer:
            for dx, dy in dst_layer:
                ax.plot([sx, dx], [sy, dy], color=color, lw=0.55, alpha=0.45, zorder=3)
    for layer in layer_nodes:
        for px, py in layer:
            ax.add_patch(mpatches.Circle((px, py), 0.085, fc=node_fc, ec=color, lw=1.0, zorder=4))
    ax.text(x + w / 2, y + h - 0.18, title, ha="center", va="top", fontsize=fontsize, fontweight="bold", color=LAB_COLORS[2], zorder=5)


def _portfolio_bars(ax, xy, wh, *, weights=(0.22, 0.18, 0.15, 0.13, 0.12, 0.20), labels=None, title="weights"):
    x, y = xy
    w, h = wh
    colors = [LAB_COLORS[0], LAB_COLORS[3], LAB_COLORS[4], LAB_COLORS[9], LAB_COLORS[7], "#cbd5e1"]
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07,rounding_size=0.10", fc="white", ec="#d4dff0", lw=1.0, zorder=2))
    small = h < 0.70
    title_fs = 7.3 if small else 8.2
    ax.text(x + 0.12, y + h - 0.09, title, ha="left", va="top", fontsize=title_fs, color=LAB_COLORS[2], fontweight="bold", zorder=4)
    left = x + 0.14
    bar_y = y + (0.10 if small else 0.12)
    bar_h = 0.17 if small else 0.22
    bar_w = w - 0.28
    for i, wt in enumerate(weights):
        seg = bar_w * float(wt)
        ax.add_patch(mpatches.Rectangle((left, bar_y), seg, bar_h, fc=colors[i % len(colors)], ec="white", lw=0.7, zorder=3))
        left += seg
    if labels:
        for i, label in enumerate(labels[:4]):
            step = min(0.54, (w - 0.42) / max(1, len(labels[:4]) - 1))
            ax.text(x + 0.14 + step * i, y + h - 0.37, label, ha="left", va="center", fontsize=6.4, color="#475569", zorder=4)


def _sparkline(ax, xy, wh, *, color=LAB_COLORS[3], fill=True):
    x, y = xy
    w, h = wh
    t = np.linspace(0.0, 1.0, 40)
    path = 0.45 + 0.28 * np.sin(2 * np.pi * (t + 0.06)) + 0.18 * t
    xs = x + t * w
    ys = y + path * h
    if fill:
        ax.fill_between(xs, y + 0.08 * h, ys, color=color, alpha=0.10, zorder=2)
    ax.plot(xs, ys, color=color, lw=1.8, zorder=3)
    ax.plot([x, x + w], [y + 0.08 * h, y + 0.08 * h], color="#cbd5e1", lw=0.8, zorder=2)


def _replay_cylinder(ax, xy, wh, text, *, fc="#eefbe4", ec="#58b947"):
    x, y = xy
    w, h = wh
    ax.add_patch(mpatches.Rectangle((x, y + 0.14), w, h - 0.28, fc=fc, ec=ec, lw=1.2, zorder=2))
    ax.add_patch(mpatches.Ellipse((x + w / 2, y + h - 0.14), w, 0.28, fc=fc, ec=ec, lw=1.2, zorder=3))
    ax.add_patch(mpatches.Ellipse((x + w / 2, y + 0.14), w, 0.28, fc=fc, ec=ec, lw=1.2, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color=LAB_COLORS[2], fontweight="bold", zorder=4)


def agent_environment_loop_diagram(ax=None, *, title: str = "RL portfolio loop"):
    fig, ax = _course_canvas(ax, figsize=(14.6, 7.0), title=title, xlim=(0, 14.6), ylim=(0, 6.4))
    _rl_panel(ax, (0.45, 1.22), (3.15, 4.10), "State observation", fc="#f0f8ff", ec="#b7d7f2")
    for yy, label in zip([4.35, 3.88, 3.41, 2.94, 2.47], ["returns", "vol/corr", "forecasts", "priors", "prev weights"], strict=False):
        _rl_chip(ax, (0.78, yy), label, fc="white", ec="#c7def2", color=LAB_COLORS[2])
    _portfolio_bars(ax, (0.92, 1.52), (2.20, 0.78), weights=(0.18, 0.15, 0.14, 0.12, 0.11, 0.30), labels=["SPY", "QQQ", "TLT", "cash"], title="portfolio state")

    _rl_panel(ax, (4.78, 0.72), (4.20, 5.12), "Agent", fc="#f9fbff", ec=LAB_COLORS[8], ls=(0, (5, 3)))
    _network_icon(ax, (5.38, 3.12), (2.85, 1.82), color=LAB_COLORS[0], title="policy network", layers=(4, 5, 4))
    _round_box(ax, (5.38, 1.55), (2.85, 0.74), "active tilts +\nrisky exposure", fc="#f5f3ff", ec=LAB_COLORS[6], text_color=LAB_COLORS[2], fontsize=8.7)
    _rl_chip(ax, (5.30, 2.55), "caps", fc="#fff7ed", ec="#fed7aa", color="#9a3412")
    _rl_chip(ax, (6.28, 2.55), "cash", fc="#f8fafc", ec="#cbd5e1", color="#334155")
    _rl_chip(ax, (7.27, 2.55), "cost", fc="#fff1f2", ec="#fecdd3", color="#9f1239")

    _rl_panel(ax, (10.62, 1.10), (3.45, 4.30), "Portfolio environment", fc="#fff8f0", ec="#ffd6a6")
    _sparkline(ax, (11.05, 3.55), (2.40, 1.02), color=LAB_COLORS[3])
    _portfolio_bars(ax, (11.05, 2.42), (2.35, 0.64), weights=(0.21, 0.18, 0.16, 0.15, 0.10, 0.20), title="daily drift")
    for yy, label in zip([1.92, 1.48], ["trading cost", "drawdown + vol"], strict=False):
        _rl_chip(ax, (11.12, yy), label, fc="white", ec="#fed7aa", color="#7c2d12")

    _arrow(ax, (3.72, 3.52), (4.68, 3.52), color=LAB_COLORS[2], lw=2.1, scale=16)
    ax.text(4.20, 3.82, r"state $s_t$", ha="center", va="bottom", fontsize=8.6, color=LAB_COLORS[2])
    _arrow(ax, (8.36, 2.02), (10.50, 2.96), color=LAB_COLORS[6], lw=2.1, scale=16)
    ax.text(9.52, 3.08, r"action $a_t$: weights", ha="center", va="bottom", fontsize=8.6, color=LAB_COLORS[6])
    _arrow(ax, (12.35, 1.02), (12.35, 0.50), color=LAB_COLORS[1], lw=1.9, scale=13)
    _arrow(ax, (12.35, 0.50), (1.82, 0.50), color=LAB_COLORS[1], lw=1.9, rad=0.0, scale=13)
    _arrow(ax, (1.82, 0.50), (1.82, 1.12), color=LAB_COLORS[1], lw=1.9, scale=13)
    ax.text(7.08, 0.18, r"reward $r_t$, next state $s_{t+1}$, updated portfolio path", ha="center", va="center", fontsize=8.5, color=LAB_COLORS[1])
    ax.text(7.80, 5.98, "Same daily holding-path mechanics for training and backtest", ha="center", va="center", fontsize=8.6, color=LAB_COLORS[2], bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#c7def2", "alpha": 0.9})
    fig.tight_layout()
    return fig, ax


def mdp_diagram(ax=None, *, title: str = "Portfolio allocation as an MDP"):
    fig, ax = _course_canvas(ax, figsize=(13.8, 6.2), title=title, xlim=(0, 13.8), ylim=(0, 5.8))
    _round_box(ax, (0.70, 3.10), (2.25, 0.92), "$S_t$\nfeatures + portfolio", fc=LAB_COLORS[0], ec=LAB_COLORS[0], text_color="white", fontsize=9)
    _round_box(ax, (3.82, 3.10), (2.05, 0.92), "$A_t$\nlong-only weights", fc=LAB_COLORS[6], ec=LAB_COLORS[6], text_color="white", fontsize=9)
    _round_box(ax, (6.88, 2.86), (2.60, 1.40), "Transition\nmarket path + cost", fc="#fff7ed", ec="#fb923c", text_color="#7c2d12", fontsize=9)
    _portfolio_bars(ax, (10.62, 3.30), (1.85, 0.66), weights=(0.20, 0.18, 0.14, 0.13, 0.10, 0.25), title=r"$S_{t+1}$")
    _sparkline(ax, (10.64, 2.36), (1.82, 0.72), color=LAB_COLORS[3])
    _round_box(ax, (6.96, 1.00), (2.44, 0.72), "$R_{t+1}$\nactive reward", fc=LAB_COLORS[1], ec=LAB_COLORS[1], text_color="white", fontsize=8.8)
    _arrow(ax, (3.02, 3.56), (3.76, 3.56), color=LAB_COLORS[2], lw=1.9)
    _arrow(ax, (5.94, 3.56), (6.82, 3.56), color=LAB_COLORS[6], lw=1.9)
    _arrow(ax, (9.56, 3.54), (10.54, 3.54), color=LAB_COLORS[2], lw=1.9)
    _arrow(ax, (8.20, 2.78), (8.20, 1.78), color=LAB_COLORS[1], lw=1.9)
    ax.text(
        3.40,
        3.92,
        r"policy $\pi(a\mid s)$",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=LAB_COLORS[2],
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.86},
        zorder=5,
    )
    ax.text(
        6.38,
        3.92,
        "rebalance",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=LAB_COLORS[6],
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.86},
        zorder=5,
    )
    ax.text(9.04, 2.22, "next observation", ha="left", va="center", fontsize=8.2, color=LAB_COLORS[2])
    ax.text(9.58, 1.36, "log excess, active return, cost, risk", ha="left", va="center", fontsize=8.1, color=LAB_COLORS[1])
    _soft_label(ax, (5.80, 0.42), "State includes previous weights, NAV, drawdown and recent returns.", fontsize=8.2, color=LAB_COLORS[2])
    fig.tight_layout()
    return fig, ax


def mdp_pomdp_diagram(ax=None, *, title: str = "MDP vs POMDP in markets"):
    fig, ax = _course_canvas(ax, figsize=(13.4, 6.2), title=title, xlim=(0, 13.4), ylim=(0, 5.8))
    _rl_panel(ax, (0.48, 0.78), (5.85, 4.45), "MDP view", fc="#f8fafc", ec="#cbd5e1")
    _rl_panel(ax, (7.05, 0.78), (5.85, 4.45), "POMDP market reality", fc="#fbfbff", ec="#c4b5fd")
    _round_box(ax, (1.00, 3.45), (2.02, 0.82), "complete\nstate $S_t$", fc=LAB_COLORS[0], ec=LAB_COLORS[0], text_color="white", fontsize=8.6)
    _round_box(ax, (3.75, 3.45), (1.66, 0.82), "action\nweights", fc=LAB_COLORS[6], ec=LAB_COLORS[6], text_color="white", fontsize=8.6)
    _round_box(ax, (2.15, 1.55), (2.20, 0.82), "next state\nis enough", fc="white", ec=LAB_COLORS[3], fontsize=8.6)
    _arrow(ax, (3.07, 3.86), (3.70, 3.86), color=LAB_COLORS[2])
    _arrow(ax, (4.58, 3.38), (4.24, 2.43), color=LAB_COLORS[6])
    _arrow(ax, (2.34, 2.43), (1.98, 3.38), color=LAB_COLORS[3])

    _round_box(ax, (7.62, 3.62), (2.18, 0.78), "hidden regime\nliquidity, risk", fc=LAB_COLORS[5], ec=LAB_COLORS[5], text_color="white", fontsize=8.3)
    _round_box(ax, (10.25, 3.62), (1.90, 0.78), "observed\nfeatures", fc=LAB_COLORS[0], ec=LAB_COLORS[0], text_color="white", fontsize=8.3)
    _round_box(ax, (10.25, 1.55), (1.90, 0.78), "recurrent\nbelief state", fc="#f5f3ff", ec=LAB_COLORS[6], text_color=LAB_COLORS[2], fontsize=8.3)
    _round_box(ax, (7.72, 1.55), (1.80, 0.78), "action\nweights", fc=LAB_COLORS[6], ec=LAB_COLORS[6], text_color="white", fontsize=8.3)
    _arrow(ax, (9.86, 4.01), (10.20, 4.01), color=LAB_COLORS[5])
    _arrow(ax, (11.20, 3.55), (11.20, 2.38), color=LAB_COLORS[2])
    _arrow(ax, (10.18, 1.94), (9.57, 1.94), color=LAB_COLORS[6])
    _arrow(ax, (8.62, 2.39), (8.62, 3.55), color=LAB_COLORS[6])
    _soft_label(ax, (3.36, 0.98), "Works when the state captures all information needed for transition probabilities.", fontsize=8.2)
    _soft_label(ax, (10.00, 0.98), "Markets are partly observed, so Recurrent PPO carries memory across weeks.", fontsize=8.2, color=LAB_COLORS[6])
    fig.tight_layout()
    return fig, ax


def policy_gradient_diagram(ax=None, *, title: str = "Policy-gradient intuition"):
    fig, ax = _course_canvas(ax, figsize=(13.8, 6.2), title=title, xlim=(0, 13.8), ylim=(0, 5.8))
    _rl_panel(ax, (0.62, 1.02), (3.35, 4.18), "Portfolio state", fc="#f0f8ff", ec="#b7d7f2")
    _rl_chip(ax, (0.95, 4.35), "asset features", fc="white", ec="#c7def2")
    _rl_chip(ax, (0.95, 3.88), "forecast signals", fc="white", ec="#c7def2")
    _rl_chip(ax, (0.95, 3.41), "prior weights", fc="white", ec="#c7def2")
    _portfolio_bars(ax, (1.05, 1.55), (2.15, 0.82), weights=(0.19, 0.16, 0.14, 0.12, 0.10, 0.29), title=r"$s_t$")
    _rl_panel(ax, (4.72, 0.98), (3.16, 4.22), "Stochastic policy", fc="#fbfbff", ec="#c4b5fd")
    _network_icon(ax, (5.18, 3.05), (2.25, 1.48), color=LAB_COLORS[6], title="policy network", layers=(4, 4, 3))
    xs = np.linspace(5.28, 7.32, 70)
    curve = 2.02 + 0.46 * np.exp(-((xs - 6.10) ** 2) / 0.16)
    ax.fill_between(xs, 1.78, curve, color=LAB_COLORS[0], alpha=0.14, zorder=2)
    ax.plot(xs, curve, color=LAB_COLORS[0], lw=1.7, zorder=3)
    ax.text(6.23, 1.48, "sample action\n" + r"$a_t \sim \pi_\theta(\cdot\mid s_t)$", ha="center", va="top", fontsize=8.1, color=LAB_COLORS[2])
    _rl_panel(ax, (9.10, 1.02), (3.95, 4.18), "Daily portfolio path", fc="#fff8f0", ec="#ffd6a6")
    _portfolio_bars(ax, (9.68, 3.92), (2.55, 0.66), weights=(0.23, 0.18, 0.12, 0.10, 0.12, 0.25), title="chosen weights")
    _sparkline(ax, (9.82, 2.50), (2.25, 0.90), color=LAB_COLORS[3])
    _round_box(ax, (9.78, 1.50), (2.28, 0.56), "reward = active return\nminus cost and risk", fc="white", ec=LAB_COLORS[1], text_color=LAB_COLORS[1], fontsize=8.0)
    _arrow(ax, (4.04, 3.18), (4.66, 3.18), color=LAB_COLORS[2], lw=2.0)
    _arrow(ax, (7.94, 3.20), (9.04, 3.20), color=LAB_COLORS[6], lw=2.0)
    ax.text(8.50, 3.52, "action weights", ha="center", va="bottom", fontsize=8.3, color=LAB_COLORS[6])
    _arrow(ax, (9.72, 1.78), (7.38, 2.02), color=LAB_COLORS[1], lw=2.0)
    ax.text(
        8.55,
        1.47,
        "advantage-weighted reward",
        ha="center",
        va="top",
        fontsize=8.1,
        color=LAB_COLORS[1],
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.84},
        zorder=5,
    )
    _soft_label(ax, (6.90, 0.46), r"Update: $\nabla_\theta \log \pi_\theta(a_t\mid s_t)\,\hat A_t$", fontsize=8.5, color=LAB_COLORS[1])
    fig.tight_layout()
    return fig, ax


def actor_critic_diagram(ax=None, *, title: str = "Actor-critic portfolio policy"):
    fig, ax = _course_canvas(ax, figsize=(14.2, 6.6), title=title, xlim=(0, 14.2), ylim=(0, 6.1))
    _rl_panel(ax, (0.55, 1.10), (3.10, 4.30), "Shared observation", fc="#f0f8ff", ec="#b7d7f2")
    _portfolio_bars(ax, (0.98, 4.05), (2.05, 0.64), weights=(0.18, 0.15, 0.14, 0.13, 0.10, 0.30), title="state")
    for yy, label in zip([3.38, 2.93, 2.48], ["forecasts", "risk regime", "previous weights"], strict=False):
        _rl_chip(ax, (0.92, yy), label, fc="white", ec="#c7def2")
    _rl_panel(ax, (4.38, 0.72), (4.25, 4.95), "Agent", fc="#fbfdff", ec=LAB_COLORS[8], ls=(0, (5, 3)))
    _network_icon(ax, (4.88, 3.58), (2.60, 1.30), color=LAB_COLORS[6], title=r"actor: $\pi(a\mid s)$", layers=(4, 4, 3))
    _network_icon(ax, (4.88, 1.35), (2.60, 1.30), color=LAB_COLORS[1], title=r"critic: $V(s)$", layers=(4, 4, 2))
    _round_box(ax, (7.92, 3.82), (2.06, 0.62), "portfolio weights", fc="#f5f3ff", ec=LAB_COLORS[6], text_color=LAB_COLORS[2], fontsize=8.4)
    _round_box(ax, (7.92, 1.58), (2.06, 0.62), r"advantage / $\delta_t$", fc="#fff1f2", ec=LAB_COLORS[1], text_color=LAB_COLORS[1], fontsize=8.4)
    _rl_panel(ax, (10.88, 1.12), (2.70, 4.22), "Environment", fc="#fff8f0", ec="#ffd6a6")
    _sparkline(ax, (11.28, 3.58), (1.84, 0.82), color=LAB_COLORS[3])
    _portfolio_bars(ax, (11.26, 2.42), (1.88, 0.62), weights=(0.23, 0.17, 0.15, 0.10, 0.10, 0.25), title="path")
    _round_box(ax, (11.30, 1.56), (1.80, 0.52), r"$r_t,\ s_{t+1}$", fc="white", ec=LAB_COLORS[1], text_color=LAB_COLORS[1], fontsize=8.2)
    _arrow(ax, (3.72, 3.48), (4.82, 4.23), color=LAB_COLORS[2], lw=1.9)
    _arrow(ax, (3.72, 3.02), (4.82, 2.00), color=LAB_COLORS[2], lw=1.9)
    _arrow(ax, (7.55, 4.23), (7.86, 4.13), color=LAB_COLORS[6], lw=1.8)
    _arrow(ax, (10.04, 4.13), (10.82, 4.02), color=LAB_COLORS[6], lw=1.8)
    _arrow(ax, (11.20, 1.82), (10.04, 1.88), color=LAB_COLORS[1], lw=1.8)
    _arrow(ax, (7.86, 1.88), (7.54, 1.96), color=LAB_COLORS[1], lw=1.8)
    _arrow(ax, (6.10, 2.72), (6.10, 3.50), color=LAB_COLORS[1], lw=1.8)
    ax.text(6.28, 3.10, r"update $\pi_\theta$", ha="left", va="center", fontsize=8.0, color=LAB_COLORS[1])
    _soft_label(ax, (6.05, 0.42), "Actor chooses allocation; critic turns reward and next state into a lower-variance update.", fontsize=8.4)
    fig.tight_layout()
    return fig, ax


def ppo_diagram(ax=None, *, title: str = "PPO clipped update"):
    fig, ax = _course_canvas(ax, figsize=(14.0, 6.3), title=title, xlim=(0, 14.0), ylim=(0, 5.9))
    _rl_panel(ax, (0.52, 1.00), (2.80, 4.18), "Rollout memory", fc="#f8fafc", ec="#cbd5e1")
    for i, label in enumerate([r"$s_t$", r"$a_t$", r"$r_t$", r"$s_{t+1}$"]):
        _rl_chip(ax, (0.95, 4.35 - 0.48 * i), label, fc="white", ec="#d7e3f5", fontsize=8.0)
    _portfolio_bars(ax, (0.88, 1.62), (1.94, 0.68), weights=(0.20, 0.15, 0.14, 0.11, 0.10, 0.30), title="old weights")
    _rl_panel(ax, (4.05, 1.00), (3.20, 4.18), "Actor-critic PPO", fc="#fbfbff", ec="#c4b5fd")
    _network_icon(ax, (4.55, 3.35), (2.10, 1.22), color=LAB_COLORS[6], title="actor", layers=(4, 4, 3))
    _network_icon(ax, (4.55, 1.58), (2.10, 1.22), color=LAB_COLORS[1], title="critic", layers=(4, 4, 2))
    _rl_panel(ax, (8.25, 1.00), (4.90, 4.18), "Clipped objective", fc="#fffdf7", ec="#fde68a")
    _round_box(ax, (8.72, 3.95), (1.95, 0.58), "ratio\n" + r"$\pi_{\mathrm{new}}/\pi_{\mathrm{old}}$", fc="white", ec=LAB_COLORS[6], text_color=LAB_COLORS[6], fontsize=8.0)
    _round_box(ax, (11.10, 3.95), (1.38, 0.58), "clip\n" + r"$1 \pm \epsilon$", fc="white", ec=LAB_COLORS[1], text_color=LAB_COLORS[1], fontsize=8.0)
    x = np.linspace(8.90, 12.45, 80)
    y = 2.08 + 0.38 * np.tanh(1.8 * (x - 10.45))
    ax.plot(x, y, color=LAB_COLORS[2], lw=2.0, zorder=3)
    ax.fill_between([9.85, 11.25], [1.62, 1.62], [2.62, 2.62], color=LAB_COLORS[0], alpha=0.13, zorder=2)
    ax.text(10.55, 2.82, "trusted update band", ha="center", fontsize=8.2, color=LAB_COLORS[2], zorder=4)
    _round_box(ax, (9.42, 1.06), (2.70, 0.48), "SGD update with clipped objective", fc="white", ec="#facc15", text_color="#854d0e", fontsize=8.0)
    _arrow(ax, (3.38, 3.42), (4.00, 3.98), color=LAB_COLORS[2], lw=1.8)
    _arrow(ax, (6.72, 3.96), (8.66, 4.24), color=LAB_COLORS[6], lw=1.8)
    _arrow(ax, (10.70, 4.24), (11.04, 4.24), color=LAB_COLORS[2], lw=1.6)
    _arrow(ax, (9.34, 1.30), (6.72, 1.88), color=LAB_COLORS[1], lw=1.8)
    ax.text(
        7.88,
        1.02,
        "parameter update",
        ha="center",
        va="center",
        fontsize=8.1,
        color=LAB_COLORS[1],
        bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.82},
        zorder=5,
    )
    _soft_label(ax, (6.95, 0.48), "PPO improves the policy while limiting destructive probability-ratio jumps.", fontsize=8.2)
    fig.tight_layout()
    return fig, ax


def sac_diagram(ax=None, *, title: str = "SAC entropy-regularized control"):
    fig, ax = _course_canvas(ax, figsize=(14.2, 6.4), title=title, xlim=(0, 14.2), ylim=(0, 6.0))
    _rl_panel(ax, (0.55, 0.92), (3.08, 4.38), "Historical transitions", fc="#f8fafc", ec="#cbd5e1")
    _sparkline(ax, (1.02, 3.78), (1.95, 0.72), color=LAB_COLORS[3])
    _portfolio_bars(ax, (1.00, 2.72), (1.98, 0.64), weights=(0.20, 0.16, 0.15, 0.11, 0.08, 0.30), title="action")
    _replay_cylinder(ax, (1.05, 1.30), (1.90, 0.86), "replay\nbuffer")

    _rl_panel(ax, (4.35, 0.92), (3.05, 4.38), "Actor", fc="#fff7ed", ec="#fdba74")
    _network_icon(ax, (4.90, 3.22), (1.95, 1.30), color=LAB_COLORS[6], title="policy", layers=(4, 4, 3))
    _round_box(ax, (4.94, 2.08), (1.88, 0.62), "sample weights\n+ entropy", fc="white", ec=LAB_COLORS[6], text_color=LAB_COLORS[6], fontsize=8.0)

    _rl_panel(ax, (8.18, 0.92), (5.15, 4.38), "Twin critics", fc="#f8fbff", ec="#bfdbfe")
    _network_icon(ax, (8.70, 3.28), (1.74, 1.16), color=LAB_COLORS[1], title=r"$Q_1$", layers=(4, 3, 1), fontsize=8.0)
    _network_icon(ax, (10.92, 3.28), (1.74, 1.16), color=LAB_COLORS[1], title=r"$Q_2$", layers=(4, 3, 1), fontsize=8.0)
    _round_box(ax, (8.88, 1.34), (3.62, 0.64), r"target: $r_t + \gamma[\min(Q_1,Q_2)-\alpha\log\pi]$", fc="white", ec=LAB_COLORS[0], text_color=LAB_COLORS[2], fontsize=7.8)

    _arrow(ax, (3.05, 1.74), (4.28, 1.98), color=LAB_COLORS[2], lw=1.8)
    ax.text(3.67, 1.50, "sample tuples", ha="center", va="top", fontsize=8.0, color=LAB_COLORS[2])
    _arrow(ax, (6.88, 2.39), (8.80, 2.00), color=LAB_COLORS[6], lw=1.8)
    ax.text(7.82, 2.58, "action + entropy", ha="center", va="bottom", fontsize=8.0, color=LAB_COLORS[6])
    _arrow(ax, (10.72, 3.22), (10.72, 2.06), color=LAB_COLORS[0], lw=1.6)
    _arrow(ax, (8.82, 1.56), (6.88, 2.18), color=LAB_COLORS[1], lw=1.8)
    ax.text(
        7.82,
        1.36,
        "critic signal",
        ha="center",
        va="top",
        fontsize=8.0,
        color=LAB_COLORS[1],
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.82},
        zorder=5,
    )
    _arrow(ax, (10.70, 1.28), (10.70, 0.96), color=LAB_COLORS[0], lw=1.4)
    ax.text(10.70, 0.80, "soft target update", ha="center", va="top", fontsize=8.0, color=LAB_COLORS[0])
    _soft_label(ax, (7.10, 0.42), r"SAC can run as a contextual allocator with $\gamma=0$ for safer off-policy updates.", fontsize=8.2)
    fig.tight_layout()
    return fig, ax


plot_bsm_comp_graph = plot_computation_dag

__all__ = [
    "actor_critic_diagram",
    "agent_environment_loop_diagram",
    "agglomerative_tree",
    "activation_loss",
    "bayesian_mixture",
    "decision_tree_split",
    "ensemble_bagging",
    "gmm_mixture",
    "hidden_markov_model",
    "hist_gradient_boosting_diagram",
    "kmeans_geometry",
    "knn_neighbors",
    "lda_projection",
    "linear_regularization_comparison",
    "lstm_architecture",
    "logistic_boundary",
    "markov_chain",
    "mdp_diagram",
    "mdp_pomdp_diagram",
    "ml_pipeline",
    "mlp_architecture",
    "plot_bsm_comp_graph",
    "plot_computation_dag",
    "plot_fixed_float_swap_diagram",
    "plot_straddle_payoff",
    "policy_gradient_diagram",
    "ppo_diagram",
    "quantile_forecast",
    "sac_diagram",
    "sequence_memory_comparison",
    "svm_margin",
    "tcn_receptive_field",
    "tree_structure",
    "overlay_payoffs",
    "unsupervised_supervised",
    "walkforward_split",
]
