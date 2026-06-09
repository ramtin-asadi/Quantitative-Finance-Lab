from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.plotting.curves import LAB_COLORS, set_plot_style


def _ax(ax=None):
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    return ax


def _history_frame(history) -> pd.DataFrame:
    if isinstance(history, pd.DataFrame):
        return history.copy()
    if hasattr(history, "history"):
        return pd.DataFrame(history.history)
    return pd.DataFrame(history)


def training_reward_curve(ax, history, *, label: str | None = None, title: str | None = None):
    set_plot_style()
    ax = _ax(ax)
    h = _history_frame(history)
    if h.empty:
        ax.text(0.5, 0.5, "No training history", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    x = h["epoch"] if "epoch" in h.columns else np.arange(1, len(h) + 1)
    y_col = "train_reward" if "train_reward" in h.columns else h.select_dtypes("number").columns[0]
    ax.plot(x, h[y_col], lw=1.8, label=label or y_col, color=LAB_COLORS[0])
    if "validation_reward" in h.columns:
        ax.plot(x, h["validation_reward"], lw=1.5, ls="--", label="validation", color=LAB_COLORS[1])
    ax.set_title(title or "Training reward")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    return ax


def ppo_loss_curves(ax, history, *, title: str | None = None):
    set_plot_style()
    ax = _ax(ax)
    h = _history_frame(history)
    x = h["epoch"] if "epoch" in h.columns else np.arange(1, len(h) + 1)
    plotted = False
    for col, color in [("policy_loss", LAB_COLORS[3]), ("value_loss", LAB_COLORS[1])]:
        if col in h.columns:
            ax.plot(x, h[col], label=col.replace("_", " "), color=color, lw=1.6)
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No PPO losses", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title or "PPO losses")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(loc="best", fontsize=8)
    return ax


def sac_loss_curves(ax, history, *, title: str | None = None):
    set_plot_style()
    ax = _ax(ax)
    h = _history_frame(history)
    x = h["epoch"] if "epoch" in h.columns else np.arange(1, len(h) + 1)
    plotted = False
    for col, color in [("critic_loss", LAB_COLORS[1]), ("actor_loss", LAB_COLORS[3])]:
        if col in h.columns:
            ax.plot(x, h[col], label=col.replace("_", " "), color=color, lw=1.6)
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No SAC losses", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title or "SAC losses")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(loc="best", fontsize=8)
    return ax


def policy_entropy_curve(ax, history, *, title: str | None = None):
    set_plot_style()
    ax = _ax(ax)
    h = _history_frame(history)
    x = h["epoch"] if "epoch" in h.columns else np.arange(1, len(h) + 1)
    if "entropy" in h.columns:
        ax.plot(x, h["entropy"], color=LAB_COLORS[4], lw=1.7)
    else:
        ax.text(0.5, 0.5, "No entropy history", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title or "Policy entropy")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.25)
    return ax


def validation_score_curve(ax, validation_table, *, title: str | None = None):
    set_plot_style()
    ax = _ax(ax)
    tbl = pd.DataFrame(validation_table)
    if tbl.empty:
        ax.text(0.5, 0.5, "No validation table", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    col = "sharpe" if "sharpe" in tbl.columns else ("total_reward" if "total_reward" in tbl.columns else tbl.select_dtypes("number").columns[0])
    tbl[col].plot(kind="bar", ax=ax, color=LAB_COLORS[: len(tbl)])
    ax.set_title(title or f"Validation {col}")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    return ax


def q_value_curve(ax, history, *, title: str | None = None):
    set_plot_style()
    ax = _ax(ax)
    h = _history_frame(history)
    x = h["epoch"] if "epoch" in h.columns else np.arange(1, len(h) + 1)
    if "q_value" in h.columns:
        ax.plot(x, h["q_value"], color=LAB_COLORS[2], lw=1.7)
    else:
        ax.text(0.5, 0.5, "No Q-value history", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title or "Q values")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.25)
    return ax


__all__ = [
    "policy_entropy_curve",
    "ppo_loss_curves",
    "q_value_curve",
    "sac_loss_curves",
    "training_reward_curve",
    "validation_score_curve",
]
