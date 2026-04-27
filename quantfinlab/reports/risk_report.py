from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantfinlab.core import RiskReportArtifacts
from quantfinlab.risk.capm import capm_table
from quantfinlab.risk.contributions import attribution_tables
from quantfinlab.risk.correlation import corr_matrix
from quantfinlab.risk.distribution import tail_shape_table
from quantfinlab.risk.drawdown import drawdown_episodes_table, drawdown_summary_table
from quantfinlab.risk.performance import performance_table
from quantfinlab.risk.stress import stress_table
from quantfinlab.risk.utils import (
    DEFAULT_ANNUALIZATION,
    VAR_BACKTEST_METHODS,
    _coerce_objects,
    _normalize_var_methods,
)
from quantfinlab.risk.var import var_es_table
from quantfinlab.risk.var_backtesting import (
    best_var_methods,
    var_backtest_details,
    var_backtest_table,
)

try:  # optional
    from IPython.display import display as ipy_display
except Exception:  # pragma: no cover
    ipy_display = None

def _display_table(df: pd.DataFrame, *, round_digits: int = 4) -> None:
    show = df.round(int(round_digits)) if isinstance(df, pd.DataFrame) else df
    if ipy_display is not None:
        ipy_display(show)
    else:  # pragma: no cover
        print(show)

def executive_bullets(
    *,
    perf_tbl: pd.DataFrame | None = None,
    dd_tbl: pd.DataFrame | None = None,
    var_tbl: pd.DataFrame | None = None,
    capm_tbl: pd.DataFrame | None = None,
    var_bt_tbl: pd.DataFrame | None = None,
) -> list[str]:
    bullets: list[str] = []
    if perf_tbl is not None and not perf_tbl.empty and "sharpe" in perf_tbl.columns:
        best = str(perf_tbl["sharpe"].idxmax())
        bullets.append(f"{best} has the highest realized Sharpe ratio.")
    if dd_tbl is not None and not dd_tbl.empty and "max_dd" in dd_tbl.columns:
        best_dd = str(dd_tbl["max_dd"].idxmax())
        bullets.append(f"Least severe maximum drawdown: {best_dd} ({dd_tbl.loc[best_dd, 'max_dd']:.2%}).")
    if var_tbl is not None and not var_tbl.empty:
        es_cols = [c for c in var_tbl.columns if c.startswith("hist_es")]
        if es_cols:
            c = es_cols[0]
            low_tail = str(var_tbl[c].idxmin())
            bullets.append(f"Lower historical ES tail risk: {low_tail} ({var_tbl.loc[low_tail, c]:.2%}).")
    if capm_tbl is not None and not capm_tbl.empty and "beta" in capm_tbl.columns:
        hi_beta = str(capm_tbl["beta"].idxmax())
        lo_beta = str(capm_tbl["beta"].idxmin())
        bullets.append(
            f"Highest market beta: {hi_beta} ({capm_tbl.loc[hi_beta, 'beta']:.2f}); "
            f"lowest: {lo_beta} ({capm_tbl.loc[lo_beta, 'beta']:.2f})."
        )
    if var_bt_tbl is not None and not var_bt_tbl.empty:
        bt_eval = var_bt_tbl
        if isinstance(bt_eval.index, pd.MultiIndex) and bt_eval.index.nlevels >= 2:
            if "is_best" in bt_eval.columns and bt_eval["is_best"].any():
                bt_eval = bt_eval[bt_eval["is_best"]].copy()
            elif "accuracy_rank" in bt_eval.columns:
                bt_eval = (
                    bt_eval.sort_values(["accuracy_rank", "abs_coverage_error", "quantile_loss"])
                    .groupby(level=0)
                    .head(1)
                )
        issue = []
        if "kupiec_p" in bt_eval.columns:
            issue.extend([str(i[0] if isinstance(i, tuple) else i) for i, v in bt_eval["kupiec_p"].items() if np.isfinite(v) and v < 0.05])
        if "christoffersen_p" in bt_eval.columns:
            issue.extend(
                [str(i[0] if isinstance(i, tuple) else i) for i, v in bt_eval["christoffersen_p"].items() if np.isfinite(v) and v < 0.05]
            )
        issue = sorted(set(issue))
        if issue:
            bullets.append("Potential VaR model instability (p<0.05): " + ", ".join(issue) + ".")
    return bullets

def risk_report(
    *,
    objects: Mapping[str, Any] | pd.DataFrame,
    market_ret: pd.Series | Sequence[float] | np.ndarray | None = None,
    rf_daily: float = 0.0,
    portfolios: Mapping[str, Any] | None = None,
    include: Mapping[str, bool] | None = None,
    var_settings: Mapping[str, Any] | None = None,
    backtest_settings: Mapping[str, Any] | None = None,
    rolling_settings: Mapping[str, Any] | None = None,
    stress_settings: Mapping[str, Any] | None = None,
    attribution_settings: Mapping[str, Any] | None = None,
    layout: Mapping[str, Any] | None = None,
    output: Mapping[str, Any] | None = None,
) -> RiskReportArtifacts:
    from quantfinlab.plotting import risk as pl

    raw_obj = _coerce_objects(objects)
    raw_names = list(raw_obj.keys())

    include_cfg = {
        "performance_tables": True,
        "shape_tables": True,
        "drawdowns": True,
        "drawdown_episodes": True,
        "var_es": True,
        "var_backtest": True,
        "stress": True,
        "capm": True,
        "rolling_beta": True,
        "correlation": True,
        "attribution": True,
        "exec_bullets": True,
    }
    if include:
        include_cfg.update({str(k): bool(v) for k, v in include.items()})

    var_cfg = {"alpha": 0.05, "methods": ["hist", "cf", "fhs"], "lookback": 252}
    if var_settings:
        var_cfg.update(dict(var_settings))
    bt_cfg = {
        "alpha": 0.05,
        "methods": list(VAR_BACKTEST_METHODS),
        "lookback": 252,
        # Method used in VaR breach plots. "best" chooses the highest-ranked method per object.
        "plot_method": "best",
    }
    if backtest_settings:
        bt_cfg.update(dict(backtest_settings))
    bt_methods = _normalize_var_methods(
        method=(None if "methods" in bt_cfg and bt_cfg.get("methods") is not None else bt_cfg.get("method")),
        methods=bt_cfg.get("methods"),
    )
    bt_plot_method = str(bt_cfg.get("plot_method", "best")).strip().lower()
    roll_cfg = {"vol_windows": [20, 60, 252], "beta_windows": [126, 252]}
    if rolling_settings:
        roll_cfg.update(dict(rolling_settings))
    stress_cfg = {
        "windows": {
            "2018_q4": ("2018-10-01", "2018-12-31"),
            "2020_covid": ("2020-02-20", "2020-04-30"),
            "2022_inflation": ("2022-01-03", "2022-10-31"),
        }
    }
    if stress_settings:
        stress_cfg.update(dict(stress_settings))
    attr_cfg = {"es_alpha": 0.05, "top_k": 10}
    if attribution_settings:
        attr_cfg.update(dict(attribution_settings))
    layout_cfg = {"ncols": 2, "sharex": True, "sharey": True}
    if layout:
        layout_cfg.update(dict(layout))
    output_cfg = {
        "round_tables": 4,
        "print_exec_bullets": True,
        "display_tables": True,
        "show_figures": True,
        # Optional controls to display only a subset of computed tables.
        "display_table_keys": None,
        "hide_table_keys": [],
        "short_labels": True,
        "label_max_len": 18,
        "label_map": {},
    }
    if output:
        output_cfg.update(dict(output))

    if bool(output_cfg.get("short_labels", True)):
        label_map = pl.short_label_map(
            raw_names,
            max_len=int(output_cfg.get("label_max_len", 18)),
            overrides=output_cfg.get("label_map", {}),
        )
    else:
        label_map = {name: name for name in raw_names}

    obj = {label_map[name]: value for name, value in raw_obj.items()}
    names = list(obj.keys())
    if portfolios:
        portfolios = {label_map.get(str(name), pl.shorten_label(name, max_len=int(output_cfg.get("label_max_len", 18)))): spec for name, spec in portfolios.items()}

    tables: dict[str, pd.DataFrame] = {}
    series: dict[str, Any] = {}
    figures: dict[str, list[Any]] = {}
    texts: dict[str, list[str]] = {}

    if include_cfg["performance_tables"]:
        tables["performance"] = performance_table(obj, rf_daily=rf_daily, annualization=DEFAULT_ANNUALIZATION)
    if include_cfg["shape_tables"]:
        tables["shape"] = tail_shape_table(obj)
    if include_cfg["drawdowns"]:
        tables["drawdown_summary"] = drawdown_summary_table(obj)
    if include_cfg["drawdown_episodes"]:
        tables["drawdown_episodes"] = drawdown_episodes_table(obj, top_n=1)
    if include_cfg["var_es"]:
        tables["var_es"] = var_es_table(obj, alpha=var_cfg["alpha"], methods=var_cfg["methods"])
    if include_cfg["var_backtest"]:
        tables["var_backtest"] = var_backtest_table(
            obj,
            alpha=bt_cfg["alpha"],
            methods=bt_methods,
            lookback=int(bt_cfg["lookback"]),
        )
        if len(bt_methods) == 1:
            series["var_backtest_detail"] = var_backtest_details(
                obj,
                alpha=bt_cfg["alpha"],
                method=bt_methods[0],
                lookback=int(bt_cfg["lookback"]),
            )
        else:
            series["var_backtest_detail"] = {
                m: var_backtest_details(
                    obj,
                    alpha=bt_cfg["alpha"],
                    method=m,
                    lookback=int(bt_cfg["lookback"]),
                )
                for m in bt_methods
            }
        series["var_backtest_best_method"] = best_var_methods(tables["var_backtest"])
    if include_cfg["stress"]:
        stress_worst_only = bool(stress_cfg.get("worst_only", True))
        stress_worst_by = str(stress_cfg.get("worst_by", "cum_return"))
        tables["stress"] = stress_table(
            obj,
            windows=stress_cfg["windows"],
            worst_only=stress_worst_only,
            worst_by=stress_worst_by,
        )
        # Keep full window-level stress for stress subplot section.
        series["stress_full"] = stress_table(
            obj,
            windows=stress_cfg["windows"],
            worst_only=False,
            worst_by=stress_worst_by,
        )
    if include_cfg["capm"] and market_ret is not None:
        capm_tbl, capm_roll = capm_table(
            obj,
            market_ret=market_ret,
            rf_daily=rf_daily,
            rolling=roll_cfg.get("beta_windows", [126, 252]),
        )
        tables["capm"] = capm_tbl
        series["capm_roll"] = capm_roll
    if include_cfg["correlation"]:
        tables["corr"] = corr_matrix(obj)
    if include_cfg["attribution"] and portfolios:
        vtbl, etbl, otbl = attribution_tables(
            portfolios,
            es_alpha=float(attr_cfg["es_alpha"]),
            top_k=int(attr_cfg["top_k"]),
        )
        tables["attribution_vol"] = vtbl
        tables["attribution_es"] = etbl
        tables["attribution_overlap"] = otbl

    if include_cfg["exec_bullets"]:
        texts["exec_bullets"] = executive_bullets(
            perf_tbl=tables.get("performance"),
            dd_tbl=tables.get("drawdown_summary"),
            var_tbl=tables.get("var_es"),
            capm_tbl=tables.get("capm"),
            var_bt_tbl=tables.get("var_backtest"),
        )

    if bool(output_cfg["display_tables"]):
        show_keys_raw = output_cfg.get("display_table_keys")
        hide_keys = {str(k) for k in output_cfg.get("hide_table_keys", [])}
        if show_keys_raw is None:
            show_keys = list(tables.keys())
        else:
            show_keys = [str(k) for k in show_keys_raw if str(k) in tables]
        for key in show_keys:
            if key in hide_keys:
                continue
            _display_table(tables[key], round_digits=int(output_cfg["round_tables"]))
        if bool(output_cfg["print_exec_bullets"]) and texts.get("exec_bullets"):
            for b in texts["exec_bullets"]:
                print(f"- {b}")

    ncols = max(int(layout_cfg["ncols"]), 1)

    def _grid_size(
        n_items: int,
        *,
        ncols_use: int,
        panel_w: float = 3.8,
        panel_h: float = 3.4,
    ) -> tuple[float, float]:
        cols = max(min(int(ncols_use), max(int(n_items), 1)), 1)
        rows = math.ceil(max(int(n_items), 1) / cols)
        width = float(np.clip(panel_w * cols, 8.0, 28.0))
        height = float(np.clip(panel_h * rows, 5.0, 30.0))
        return width, height

    if include_cfg["drawdowns"]:
        fig, ax = plt.subplots(2, 1, figsize=(10.5, 6.5), sharex=True)
        pl.plot_nav_compare(ax[0], obj)
        pl.plot_drawdown_compare_objects(ax[1], obj)
        plt.tight_layout()
        figures.setdefault("drawdown_compare", []).append(fig)
        if bool(output_cfg["show_figures"]):
            plt.show()

    if include_cfg["drawdowns"]:
        fig, axes = pl.auto_grid(
            len(names),
            ncols=ncols,
            figsize=_grid_size(len(names), ncols_use=ncols),
            sharex=bool(layout_cfg["sharex"]),
            sharey=bool(layout_cfg["sharey"]),
        )
        for a, nm in zip(axes, names, strict=False):
            pl.plot_rolling_vol(
                a,
                obj[nm],
                windows=roll_cfg.get("vol_windows", [20, 60, 252]),
                annualization=DEFAULT_ANNUALIZATION,
                name=nm,
            )
        pl.turn_off_unused_axes(axes, used=len(names))
        plt.tight_layout()
        figures["rolling_vol"] = [fig]
        if bool(output_cfg["show_figures"]):
            plt.show()

    if include_cfg["var_backtest"]:
        fig, axes = pl.auto_grid(
            len(names),
            ncols=ncols,
            figsize=_grid_size(len(names), ncols_use=ncols),
            sharex=bool(layout_cfg["sharex"]),
            sharey=bool(layout_cfg["sharey"]),
        )
        best_method_map = series.get("var_backtest_best_method", {}) if isinstance(series.get("var_backtest_best_method", {}), Mapping) else {}
        for a, nm in zip(axes, names, strict=False):
            chosen_method = bt_plot_method
            if bt_plot_method == "best":
                chosen_method = str(best_method_map.get(nm, bt_methods[0]))
            pl.plot_var_backtest(
                a,
                obj[nm],
                alpha=float(bt_cfg["alpha"]),
                lookback=int(bt_cfg["lookback"]),
                method=chosen_method,
                methods=bt_methods,
                name=nm,
            )
        pl.turn_off_unused_axes(axes, used=len(names))
        plt.tight_layout()
        figures["var_backtest"] = [fig]
        if bool(output_cfg["show_figures"]):
            plt.show()

    if include_cfg["stress"] and "stress_full" in series and isinstance(series["stress_full"], pd.DataFrame) and not series["stress_full"].empty:
        stress_plot_tbl = series["stress_full"]
        windows = [str(w) for w in pd.Index(stress_plot_tbl.index).unique()]
        stress_ncols = min(ncols, max(1, len(windows)))
        fig, axes = pl.auto_grid(
            len(windows),
            ncols=stress_ncols,
            figsize=_grid_size(len(windows), ncols_use=stress_ncols, panel_w=4.0, panel_h=2.9),
            sharex=False,
            sharey=bool(layout_cfg["sharey"]),
        )
        for a, wn in zip(axes, windows, strict=False):
            pl.plot_stress_bar(a, stress_plot_tbl, window=wn, metric="cum_return")
        pl.turn_off_unused_axes(axes, used=len(windows))
        plt.tight_layout()
        figures["stress"] = [fig]
        if bool(output_cfg["show_figures"]):
            plt.show()

    if include_cfg["capm"] and market_ret is not None:
        fig, axes = pl.auto_grid(
            len(names),
            ncols=ncols,
            figsize=_grid_size(len(names), ncols_use=ncols),
            sharex=bool(layout_cfg["sharex"]),
            sharey=bool(layout_cfg["sharey"]),
        )
        color_map = pl.make_color_map(names, pl.LAB_COLORS)
        for a, nm in zip(axes, names, strict=False):
            pl.plot_capm_scatter(
                a,
                obj[nm],
                market_ret,
                rf_daily=rf_daily,
                name=nm,
                color=color_map.get(nm),
            )
        pl.turn_off_unused_axes(axes, used=len(names))
        plt.tight_layout()
        figures["capm_scatter"] = [fig]
        if bool(output_cfg["show_figures"]):
            plt.show()

    if include_cfg["rolling_beta"] and "capm_roll" in series:
        beta_windows = [int(v) for v in roll_cfg.get("beta_windows", [126, 252]) if int(v) > 1]
        if beta_windows:
            fig, axes = plt.subplots(len(beta_windows), 1, figsize=(11, 3.2 * len(beta_windows)), sharex=True)
            axes_arr = np.asarray([axes]) if isinstance(axes, plt.Axes) else np.asarray(axes).reshape(-1)
            for a, w in zip(axes_arr, beta_windows, strict=False):
                pl.plot_rolling_beta_compare(a, series["capm_roll"], window=int(w), metric="beta")
            plt.tight_layout()
            figures.setdefault("rolling_beta", []).append(fig)
            if bool(output_cfg["show_figures"]):
                plt.show()

    if include_cfg["correlation"] and "corr" in tables:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
        pl.plot_corr_heatmap(ax, tables["corr"])
        plt.tight_layout()
        figures.setdefault("correlation", []).append(fig)
        if bool(output_cfg["show_figures"]):
            plt.show()

    if include_cfg["attribution"] and portfolios and "attribution_vol" in tables and "attribution_es" in tables:
        pnames = list(tables["attribution_vol"].index)
        top_k = int(attr_cfg["top_k"])
        fig_vol, axes_vol = pl.auto_grid(
            len(pnames),
            ncols=ncols,
            figsize=_grid_size(len(pnames), ncols_use=ncols),
            sharex=False,
            sharey=False,
        )
        for a, pname in zip(axes_vol, pnames, strict=False):
            pl.plot_top_contrib(
                a,
                tables["attribution_vol"].loc[pname],
                title=f"Vol RC - {pname}",
                k=top_k,
            )
        pl.turn_off_unused_axes(axes_vol, used=len(pnames))
        plt.tight_layout()

        fig_es, axes_es = pl.auto_grid(
            len(pnames),
            ncols=ncols,
            figsize=_grid_size(len(pnames), ncols_use=ncols),
            sharex=False,
            sharey=False,
        )
        for a, pname in zip(axes_es, pnames, strict=False):
            pl.plot_top_contrib(
                a,
                tables["attribution_es"].loc[pname],
                title=f"ES RC - {pname}",
                k=top_k,
            )
        pl.turn_off_unused_axes(axes_es, used=len(pnames))
        plt.tight_layout()

        figures["attribution"] = [fig_vol, fig_es]
        if bool(output_cfg["show_figures"]):
            plt.show()
            plt.show()

    return RiskReportArtifacts(tables=tables, figures=figures, series=series, text=texts)


__all__ = ["RiskReportArtifacts", "executive_bullets", "risk_report"]
