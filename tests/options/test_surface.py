from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.options.surface import (
    common_surface_grid,
    fit_log_total_variance_surface,
    fit_surface_panel,
    pchip_spline_comparison,
    pchip_surface,
    surface_cube,
    surface_features,
    surface_fit_summary,
    surface_grid,
    surface_iv,
    surface_iv_grid,
    surface_pca,
    surface_pca_shocks,
    surface_project_tables,
    surface_residuals,
    surface_total_variance,
)
from tests.synthetic.generators import option_surface_quotes


def test_log_total_variance_surface_fits_and_evaluates_on_grid() -> None:
    quotes = option_surface_quotes()

    grid = surface_grid(quotes, n_k=9, n_tau=7)
    fit = fit_log_total_variance_surface(quotes, n_k_basis=5, n_tau_basis=4, degree=2, lambda_k=0.01, lambda_tau=0.01, label="synthetic")
    iv_grid = surface_iv_grid(fit, grid)
    summary = surface_fit_summary(quotes, {"synthetic": fit})
    residuals = surface_residuals(quotes, {"synthetic": fit})

    assert iv_grid.shape == (7, 9)
    assert np.isfinite(surface_iv(fit, np.array([0.0]), np.array([45.0 / 365.25]))).all()
    assert summary.loc[0, "quote_count"] == len(quotes)
    assert residuals["abs_residual_synthetic"].median() < 0.02


def test_pchip_and_common_grid_helpers_share_surface_support() -> None:
    quotes = option_surface_quotes(dates=("2024-01-02", "2024-01-03"))
    grid = common_surface_grid(quotes, k_min=-0.20, k_max=0.12, tau_min=21 / 365.25, tau_max=105 / 365.25, n_k=7, n_tau=5, min_support_share=1.0)
    fit = fit_log_total_variance_surface(quotes, n_k_basis=5, n_tau_basis=4, degree=2, lambda_k=0.01, lambda_tau=0.01)

    one_day = quotes.loc[quotes["date"].eq(quotes["date"].min())].copy()
    raw = pchip_surface(one_day, grid=grid, min_k=5)
    comparison = pchip_spline_comparison(one_day, fit=fit, grid=grid)

    assert raw.shape == (5, 7)
    assert grid["support_mask"].shape == (5, 7)
    assert comparison.loc[0, "finite_nodes"] > 0


def test_surface_panel_cube_pca_and_project_tables_cover_historical_workflow() -> None:
    quotes = option_surface_quotes(dates=("2024-01-02", "2024-01-03", "2024-01-04"))
    params = {"n_k_basis": 5, "n_tau_basis": 4, "degree": 2, "lambda_k": 0.01, "lambda_tau": 0.01}

    panel = fit_surface_panel(
        quotes,
        visual_params=params,
        dupire_params=params,
        min_quotes=20,
        min_expiries=3,
    )
    grid = common_surface_grid(
        quotes,
        k_min=-0.20,
        k_max=0.12,
        tau_min=21 / 365.25,
        tau_max=105 / 365.25,
        n_k=7,
        n_tau=5,
        min_support_share=1.0,
    )
    cube = surface_cube(quotes, fits=panel["visual_fits"], grid=grid)
    pca = surface_pca(cube, n_components=2, random_state=7)
    shocks = surface_pca_shocks(pca, components=(1, 2))
    features = surface_features(cube, pca=pca)
    residuals = surface_residuals(
        quotes.loc[quotes["date"].eq(quotes["date"].min())],
        {"visual": panel["visual_fits"][pd.Timestamp("2024-01-02")]},
    )
    tables = surface_project_tables(
        fit_summary=panel["fit_summary"],
        residuals=residuals,
        pca=pca,
        features=features.assign(dupire_stress=features["short_atm_iv"]),
    )

    assert set(panel) == {"visual_fits", "dupire_fits", "fit_summary", "skipped", "elapsed_sec"}
    assert len(panel["visual_fits"]) == 3
    assert cube["values"].shape == (3, 5, 7)
    assert np.isfinite(surface_total_variance(panel["visual_fits"][pd.Timestamp("2024-01-02")], [0.0], [45 / 365.25])).all()
    assert pca["scores"].shape[0] == 2
    assert shocks["pc1"].shape == (5, 7)
    assert {"term_slope", "downside_skew", "curvature"}.issubset(features.columns)
    assert not tables["largest_residuals"].empty
    assert not tables["pca_explained_variance"].empty
    assert not tables["top_regimes"].empty
