from __future__ import annotations

import tomllib
from pathlib import Path

import quantfinlab as qfl
from quantfinlab.common import MissingKernelsError, QuantFinLabError, RiskReportArtifacts
from quantfinlab.reports import risk_report


def test_top_level_package_exports_core_names_and_submodules() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert qfl.__version__ == project["project"]["version"]
    assert qfl.RiskReportArtifacts is RiskReportArtifacts
    assert issubclass(qfl.InputError, QuantFinLabError)
    assert issubclass(qfl.MissingKernelsError, MissingKernelsError)
    assert qfl.reports.risk_report is risk_report
    assert {"options", "portfolio", "risk", "reports", "RiskReportArtifacts"}.issubset(qfl.__all__)


def test_credit_and_realtime_exports_are_available() -> None:
    from quantfinlab import dataio, fixed_income, macro, ml

    assert "credit" in qfl.__all__
    for name in ["read_credit_facts", "read_credit_curves", "read_mpt_bins", "read_alfred", "read_statscan_vintages"]:
        assert name in dataio.__all__
        assert callable(getattr(dataio, name))
    assert callable(fixed_income.discount_from_zero)
    assert "overnight" in fixed_income.__all__
    assert {"dfm", "bvar", "realtime", "policy"}.issubset(macro.__all__)
    assert {"combination", "validation"}.issubset(ml.__all__)
    assert {"draw_summary", "gaussian_crps"}.issubset(ml.probabilistic.__all__)
