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
