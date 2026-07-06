from __future__ import annotations

import quantfinlab as qfl
from quantfinlab.common import MissingKernelsError, QuantFinLabError, RiskReportArtifacts
from quantfinlab.reports import risk_report


def test_top_level_package_exports_core_names_and_submodules() -> None:
    assert qfl.__version__ == "0.5.0"
    assert qfl.RiskReportArtifacts is RiskReportArtifacts
    assert issubclass(qfl.InputError, QuantFinLabError)
    assert issubclass(qfl.MissingKernelsError, MissingKernelsError)
    assert qfl.reports.risk_report is risk_report
    assert {"options", "portfolio", "risk", "reports", "RiskReportArtifacts"}.issubset(qfl.__all__)
