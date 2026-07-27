from __future__ import annotations

from quantfinlab.common.contracts import FundamentalReportArtifacts, RiskReportArtifacts

from .fundamental_report import fundamental_report
from .risk_report import executive_bullets, risk_report

__all__ = [
    "FundamentalReportArtifacts",
    "RiskReportArtifacts",
    "executive_bullets",
    "fundamental_report",
    "risk_report",
]
