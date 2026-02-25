"""Data models for Azure Capacity Advisor."""

from models.machine import Machine
from models.result import AnalysisResult, SkuStatus, AnalysisSummary

__all__ = ["Machine", "AnalysisResult", "SkuStatus", "AnalysisSummary"]
