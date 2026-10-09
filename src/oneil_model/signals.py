"""Ordered Q0 gates and explicit incomplete states from 09 §17."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import OneilResearchConfig
from .models import EvaluationStatus, RuleResult, SourceTag


def _num(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _rule(rule_id: str, actual: Any, threshold: Any, source: SourceTag, comparator: str = "ge") -> RuleResult:
    value = _num(actual)
    if value is None:
        return RuleResult(rule_id, None, None, threshold, source, "数据缺失，不按通过处理")
    passed = value >= float(threshold) if comparator == "ge" else value <= float(threshold)
    return RuleResult(rule_id, bool(passed), value, threshold, source)


def _bool_rule(rule_id: str, actual: Any, source: SourceTag) -> RuleResult:
    if actual is None or pd.isna(actual):
        return RuleResult(rule_id, None, None, True, source, "数据缺失，不按通过处理")
    return RuleResult(rule_id, bool(actual), bool(actual), True, source)


def evaluate_q0_rows(features: pd.DataFrame, config: OneilResearchConfig) -> pd.DataFrame:
    """Evaluate the explicitly limited Q0 observation pool.

    Complete O'Neil status remains INCOMPLETE because I, the full pattern detector,
    and M entry gate are not implemented. See 09 §17.
    """
    if features is None or features.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in features.iterrows():
        rules = [
            _rule("C_EPS_YOY", row.get("eps_yoy_q"), config.min_eps_yoy, SourceTag.ORIGINAL),
            _rule("C_SALES_YOY", row.get("sales_yoy_q"), config.min_sales_yoy, SourceTag.ORIGINAL),
            _bool_rule("A_EPS_UP", row.get("annual_eps_up"), SourceTag.ORIGINAL),
            _rule("A_EPS_CAGR", row.get("annual_eps_cagr"), config.min_annual_eps_cagr, SourceTag.ORIGINAL),
            _rule("A_ROE", row.get("annual_roe"), config.min_roe, SourceTag.ORIGINAL),
            _rule("N_NEAR_HIGH", row.get("near_high_ratio"), config.near_high_ratio, SourceTag.DERIVED),
            _bool_rule("N_NEW_HIGH", row.get("new_high_breakout"), SourceTag.DERIVED),
            _rule("S_RVOL", row.get("breakout_rvol"), config.min_breakout_rvol, SourceTag.ORIGINAL),
            _rule("L_RS", row.get("rs_percentile"), config.min_rs_percentile, SourceTag.ORIGINAL),
        ]
        if not bool(row.get("financial_complete", False)):
            rules.append(RuleResult("DATA_FINANCIAL", None, None, "complete", SourceTag.A_SHARE, "财务点时数据不完整或兜底数据有修订风险"))
        if not bool(row.get("price_complete", False)):
            rules.append(RuleResult("DATA_PRICE", None, None, "complete", SourceTag.A_SHARE, "价格历史不足或观察日停牌"))
        if bool(row.get("universe_point_in_time_risk", False)):
            rules.append(RuleResult("DATA_UNIVERSE", None, None, "point-in-time", SourceTag.A_SHARE, "股票池来自当前名单兜底，存在生存者偏差"))
        known_failures = [item for item in rules if item.passed is False]
        unknown = [item for item in rules if item.passed is None]
        core_rules = [item for item in rules if not item.rule_id.startswith("DATA_")]
        q0_pass = not unknown and not known_failures and all(item.passed is True for item in core_rules)
        if unknown:
            status = EvaluationStatus.INCOMPLETE
        elif known_failures:
            status = EvaluationStatus.REJECT
        else:
            status = EvaluationStatus.PASS
        passed_count = sum(item.passed is True for item in core_rules)
        failure_reasons = [
            f"{item.rule_id}:{item.note or '未达到阈值'}"
            for item in rules
            if item.passed is not True
        ]
        output = row.to_dict()
        output.update(
            {
                "status": status.value,
                "q0_observation_pass": bool(q0_pass),
                "complete_oneil_status": EvaluationStatus.INCOMPLETE.value,
                "complete_oneil_note": "I、完整基底形态及M买入门控尚未实现；Q0通过不等于买入信号",
                "gate_completion_pct": round(passed_count / len(core_rules) * 100, 1),
                "rule_results": [item.to_dict() for item in rules],
                "failure_reasons": failure_reasons,
            }
        )
        rows.append(output)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["_status_order"] = result["status"].map({"PASS": 0, "INCOMPLETE": 1, "REJECT": 2}).fillna(9)
    result = result.sort_values(
        ["_status_order", "gate_completion_pct", "rs_percentile", "breakout_rvol"],
        ascending=[True, False, False, False],
        na_position="last",
    ).drop(columns=["_status_order"])
    return result.reset_index(drop=True)
