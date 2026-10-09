"""End-to-end Q0 orchestration corresponding to 09 §3 and §26."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pandas as pd

from .config import OneilResearchConfig
from .execution import tradability_preview
from .factors import compute_financial_factors, compute_market_context, compute_price_factors
from .models import DataSnapshot
from .performance import screening_summary
from .signals import evaluate_q0_rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


class OneilResearchService:
    """Compose point-in-time factors into an auditable Q0 observation screen."""

    def __init__(self, config: OneilResearchConfig) -> None:
        config.validate()
        self.config = config

    def screen(self, snapshot: DataSnapshot) -> dict[str, Any]:
        price = compute_price_factors(snapshot.daily, self.config)
        financial = compute_financial_factors(snapshot.financials, snapshot.as_of_date, self.config)
        features = snapshot.basic.copy()
        features["ts_code"] = features["ts_code"].astype(str)
        features = features.merge(price, on="ts_code", how="left")
        features = features.merge(financial, on="ts_code", how="left")
        features["universe_point_in_time_risk"] = bool(snapshot.metadata.get("survivorship_risk", False))

        if snapshot.daily_basic is not None and not snapshot.daily_basic.empty and "ts_code" in snapshot.daily_basic.columns:
            daily_basic = snapshot.daily_basic.copy()
            daily_basic["ts_code"] = daily_basic["ts_code"].astype(str)
            keep = [
                col
                for col in ["ts_code", "turnover_rate_f", "volume_ratio", "total_mv", "circ_mv"]
                if col in daily_basic.columns
            ]
            features = features.merge(daily_basic[keep].drop_duplicates("ts_code"), on="ts_code", how="left")
        if "total_mv" in features.columns:
            features["market_cap_yi"] = pd.to_numeric(features["total_mv"], errors="coerce") / 10000
        if "amount" in features.columns:
            features["amount_yi"] = pd.to_numeric(features["amount"], errors="coerce") / 100000

        evaluated = evaluate_q0_rows(features, self.config)
        limit_lookup: dict[str, pd.Series] = {}
        if snapshot.limit_list is not None and not snapshot.limit_list.empty and "ts_code" in snapshot.limit_list.columns:
            limit_lookup = {
                str(row["ts_code"]): row
                for _, row in snapshot.limit_list.drop_duplicates("ts_code").iterrows()
            }
        if not evaluated.empty:
            evaluated["tradability"] = evaluated.apply(
                lambda row: tradability_preview(row, limit_lookup.get(str(row.get("ts_code", "")))), axis=1
            )
        market = compute_market_context(snapshot.market_indices, snapshot.daily, snapshot.limit_list)
        summary = screening_summary(evaluated)
        summary.update(
            {
                "universe": int(len(snapshot.basic)),
                "price_complete": int(evaluated.get("price_complete", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "financial_complete": int(evaluated.get("financial_complete", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
            }
        )
        visible = evaluated.head(self.config.top_n).copy()
        public_columns = [
            "ts_code",
            "symbol",
            "name",
            "stock_type",
            "industry",
            "status",
            "q0_observation_pass",
            "complete_oneil_status",
            "complete_oneil_note",
            "gate_completion_pct",
            "price_date",
            "close",
            "pct_chg",
            "amount_yi",
            "market_cap_yi",
            "eps_yoy_q",
            "sales_yoy_q",
            "annual_eps_up",
            "annual_eps_cagr",
            "annual_roe",
            "near_high_ratio",
            "new_high_breakout",
            "breakout_rvol",
            "rs_percentile",
            "history_days",
            "report_date",
            "max_ann_date",
            "financial_source",
            "financial_point_in_time_risk",
            "research_price_method",
            "tradability",
            "failure_reasons",
            "rule_results",
        ]
        for col in public_columns:
            if col not in visible.columns:
                visible[col] = pd.NA
        run_id = f"oneil-{snapshot.as_of_date}-{self.config.config_hash}"
        return _json_safe(
            {
                "ok": True,
                "run_id": run_id,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "as_of_date": snapshot.as_of_date,
                "data_cutoff": f"{snapshot.as_of_date} after_close",
                "research_level": self.config.research_level,
                "config": self.config.to_public_dict(),
                "data_sources": snapshot.metadata,
                "market": market,
                "summary": summary,
                "rows": visible[public_columns].to_dict("records"),
                "limitations": [
                    "Q0 PASS仅为观察池，不是买入信号",
                    "完整形态、事件型N、机构I和M买入门控未实现",
                    "A股情绪只展示，不参与筛选",
                    "组合仓位、成本与回测仍被配置门阻断",
                ],
            }
        )
