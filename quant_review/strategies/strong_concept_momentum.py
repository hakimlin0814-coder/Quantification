from __future__ import annotations

import math
from typing import Any

import pandas as pd

from quant_review.strategies.concept_radar import ConceptRadarStrategy


class StrongConceptMomentumStrategy(ConceptRadarStrategy):
    """强势股票与强势概念的集合交集筛选。

    复用概念雷达的概念行情、成分股和盘中 rt_k 快照能力，新增一层：
    在全部合格股票中先按 N 日涨幅取前 P%，再与强势概念成分股取交集。
    """

    def run(self) -> dict[str, Any]:
        saved = {
            "concept_top_count": getattr(self.config, "concept_top_count", 80),
            "concept_rank_period": getattr(self.config, "concept_rank_period", 10),
            "concept_sort_field": getattr(self.config, "concept_sort_field", "ret10"),
            "concept_stock_limit_per_board": getattr(self.config, "concept_stock_limit_per_board", 120),
            "concept_recommend_count": getattr(self.config, "concept_recommend_count", 8),
            "concept_exclude_st": getattr(self.config, "concept_exclude_st", False),
            "concept_exclude_star": getattr(self.config, "concept_exclude_star", False),
            "concept_exclude_chinext": getattr(self.config, "concept_exclude_chinext", False),
            "concept_exclude_bj": getattr(self.config, "concept_exclude_bj", False),
            "lookback_days": getattr(self.config, "lookback_days", 92),
        }
        self._stock_rank_map: dict[str, dict[str, float]] = {}
        self._stock_return_days = max(1, int(getattr(self.config, "strong_concept_stock_days", 2)))
        self._concept_return_days = max(1, int(getattr(self.config, "strong_concept_concept_days", 2)))
        try:
            self.config.concept_top_count = max(1, int(getattr(self.config, "strong_concept_top_count", 10)))
            self.config.concept_rank_period = self._concept_return_days
            self.config.concept_sort_field = f"ret{self.config.concept_rank_period}"
            self.config.lookback_days = max(
                int(saved["lookback_days"]),
                92,
                self._stock_return_days + 2,
                self._concept_return_days + 2,
            )
            self.config.concept_stock_limit_per_board = max(
                120,
                int(getattr(self.config, "strong_concept_member_limit", 1000)),
            )
            self.config.concept_recommend_count = int(getattr(self.config, "strong_concept_recommend_count", 10))
            self.config.concept_exclude_st = bool(getattr(self.config, "strong_concept_exclude_st", True))
            self.config.concept_exclude_star = bool(getattr(self.config, "strong_concept_exclude_star", True))
            self.config.concept_exclude_chinext = bool(getattr(self.config, "strong_concept_exclude_chinext", True))
            self.config.concept_exclude_bj = bool(getattr(self.config, "strong_concept_exclude_bj", True))
            result = super().run()
            return self._finalize(result)
        finally:
            for key, value in saved.items():
                setattr(self.config, key, value)

    def _concept_rows(
        self,
        concepts: pd.DataFrame,
        daily: pd.DataFrame,
        moneyflow: pd.DataFrame,
        trade_dates: list[str],
    ) -> pd.DataFrame:
        rows = super()._concept_rows(concepts, daily, moneyflow, trade_dates)
        return self._attach_dynamic_concept_return(rows, daily, trade_dates)

    def _attach_dynamic_concept_return(
        self,
        rows: pd.DataFrame,
        daily: pd.DataFrame,
        trade_dates: list[str],
    ) -> pd.DataFrame:
        if rows.empty or daily.empty or "concept_code" not in rows.columns or not trade_dates:
            return rows
        data = daily.copy()
        data["ts_code"] = data["ts_code"].astype(str)
        data["trade_date"] = data["trade_date"].astype(str)
        data["close"] = pd.to_numeric(data.get("close", 0), errors="coerce")
        close = data.pivot_table(index="ts_code", columns="trade_date", values="close", aggfunc="last").reindex(columns=trade_dates)
        latest = trade_dates[-1]
        if latest not in close.columns:
            return rows
        field = f"ret{self._concept_return_days}"
        base = close.shift(self._concept_return_days, axis=1)[latest]
        values = (pd.to_numeric(close[latest], errors="coerce") / pd.to_numeric(base, errors="coerce") - 1) * 100
        lookup = values.to_dict()
        result = rows.copy()
        result["concept_code"] = result["concept_code"].astype(str)
        result[field] = result["concept_code"].map(lookup)
        result[f"rps{self._concept_return_days}"] = result[field].rank(pct=True) * 100
        return result

    def _stock_features(
        self,
        daily: pd.DataFrame,
        basic: pd.DataFrame,
        daily_basic: pd.DataFrame,
        trade_dates: list[str],
    ) -> pd.DataFrame:
        features = super()._stock_features(daily, basic, daily_basic, trade_dates)
        if features.empty or daily.empty or not trade_dates:
            return features
        data = daily.copy()
        data["ts_code"] = data["ts_code"].astype(str)
        data["trade_date"] = data["trade_date"].astype(str)
        data["close"] = pd.to_numeric(data.get("close", 0), errors="coerce")
        close = data.pivot_table(index="ts_code", columns="trade_date", values="close", aggfunc="last").reindex(columns=trade_dates)
        latest = trade_dates[-1]
        if latest not in close.columns:
            return features
        field = f"ret{self._stock_return_days}"
        base = close.shift(self._stock_return_days, axis=1)[latest]
        values = (pd.to_numeric(close[latest], errors="coerce") / pd.to_numeric(base, errors="coerce") - 1) * 100
        lookup = values.to_dict()
        result = features.copy()
        result["ts_code"] = result["ts_code"].astype(str)
        result[field] = result["ts_code"].map(lookup)
        result[f"rps{self._stock_return_days}"] = result[field].rank(pct=True) * 100
        return self._attach_industry_market_cap_rank(result, basic, daily_basic)

    def _attach_industry_market_cap_rank(
        self,
        features: pd.DataFrame,
        basic: pd.DataFrame,
        daily_basic: pd.DataFrame,
    ) -> pd.DataFrame:
        """按选定申万行业级别分别计算个股总市值排名。"""
        enabled = bool(getattr(self.config, "strong_concept_industry_mv_enabled", False))
        if not enabled or features.empty:
            return features

        level = str(getattr(self.config, "strong_concept_industry_rank_level", "sw_l3") or "sw_l3").strip()
        if level not in {"sw_l1", "sw_l2", "sw_l3"}:
            level = "sw_l3"
        level_label = {"sw_l1": "申万一级", "sw_l2": "申万二级", "sw_l3": "申万三级"}[level]
        members = self.engine._sw_industry_members(level)
        required = {"ts_code", "rank_industry", "rank_industry_code"}
        if members.empty or not required.issubset(members.columns):
            raise RuntimeError(f"{level_label}行业成分为空，无法应用行业市值排名条件。")
        if daily_basic.empty or "total_mv" not in daily_basic.columns:
            raise RuntimeError(f"总市值数据为空，无法应用{level_label}市值排名条件。")

        industry = members[["ts_code", "rank_industry", "rank_industry_code"]].copy()
        industry["ts_code"] = industry["ts_code"].astype(str)
        industry["rank_industry"] = industry["rank_industry"].fillna("").astype(str).str.strip()
        industry["rank_industry_code"] = industry["rank_industry_code"].fillna("").astype(str).str.strip()
        industry = industry[industry["rank_industry"] != ""].drop_duplicates("ts_code", keep="first")

        caps = daily_basic[["ts_code", "total_mv"]].copy()
        caps["ts_code"] = caps["ts_code"].astype(str)
        caps["total_mv"] = pd.to_numeric(caps["total_mv"], errors="coerce").fillna(0)
        ranked = industry.merge(caps, on="ts_code", how="inner")
        ranked = ranked[ranked["total_mv"] > 0].copy()
        if ranked.empty:
            raise RuntimeError(f"没有可用的股票总市值数据，无法应用{level_label}市值排名条件。")
        ranked = ranked.sort_values(
            ["rank_industry", "total_mv", "ts_code"],
            ascending=[True, False, True],
        )
        ranked["industry_mv_rank"] = ranked.groupby("rank_industry", sort=False).cumcount() + 1
        ranked["market_cap_yi"] = ranked["total_mv"] / 10000
        rank_cols = ["rank_industry", "rank_industry_code", "market_cap_yi", "industry_mv_rank"]
        result = features.drop(columns=[col for col in rank_cols if col in features.columns]).copy()
        return result.merge(ranked[["ts_code", *rank_cols]], on="ts_code", how="left")

    def _stock_rows(
        self,
        members: pd.DataFrame,
        features: pd.DataFrame,
        boards: pd.DataFrame,
    ) -> pd.DataFrame:
        self._stock_rank_map = self._build_global_stock_rank(features)
        member_input = members
        industry_enabled = bool(getattr(self.config, "strong_concept_industry_mv_enabled", False))
        if industry_enabled:
            top = max(1, int(getattr(self.config, "strong_concept_industry_mv_rank_top", 2)))
            rank_cols = ["ts_code", "rank_industry", "rank_industry_code", "industry_mv_rank"]
            rank_data = features[[col for col in rank_cols if col in features.columns]].copy()
            if "industry_mv_rank" not in rank_data.columns:
                raise RuntimeError("行业市值排名未生成，无法应用筛选条件。")
            rank_data["industry_mv_rank"] = pd.to_numeric(rank_data["industry_mv_rank"], errors="coerce")
            eligible_codes = set(
                rank_data.loc[
                    rank_data["industry_mv_rank"].between(1, top, inclusive="both"),
                    "ts_code",
                ].astype(str)
            )
            member_input = members[members["ts_code"].astype(str).isin(eligible_codes)].copy()

        rows = super()._stock_rows(member_input, features, boards)
        if rows.empty:
            return rows
        result = rows.copy()
        result["stock_rank"] = result["ts_code"].astype(str).map(
            lambda code: self._stock_rank_map.get(code, {}).get("rank", float("nan"))
        )
        result["stock_rank_pct"] = result["ts_code"].astype(str).map(
            lambda code: self._stock_rank_map.get(code, {}).get("strength_pct", float("nan"))
        )
        result["stock_top_value"] = self._stock_rank_value()
        stock_return_field = f"ret{self._stock_return_days}"
        stock_return_lookup = features.set_index("ts_code")[stock_return_field].to_dict() if stock_return_field in features.columns else {}
        result["stock_return"] = pd.to_numeric(
            result["ts_code"].astype(str).map(stock_return_lookup), errors="coerce"
        )
        result = result[result["stock_rank"].notna()].copy()
        result["stock_rank"] = pd.to_numeric(result["stock_rank"], errors="coerce")
        result["stock_rank_pct"] = pd.to_numeric(result["stock_rank_pct"], errors="coerce")
        result = result[result["stock_rank"] <= self._stock_rank_cutoff()].copy()
        if result.empty:
            return result
        rank_label = self._stock_rank_label()
        concept_top = int(getattr(self.config, "strong_concept_top_count", 10))
        result["intersection"] = f"{rank_label} ∩ 强势概念前{concept_top}名"
        result["strong_concept_pass"] = "是"
        if industry_enabled:
            lookup = features.set_index("ts_code")
            for column in ["rank_industry", "rank_industry_code", "industry_mv_rank", "market_cap_yi"]:
                if column in lookup.columns:
                    result[column] = result["ts_code"].astype(str).map(lookup[column].to_dict())
            result["industry_mv_rank_top"] = max(
                1, int(getattr(self.config, "strong_concept_industry_mv_rank_top", 2))
            )
        return result

    def _build_global_stock_rank(self, features: pd.DataFrame) -> dict[str, dict[str, float]]:
        if features.empty or "ts_code" not in features.columns:
            return {}
        data = features.copy()
        data["ts_code"] = data["ts_code"].astype(str)
        data = data[self._market_mask(data)].copy()
        if bool(getattr(self.config, "strong_concept_exclude_st", True)) and "name" in data.columns:
            data = data[~data["name"].astype(str).str.upper().str.contains("ST", regex=False, na=False)].copy()
        return_field = f"ret{self._stock_return_days}"
        if return_field not in data.columns:
            return {}
        data[return_field] = pd.to_numeric(data[return_field], errors="coerce")
        data = data.dropna(subset=[return_field]).copy()
        data = data[data.get("close", 0).astype(float) > 0].copy()
        if data.empty:
            return {}
        data = data.sort_values([return_field, "ts_code"], ascending=[False, True]).reset_index(drop=True)
        data["rank"] = data.index + 1
        count = len(data)
        data["strength_pct"] = (
            (count - data["rank"] + 1) / count * 100
        )
        return {
            str(row.ts_code): {
                "rank": float(row.rank),
                "strength_pct": float(row.strength_pct),
            }
            for row in data.itertuples()
        }

    def _market_mask(self, data: pd.DataFrame) -> pd.Series:
        scope = getattr(self.config, "strong_concept_market_scope", ["main_board"])
        if isinstance(scope, str):
            scope = [item.strip() for item in scope.split(",") if item.strip()]
        scope = set(scope or ["main_board"])
        symbol = data.get("symbol", data["ts_code"].astype(str).str[:6]).astype(str).str.zfill(6)
        ts_code = data["ts_code"].astype(str).str.upper()
        stock_type = data.get("stock_type", pd.Series("", index=data.index)).astype(str)

        mask = pd.Series(False, index=data.index)
        if "all" in scope:
            mask = pd.Series(True, index=data.index)
        if "main_board" in scope:
            mask |= stock_type.str.contains("主板", na=False) | symbol.str.startswith(
                ("000", "001", "002", "003", "600", "601", "603", "605")
            )
        if "chinext" in scope and not bool(getattr(self.config, "strong_concept_exclude_chinext", True)):
            mask |= symbol.str.startswith(("300", "301"))
        if "star" in scope and not bool(getattr(self.config, "strong_concept_exclude_star", True)):
            mask |= symbol.str.startswith(("688", "689"))
        if "bj" in scope and not bool(getattr(self.config, "strong_concept_exclude_bj", True)):
            mask |= ts_code.str.endswith(".BJ") | symbol.str.startswith(("4", "8", "92"))
        return mask

    def _stock_rank_cutoff(self) -> int:
        mode = str(getattr(self.config, "strong_concept_stock_rank_mode", "percentile"))
        if mode == "count":
            return max(1, int(getattr(self.config, "strong_concept_stock_top_count", 100)))
        total = max(1, len(self._stock_rank_map))
        pct = max(0.1, min(100.0, float(getattr(self.config, "strong_concept_stock_top_percent", 10.0))))
        return max(1, math.ceil(total * pct / 100.0))

    def _stock_rank_value(self) -> float | int:
        mode = str(getattr(self.config, "strong_concept_stock_rank_mode", "percentile"))
        if mode == "count":
            return max(1, int(getattr(self.config, "strong_concept_stock_top_count", 100)))
        return max(0.1, min(100.0, float(getattr(self.config, "strong_concept_stock_top_percent", 10.0))))

    def _stock_rank_label(self) -> str:
        value = self._stock_rank_value()
        mode = str(getattr(self.config, "strong_concept_stock_rank_mode", "percentile"))
        return f"股票前{int(value)}名" if mode == "count" else f"股票前{value:g}%"

    def _finalize(self, result: dict[str, Any]) -> dict[str, Any]:
        stocks = pd.DataFrame(result.get("stocks", []))
        if not stocks.empty:
            stocks["stock_rank"] = pd.to_numeric(stocks["stock_rank"], errors="coerce")
            stocks["stock_rank_pct"] = pd.to_numeric(stocks["stock_rank_pct"], errors="coerce")
            stocks = stocks.sort_values(["stock_rank", "stock_return"], ascending=[True, False])
            boards = pd.DataFrame(result.get("boards", []))
            board_lookup = {}
            if not boards.empty and "concept_code" in boards.columns:
                for board in boards.to_dict("records"):
                    board_lookup[str(board.get("concept_code", ""))] = board
            grouped = []
            for code, group in stocks.groupby("ts_code", sort=False):
                row = group.iloc[0].to_dict()
                row["concept_name"] = "、".join(dict.fromkeys(group["concept_name"].astype(str).tolist()))
                row["concept_code"] = ",".join(dict.fromkeys(group["concept_code"].astype(str).tolist()))
                row["strong_concept_count"] = int(group["concept_code"].nunique())
                concept_rows = [board_lookup.get(str(item), {}) for item in group["concept_code"].astype(str).tolist()]
                concept_rows = [item for item in concept_rows if item]
                if concept_rows:
                    row["concept_rank"] = min(float(item.get("rank", 999)) for item in concept_rows)
                    concept_field = f"ret{self._concept_return_days}"
                    row["concept_return"] = max(float(item.get(concept_field, 0) or 0) for item in concept_rows)
                grouped.append(row)
            stocks = pd.DataFrame(grouped).sort_values(["stock_rank", "stock_return"], ascending=[True, False])
            stocks = stocks.reset_index(drop=True)
            result["stocks"] = stocks.fillna("").to_dict("records")
            count = int(getattr(self.config, "strong_concept_recommend_count", 10))
            result["recommend_stocks"] = stocks.head(max(1, count)).fillna("").to_dict("records")
        else:
            result["stocks"] = []
            result["recommend_stocks"] = []
        market = result.setdefault("market", {})
        market.update(
            {
                "strategy": "strong_concept_momentum",
                "strategy_name": "强势股票×强势概念交集",
                "stock_rank_mode": getattr(self.config, "strong_concept_stock_rank_mode", "percentile"),
                "stock_top_percent": float(getattr(self.config, "strong_concept_stock_top_percent", 10.0)),
                "stock_top_count": self._stock_rank_cutoff(),
                "stock_top_value": self._stock_rank_value(),
                "stock_rank_label": self._stock_rank_label(),
                "concept_top_count": int(getattr(self.config, "strong_concept_top_count", 10)),
                "stock_return_days": int(getattr(self.config, "strong_concept_stock_days", 2)),
                "concept_return_days": int(getattr(self.config, "strong_concept_concept_days", 2)),
                "stock_universe_count": len(self._stock_rank_map),
                "market_scope": getattr(self.config, "strong_concept_market_scope", ["main_board"]),
                "exclude_st": bool(getattr(self.config, "strong_concept_exclude_st", True)),
                "industry_mv_rank_enabled": bool(getattr(self.config, "strong_concept_industry_mv_enabled", False)),
                "industry_mv_rank_top": max(1, int(getattr(self.config, "strong_concept_industry_mv_rank_top", 2))),
                "industry_rank_level": self._industry_rank_level(),
                "industry_rank_level_label": self._industry_rank_level_label(),
                "selection_formula": (
                    f"{self._stock_rank_label()} ∩ 概念涨幅前{int(getattr(self.config, 'strong_concept_top_count', 10))}名"
                    + (f" ∩ {self._industry_rank_level_label()}市值前{int(getattr(self.config, 'strong_concept_industry_mv_rank_top', 2))}名"
                       if bool(getattr(self.config, "strong_concept_industry_mv_enabled", False)) else "")
                ),
            }
        )
        return result

    def _industry_rank_level(self) -> str:
        level = str(getattr(self.config, "strong_concept_industry_rank_level", "sw_l3") or "sw_l3").strip()
        return level if level in {"sw_l1", "sw_l2", "sw_l3"} else "sw_l3"

    def _industry_rank_level_label(self) -> str:
        return {"sw_l1": "申万一级", "sw_l2": "申万二级", "sw_l3": "申万三级"}[self._industry_rank_level()]
