from __future__ import annotations

import math
from typing import Any

import pandas as pd


class ShortMainlineStrategy:
    """Short-term mainline launch strategy.

    The strategy is intentionally narrow: find a few SW L2 industries that are
    moving from ignition into stable early consensus, then only pick stocks
    from those industries. Concepts are used as a naming and confirmation layer,
    not as the primary universe.
    """

    MAIN_BOARD_PREFIXES = ("000", "001", "002", "003", "600", "601", "603", "605")
    STAR_PREFIXES = ("688", "689")
    CHINEXT_PREFIXES = ("300", "301")
    BJ_PREFIXES = ("8", "4", "92")

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.config = engine.config

    def log(self, message: str) -> None:
        self.engine.log(message)

    def run(self) -> dict[str, Any]:
        old_lookback = int(self.config.lookback_days)
        self.config.lookback_days = max(old_lookback, 90)
        try:
            self.log("读取交易日历，短线主线策略需要约90个交易日用于5/10/20/60日RPS...")
            trade_dates = self.engine._trade_dates()
        finally:
            self.config.lookback_days = old_lookback

        trade_date = trade_dates[-1]
        self.config.end_date = trade_date
        self.log(f"使用交易日 {trade_date}")

        self.log("读取股票基础信息...")
        basic = self.engine._stock_basic()

        self.log("读取申万一级、二级行业成分...")
        l2_members = self.engine._sw_industry_members("sw_l2")
        l1_members = self.engine._sw_industry_members("sw_l1")

        self.log("读取日线行情...")
        daily = self.engine._daily_window(trade_dates)
        intraday_note = ""
        if self.config.review_mode == "intraday":
            self.log("盘中模式：用实时行情覆盖当日临时日K...")
            daily = self.engine._apply_intraday_snapshot(daily, basic, trade_date)
            intraday_note = "盘中临时日K会随行情变化，收盘后需要复核。"

        self.log("读取每日指标...")
        daily_basic = self.engine._daily_basic(trade_date)
        if self.config.review_mode == "intraday" and self.engine._daily_basic_is_empty(daily_basic) and len(trade_dates) >= 2:
            prev_date = trade_dates[-2]
            self.log(f"盘中模式：当日每日指标未更新，回退到 {prev_date}。")
            daily_basic = self.engine._daily_basic(prev_date)

        self.log("计算个股短线强度和行业归属...")
        features = self._stock_features(daily, basic, daily_basic, l2_members, l1_members, trade_dates)

        self.log("计算二级行业短线主线分...")
        industries = self._industry_rows(features)
        selected_industries = self._select_industries(industries)

        self.log("读取产业概念，用于给主线命名和共振加分...")
        concept_map = self._concept_map(features, trade_date, trade_dates)
        features["concepts"] = features["ts_code"].map(concept_map).fillna("")
        concept_stats = self._concept_stats(features, selected_industries)

        self.log("只在入围行业内部选前排、中军和弹性候选...")
        rows = self._stock_rows(features, selected_industries, concept_stats)
        market = self._market_payload(features, selected_industries, rows, intraday_note)

        self.log(f"短线主线稳定爆发策略完成：主线 {len(selected_industries)} 条，个股 {len(rows)} 只。")
        return {
            "ok": True,
            "trade_date": trade_date,
            "market": market,
            "industries": selected_industries,
            "rows": rows,
            "logs": self.engine.logs,
        }

    def _stock_features(
        self,
        daily: pd.DataFrame,
        basic: pd.DataFrame,
        daily_basic: pd.DataFrame,
        l2_members: pd.DataFrame,
        l1_members: pd.DataFrame,
        trade_dates: list[str],
    ) -> pd.DataFrame:
        data = daily.copy()
        data["ts_code"] = data["ts_code"].astype(str)
        data["trade_date"] = data["trade_date"].astype(str)
        for col in ["open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]:
            if col not in data.columns:
                data[col] = 0
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
        data = data[data["close"] > 0].sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        basic_use = basic[[col for col in ["ts_code", "symbol", "name", "industry", "market", "stock_type", "list_date"] if col in basic.columns]].copy()
        basic_use["ts_code"] = basic_use["ts_code"].astype(str)
        data = data.merge(basic_use, on="ts_code", how="left")
        data["symbol"] = data["symbol"].fillna(data["ts_code"].str.slice(0, 6)).astype(str).str.zfill(6)
        data["name"] = data["name"].fillna("")
        data["industry"] = data["industry"].fillna("未分类").replace("", "未分类")
        data["stock_type"] = data["stock_type"].fillna("")
        data["list_date"] = data["list_date"].fillna("19000101").astype(str)

        l2_map = self._member_map(l2_members, "l2")
        l1_map = self._member_map(l1_members, "l1")
        data = data.merge(l2_map, on="ts_code", how="left")
        data = data.merge(l1_map, on="ts_code", how="left")
        data["sw_l2"] = data["sw_l2"].fillna(data["industry"]).replace("", "未分类")
        data["sw_l2_code"] = data["sw_l2_code"].fillna(data["sw_l2"])
        data["sw_l1"] = data["sw_l1"].fillna(data["industry"]).replace("", "未分类")
        data["sw_l1_code"] = data["sw_l1_code"].fillna(data["sw_l1"])

        db = daily_basic.copy()
        if "ts_code" not in db.columns:
            db["ts_code"] = pd.Series(dtype=str)
        db["ts_code"] = db["ts_code"].astype(str)
        for col in ["turnover_rate", "turnover_rate_f", "volume_ratio", "total_mv", "circ_mv"]:
            if col not in db.columns:
                db[col] = 0
            db[col] = pd.to_numeric(db[col], errors="coerce").fillna(0)
        data = data.merge(db[["ts_code", "turnover_rate", "turnover_rate_f", "volume_ratio", "total_mv", "circ_mv"]], on="ts_code", how="left")
        for col in ["turnover_rate", "turnover_rate_f", "volume_ratio", "total_mv", "circ_mv"]:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

        ret_factor = (1 + data["pct_chg"] / 100).replace([math.inf, -math.inf], 1).fillna(1)
        cum_ret = ret_factor.groupby(data["ts_code"]).cumprod()
        first_close = data.groupby("ts_code")["close"].transform("first")
        first_cum = cum_ret.groupby(data["ts_code"]).transform("first").replace(0, 1).fillna(1)
        data["adj_close"] = first_close * cum_ret / first_cum
        price_factor = (data["adj_close"] / data["close"].replace(0, pd.NA)).replace([math.inf, -math.inf], 1).fillna(1)
        for col in ["open", "high", "low", "close"]:
            data[f"adj_{col}"] = data[col] * price_factor

        grouped = data.groupby("ts_code", group_keys=False)
        data["amount_yi"] = data["amount"] / 100000
        data["market_cap_yi"] = data["total_mv"] / 10000
        data["circ_mv_yi"] = data["circ_mv"] / 10000
        data["list_age_days"] = (
            pd.to_datetime(data["trade_date"], format="%Y%m%d", errors="coerce")
            - pd.to_datetime(data["list_date"], format="%Y%m%d", errors="coerce")
        ).dt.days.fillna(9999)
        data["eligible_base"] = self._eligible_base(data)

        for window in [5, 10, 20, 60]:
            min_periods = max(3, int(window * 0.75))
            data[f"ma{window}"] = grouped["adj_close"].transform(lambda series, window=window, min_periods=min_periods: series.rolling(window, min_periods=min_periods).mean())
            data[f"ret{window}"] = data["adj_close"] / grouped["adj_close"].shift(window) - 1
            rps_source = data[f"ret{window}"].where(data["eligible_base"])
            data[f"rps{window}"] = rps_source.groupby(data["trade_date"]).rank(pct=True) * 100

        data["rps20_delta5"] = data["rps20"] - grouped["rps20"].shift(5)
        data["rps20_delta10"] = data["rps20"] - grouped["rps20"].shift(10)
        data["avg_amount5"] = grouped["amount"].transform(lambda series: series.rolling(5, min_periods=3).mean())
        data["avg_amount20"] = grouped["amount"].transform(lambda series: series.rolling(20, min_periods=12).mean())
        data["amount_ratio"] = data["avg_amount5"] / data["avg_amount20"].replace(0, pd.NA)
        data["distance_ma5"] = (data["adj_close"] / data["ma5"] - 1) * 100
        data["distance_ma10"] = (data["adj_close"] / data["ma10"] - 1) * 100
        data["distance_ma20"] = (data["adj_close"] / data["ma20"] - 1) * 100
        data["high20"] = grouped["adj_high"].transform(lambda series: series.rolling(20, min_periods=15).max())
        data["high60"] = grouped["adj_high"].transform(lambda series: series.rolling(60, min_periods=45).max())
        data["high_position_60"] = data["adj_close"] / data["high60"].replace(0, pd.NA)
        data["breakout20"] = data["adj_close"] >= data["high20"] * 0.995
        data["ma_short_ok"] = (data["adj_close"] > data["ma5"]) & (data["ma5"] >= data["ma10"] * 0.995) & (data["ma10"] >= data["ma20"] * 0.99)
        return data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    def _industry_rows(self, features: pd.DataFrame) -> list[dict[str, Any]]:
        latest_date = str(self.config.end_date)
        latest = features[(features["trade_date"] == latest_date) & features["eligible_base"]].copy()
        if latest.empty:
            return []

        industry = latest.groupby(["sw_l2", "sw_l2_code", "sw_l1", "sw_l1_code"], as_index=False).agg(
            member_count=("ts_code", "nunique"),
            industry_ret5=("ret5", "median"),
            industry_ret10=("ret10", "median"),
            industry_ret20=("ret20", "median"),
            industry_ret60=("ret60", "median"),
            industry_amount_ratio=("amount_ratio", "median"),
            industry_up_ratio=("pct_chg", lambda values: (values > 0).mean() * 100),
            strong_stock_count=("rps10", lambda values: int((pd.to_numeric(values, errors="coerce") >= 80).sum())),
            stock_rps20_count=("rps20", lambda values: int((pd.to_numeric(values, errors="coerce") >= 70).sum())),
            above_ma20_ratio=("distance_ma20", lambda values: (pd.to_numeric(values, errors="coerce") > 0).mean() * 100),
        )
        for window in [5, 10, 20, 60]:
            industry[f"industry_rps{window}"] = industry[f"industry_ret{window}"].rank(pct=True, method="average") * 100

        l1 = latest.groupby(["sw_l1", "sw_l1_code"], as_index=False).agg(
            l1_ret5=("ret5", "median"),
            l1_ret10=("ret10", "median"),
            l1_ret20=("ret20", "median"),
        )
        for window in [5, 10, 20]:
            l1[f"l1_rps{window}"] = l1[f"l1_ret{window}"].rank(pct=True, method="average") * 100
        industry = industry.merge(l1[["sw_l1", "sw_l1_code", "l1_rps5", "l1_rps10", "l1_rps20"]], on=["sw_l1", "sw_l1_code"], how="left")

        cfg = self.config
        industry["rps_gap"] = industry["industry_rps5"] - industry["industry_rps20"]
        industry["launch_score"] = self._industry_launch_score(industry)
        industry["stable_score"] = self._industry_stable_score(industry)
        industry["l1_confirm_score"] = ((industry["l1_rps10"].fillna(0) + industry["l1_rps20"].fillna(0)) / 2).clip(0, 100)
        industry["hot_penalty"] = self._industry_hot_penalty(industry)
        industry["score"] = (
            industry["launch_score"] * 0.45
            + industry["stable_score"] * 0.25
            + industry["l1_confirm_score"] * 0.12
            + (industry["industry_amount_ratio"].fillna(1).clip(0, 2.2) / 2.2 * 100) * 0.10
            + (industry["above_ma20_ratio"].fillna(0).clip(0, 100)) * 0.08
            - industry["hot_penalty"]
        ).clip(0, 100)
        industry["state"] = industry.apply(self._industry_state, axis=1)
        industry["reason"] = industry.apply(self._industry_reason, axis=1)

        min_members = int(getattr(cfg, "short_mainline_min_industry_members", 8))
        industry = industry[industry["member_count"] >= max(1, min_members)].copy()
        industry = industry.sort_values(["score", "industry_rps5"], ascending=False).reset_index(drop=True)
        industry["rank"] = range(1, len(industry) + 1)
        return [self._industry_dict(row) for _, row in industry.iterrows()]

    def _select_industries(self, industries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cfg = self.config
        top_n = max(1, int(getattr(cfg, "short_mainline_top_industry_count", 5)))
        accepted = []
        for item in industries:
            if item["state"] not in {"刚稳定爆发", "爆发确认"}:
                continue
            if item["industry_rps5"] < float(getattr(cfg, "short_mainline_min_industry_rps5", 88)):
                continue
            if item["industry_rps10"] < float(getattr(cfg, "short_mainline_min_industry_rps10", 82)):
                continue
            if item["industry_rps20"] < float(getattr(cfg, "short_mainline_min_industry_rps20", 70)):
                continue
            if item["industry_rps60"] > float(getattr(cfg, "short_mainline_max_industry_rps60", 90)):
                continue
            if item["rps_gap"] < float(getattr(cfg, "short_mainline_min_industry_rps_gap", 8)):
                continue
            if item["industry_amount_ratio"] < float(getattr(cfg, "short_mainline_min_industry_amount_ratio", 1.12)):
                continue
            if item["industry_up_ratio"] < float(getattr(cfg, "short_mainline_min_industry_up_ratio", 52)):
                continue
            if item["strong_stock_count"] < int(getattr(cfg, "short_mainline_min_strong_stocks", 3)):
                continue
            accepted.append(item)
            if len(accepted) >= top_n:
                break
        if accepted:
            return accepted
        return [item for item in industries if item["state"] in {"刚稳定爆发", "爆发确认", "局部脉冲"}][:top_n]

    def _stock_rows(
        self,
        features: pd.DataFrame,
        industries: list[dict[str, Any]],
        concept_stats: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        latest = features[(features["trade_date"] == str(self.config.end_date)) & features["eligible_base"]].copy()
        if latest.empty or not industries:
            return []
        industry_by_name = {item["industry"]: item for item in industries}
        latest = latest[latest["sw_l2"].isin(industry_by_name)].copy()
        if latest.empty:
            return []

        latest["industry_stock_rank_pct"] = latest.groupby("sw_l2")["rps10"].rank(pct=True, ascending=False) * 100
        latest["concept_bonus"] = latest.apply(lambda row: self._concept_bonus(row, concept_stats), axis=1)
        latest["stock_score"] = latest.apply(lambda row: self._stock_score(row, industry_by_name.get(str(row.get("sw_l2", "")), {})), axis=1)
        latest["stock_score"] = (latest["stock_score"] + latest["concept_bonus"]).clip(0, 100)
        latest["role"] = latest.apply(self._stock_role, axis=1)
        latest["risk_flags"] = latest.apply(self._risk_flags, axis=1)

        cfg = self.config
        mask = (
            (latest["rps5"] >= float(getattr(cfg, "short_mainline_min_stock_rps5", 85)))
            & (latest["rps10"] >= float(getattr(cfg, "short_mainline_min_stock_rps10", 80)))
            & (latest["rps20"] >= float(getattr(cfg, "short_mainline_min_stock_rps20", 65)))
            & (latest["rps60"] >= float(getattr(cfg, "short_mainline_min_stock_rps60", 40)))
            & (latest["amount_yi"] >= float(getattr(cfg, "short_mainline_min_stock_amount_yi", 1.0)))
            & (latest["amount_ratio"] >= float(getattr(cfg, "short_mainline_min_stock_amount_ratio", 1.05)))
            & (latest["distance_ma20"] <= float(getattr(cfg, "short_mainline_max_stock_distance_ma20", 15)))
            & ((latest["ret10"].fillna(0) * 100) <= float(getattr(cfg, "short_mainline_max_stock_ret10", 35)))
            & (latest["ma_short_ok"])
        )
        candidates = latest[mask].copy()
        if candidates.empty:
            candidates = latest.sort_values("stock_score", ascending=False).head(int(getattr(cfg, "short_mainline_top_count", 20))).copy()

        per_industry = max(1, int(getattr(cfg, "short_mainline_stocks_per_industry", 4)))
        top_count = max(1, int(getattr(cfg, "short_mainline_top_count", 20)))
        rows: list[dict[str, Any]] = []
        for industry in industries:
            group = candidates[candidates["sw_l2"] == industry["industry"]].sort_values(["stock_score", "rps10"], ascending=False).head(per_industry)
            for _, row in group.iterrows():
                rows.append(self._stock_dict(row, industry, concept_stats))
        rows = sorted(rows, key=lambda item: item["score"], reverse=True)
        return rows[:top_count]

    def _concept_map(self, features: pd.DataFrame, trade_date: str, trade_dates: list[str]) -> dict[str, str]:
        if not bool(getattr(self.config, "short_mainline_concept_enabled", True)):
            return {}
        latest = features[features["trade_date"] == str(trade_date)]
        codes = latest["ts_code"].astype(str).tolist()
        return self.engine._stock_concept_map(
            str(trade_date),
            codes,
            board_limit=260,
            concepts_per_stock=5,
            fallback_trade_dates=list(reversed(trade_dates[-6:-1])),
        )

    def _concept_stats(self, features: pd.DataFrame, industries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if not bool(getattr(self.config, "short_mainline_concept_enabled", True)) or features.empty or not industries:
            return {}
        latest = features[(features["trade_date"] == str(self.config.end_date)) & features["sw_l2"].isin([item["industry"] for item in industries])].copy()
        rows = []
        for _, row in latest.iterrows():
            concepts = [item.strip() for item in str(row.get("concepts", "")).split("；") if item.strip()]
            for concept in concepts[:5]:
                rows.append(
                    {
                        "sw_l2": row.get("sw_l2", ""),
                        "concept": concept,
                        "ts_code": row.get("ts_code", ""),
                        "rps10": self._num(row.get("rps10")),
                        "rps20": self._num(row.get("rps20")),
                    }
                )
        data = pd.DataFrame(rows)
        if data.empty:
            return {}
        grouped = data.groupby(["sw_l2", "concept"], as_index=False).agg(
            concept_stock_count=("ts_code", "nunique"),
            concept_strong_count=("rps10", lambda values: int((pd.to_numeric(values, errors="coerce") >= 80).sum())),
            concept_avg_rps10=("rps10", "mean"),
            concept_avg_rps20=("rps20", "mean"),
        )
        min_count = int(getattr(self.config, "short_mainline_min_concept_stocks", 2))
        grouped = grouped[grouped["concept_strong_count"] >= max(1, min_count)].copy()
        grouped["concept_score"] = (
            grouped["concept_avg_rps10"].fillna(0) * 0.50
            + grouped["concept_avg_rps20"].fillna(0) * 0.30
            + (grouped["concept_strong_count"].clip(0, 6) / 6 * 100) * 0.20
        )
        result: dict[str, dict[str, Any]] = {}
        for industry, group in grouped.sort_values("concept_score", ascending=False).groupby("sw_l2"):
            top = group.head(3)
            result[str(industry)] = {
                "top_concepts": "；".join(top["concept"].astype(str).tolist()),
                "concept_score": round(float(top["concept_score"].max()), 2),
                "strong_count": int(top["concept_strong_count"].max()),
            }
        return result

    def _industry_launch_score(self, data: pd.DataFrame) -> pd.Series:
        clip = lambda series: series.clip(lower=0, upper=1)
        return (
            35 * clip((data["industry_rps5"] - 70) / 28)
            + 25 * clip((data["industry_rps10"] - 65) / 30)
            + 18 * clip((data["industry_rps20"] - 55) / 35)
            + 12 * clip((data["industry_rps5"] - data["industry_rps20"]) / 25)
            + 10 * clip((data["strong_stock_count"] - 1) / 5)
        )

    def _industry_stable_score(self, data: pd.DataFrame) -> pd.Series:
        clip = lambda series: series.clip(lower=0, upper=1)
        return (
            30 * clip((data["industry_rps10"] - 70) / 25)
            + 25 * clip((data["industry_rps20"] - 62) / 28)
            + 20 * clip((data["industry_up_ratio"] - 45) / 25)
            + 15 * clip((data["industry_amount_ratio"] - 1.0) / 0.8)
            + 10 * clip((data["stock_rps20_count"] - 1) / 5)
        )

    def _industry_hot_penalty(self, data: pd.DataFrame) -> pd.Series:
        penalty = pd.Series(0.0, index=data.index)
        penalty += (data["industry_rps60"] - 88).clip(lower=0) * 0.45
        penalty += ((data["industry_ret10"].fillna(0) * 100) - 24).clip(lower=0) * 0.5
        return penalty

    def _industry_state(self, row: pd.Series) -> str:
        if self._num(row.get("industry_rps60")) >= 90 and self._num(row.get("industry_rps5")) < self._num(row.get("industry_rps20")):
            return "退潮剔除"
        if self._num(row.get("industry_rps5")) >= 88 and self._num(row.get("industry_rps10")) >= 82 and self._num(row.get("industry_rps20")) >= 70 and self._num(row.get("industry_rps60")) <= 90:
            return "刚稳定爆发" if self._num(row.get("industry_rps5")) - self._num(row.get("industry_rps20")) >= 8 else "爆发确认"
        if self._num(row.get("industry_rps5")) >= 90 and self._num(row.get("industry_rps10")) < 75:
            return "一日游嫌疑"
        if self._num(row.get("industry_rps5")) >= 85 and self._num(row.get("industry_rps10")) >= 78:
            return "局部脉冲"
        if self._num(row.get("industry_rps60")) > 90:
            return "老主线观察"
        return "未入围"

    def _industry_reason(self, row: pd.Series) -> str:
        pieces = [
            f"RPS5/10/20={self._num(row.get('industry_rps5')):.0f}/{self._num(row.get('industry_rps10')):.0f}/{self._num(row.get('industry_rps20')):.0f}",
            f"60日={self._num(row.get('industry_rps60')):.0f}",
            f"量能{self._num(row.get('industry_amount_ratio')):.2f}倍",
            f"上涨{self._num(row.get('industry_up_ratio')):.0f}%",
            f"强股{int(self._num(row.get('strong_stock_count')))}只",
        ]
        return "；".join(pieces)

    def _stock_score(self, row: pd.Series, industry: dict[str, Any]) -> float:
        clip = lambda value: min(max(value, 0.0), 1.0)
        score = 0.0
        score += 25 * clip((self._num(row.get("rps5")) - 70) / 25)
        score += 22 * clip((self._num(row.get("rps10")) - 68) / 27)
        score += 15 * clip((self._num(row.get("rps20")) - 55) / 35)
        score += 10 * clip(self._num(row.get("rps20_delta5")) / 18)
        score += 10 * clip((self._num(row.get("amount_ratio")) - 1.0) / 1.2)
        score += 8 * (1 if bool(row.get("ma_short_ok", False)) else 0)
        score += 6 * (1 if bool(row.get("breakout20", False)) else 0)
        score += 4 * clip((100 - self._num(row.get("industry_stock_rank_pct"))) / 100)
        score += min(self._num(industry.get("score")) * 0.04, 4)
        overheat = max(self._num(row.get("distance_ma20")) - float(getattr(self.config, "short_mainline_max_stock_distance_ma20", 15)), 0) * 0.8
        overheat += max(self._num(row.get("ret10")) * 100 - float(getattr(self.config, "short_mainline_max_stock_ret10", 35)), 0) * 0.35
        return max(score - overheat, 0.0)

    def _concept_bonus(self, row: pd.Series, concept_stats: dict[str, dict[str, Any]]) -> float:
        stats = concept_stats.get(str(row.get("sw_l2", "")))
        if not stats:
            return 0.0
        row_concepts = set(item.strip() for item in str(row.get("concepts", "")).split("；") if item.strip())
        top_concepts = set(item.strip() for item in str(stats.get("top_concepts", "")).split("；") if item.strip())
        if not row_concepts.intersection(top_concepts):
            return 0.0
        return min(float(stats.get("concept_score", 0)) * 0.10, 8.0)

    def _stock_role(self, row: pd.Series) -> str:
        market_cap = self._num(row.get("market_cap_yi"))
        if self._num(row.get("industry_stock_rank_pct")) <= 25 and self._num(row.get("rps10")) >= 90:
            return "前排龙头"
        if market_cap >= 300 and self._num(row.get("amount_yi")) >= 5:
            return "中军确认"
        if self._num(row.get("rps20_delta5")) >= 10 and self._num(row.get("distance_ma20")) <= 10:
            return "补涨弹性"
        return "跟随确认"

    def _risk_flags(self, row: pd.Series) -> str:
        flags = []
        if self._num(row.get("distance_ma20")) > 12:
            flags.append("偏离MA20")
        if self._num(row.get("ret10")) * 100 > 28:
            flags.append("10日涨幅偏大")
        if self._num(row.get("amount_ratio")) > 2.8:
            flags.append("量能偏猛")
        if self._num(row.get("turnover_rate_f")) > 18:
            flags.append("换手偏高")
        return "；".join(flags) if flags else "正常"

    def _stock_dict(self, row: pd.Series, industry: dict[str, Any], concept_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
        stats = concept_stats.get(str(row.get("sw_l2", "")), {})
        return {
            "ts_code": str(row.get("ts_code", "")),
            "code": str(row.get("symbol", str(row.get("ts_code", ""))[:6])).zfill(6),
            "name": str(row.get("name", "")),
            "stock_type": str(row.get("stock_type", "")),
            "industry": str(row.get("sw_l2", "")),
            "industry_state": industry.get("state", ""),
            "sw_l1": str(row.get("sw_l1", "")),
            "concepts": str(row.get("concepts", "")),
            "strong_concepts": stats.get("top_concepts", ""),
            "role": str(row.get("role", "")),
            "score": round(self._num(row.get("stock_score")), 2),
            "industry_score": round(self._num(industry.get("score")), 2),
            "close": round(self._num(row.get("close")), 2),
            "pct_chg": round(self._num(row.get("pct_chg")), 2),
            "amount_yi": round(self._num(row.get("amount_yi")), 2),
            "amount_ratio": round(self._num(row.get("amount_ratio")), 2),
            "turnover_rate_f": round(self._num(row.get("turnover_rate_f")), 2),
            "market_cap_yi": round(self._num(row.get("market_cap_yi")), 2),
            "rps5": round(self._num(row.get("rps5")), 2),
            "rps10": round(self._num(row.get("rps10")), 2),
            "rps20": round(self._num(row.get("rps20")), 2),
            "rps60": round(self._num(row.get("rps60")), 2),
            "rps20_delta5": round(self._num(row.get("rps20_delta5")), 2),
            "ret5": round(self._num(row.get("ret5")) * 100, 2),
            "ret10": round(self._num(row.get("ret10")) * 100, 2),
            "ret20": round(self._num(row.get("ret20")) * 100, 2),
            "distance_ma5": round(self._num(row.get("distance_ma5")), 2),
            "distance_ma10": round(self._num(row.get("distance_ma10")), 2),
            "distance_ma20": round(self._num(row.get("distance_ma20")), 2),
            "high_position_60": round(self._num(row.get("high_position_60")), 3),
            "breakout20": "是" if bool(row.get("breakout20", False)) else "否",
            "risk_flags": str(row.get("risk_flags", "")),
            "reason": self._stock_reason(row, industry, stats),
        }

    def _industry_dict(self, row: pd.Series) -> dict[str, Any]:
        return {
            "rank": int(self._num(row.get("rank"))),
            "industry": str(row.get("sw_l2", "")),
            "industry_code": str(row.get("sw_l2_code", "")),
            "sw_l1": str(row.get("sw_l1", "")),
            "sw_l1_code": str(row.get("sw_l1_code", "")),
            "state": str(row.get("state", "")),
            "score": round(self._num(row.get("score")), 2),
            "industry_rps5": round(self._num(row.get("industry_rps5")), 2),
            "industry_rps10": round(self._num(row.get("industry_rps10")), 2),
            "industry_rps20": round(self._num(row.get("industry_rps20")), 2),
            "industry_rps60": round(self._num(row.get("industry_rps60")), 2),
            "rps_gap": round(self._num(row.get("rps_gap")), 2),
            "industry_amount_ratio": round(self._num(row.get("industry_amount_ratio")), 2),
            "industry_up_ratio": round(self._num(row.get("industry_up_ratio")), 2),
            "strong_stock_count": int(self._num(row.get("strong_stock_count"))),
            "stock_rps20_count": int(self._num(row.get("stock_rps20_count"))),
            "above_ma20_ratio": round(self._num(row.get("above_ma20_ratio")), 2),
            "l1_rps10": round(self._num(row.get("l1_rps10")), 2),
            "l1_rps20": round(self._num(row.get("l1_rps20")), 2),
            "member_count": int(self._num(row.get("member_count"))),
            "reason": str(row.get("reason", "")),
        }

    def _stock_reason(self, row: pd.Series, industry: dict[str, Any], concept_stats: dict[str, Any]) -> str:
        pieces = [
            str(row.get("role", "")),
            f"行业{industry.get('state', '')}",
            f"个股RPS5/10/20={self._num(row.get('rps5')):.0f}/{self._num(row.get('rps10')):.0f}/{self._num(row.get('rps20')):.0f}",
            f"量能{self._num(row.get('amount_ratio')):.2f}倍",
            f"距MA20 {self._num(row.get('distance_ma20')):.1f}%",
        ]
        if concept_stats.get("top_concepts"):
            pieces.append(f"概念{concept_stats.get('top_concepts')}")
        return "；".join([item for item in pieces if item])

    def _market_payload(
        self,
        features: pd.DataFrame,
        industries: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        intraday_note: str,
    ) -> dict[str, Any]:
        latest = features[(features["trade_date"] == str(self.config.end_date)) & features["eligible_base"]]
        if latest.empty:
            return {}
        note = "只抓少数刚稳定爆发主线，信号过多时以行业得分和前排扩散优先。"
        if intraday_note:
            note = f"{note} {intraday_note}"
        return {
            "industry_count": len(industries),
            "stock_count": len(rows),
            "market_up_ratio": round(float((latest["pct_chg"] > 0).mean() * 100), 2),
            "market_above_ma20": round(float((latest["distance_ma20"] > 0).mean() * 100), 2),
            "top_industries": "；".join(item["industry"] for item in industries[:5]),
            "note": note,
        }

    def _eligible_base(self, data: pd.DataFrame) -> pd.Series:
        symbol = data["symbol"].astype(str)
        name = data["name"].astype(str)
        mask = (data["list_age_days"] >= 90) & ~symbol.str.startswith(self.BJ_PREFIXES)
        if bool(getattr(self.config, "short_mainline_exclude_st", True)):
            mask &= ~name.str.upper().str.contains("ST", regex=False)
        if bool(getattr(self.config, "short_mainline_exclude_star", True)):
            mask &= ~symbol.str.startswith(self.STAR_PREFIXES)
        if bool(getattr(self.config, "short_mainline_exclude_chinext", True)):
            mask &= ~symbol.str.startswith(self.CHINEXT_PREFIXES)
        return mask.fillna(False)

    @staticmethod
    def _member_map(members: pd.DataFrame, suffix: str) -> pd.DataFrame:
        columns = [f"sw_{suffix}", f"sw_{suffix}_code"]
        if members.empty or "ts_code" not in members.columns or "rank_industry" not in members.columns:
            return pd.DataFrame(columns=["ts_code", *columns])
        data = members.copy()
        data["ts_code"] = data["ts_code"].astype(str)
        data[f"sw_{suffix}"] = data["rank_industry"].fillna("").astype(str)
        if "rank_industry_code" not in data.columns:
            data["rank_industry_code"] = ""
        data[f"sw_{suffix}_code"] = data["rank_industry_code"].fillna("").astype(str)
        return data[["ts_code", *columns]].drop_duplicates("ts_code", keep="first")

    @staticmethod
    def _num(value: Any, default: float = 0.0) -> float:
        try:
            if pd.isna(value):
                return default
            result = float(value)
            if math.isfinite(result):
                return result
        except Exception:
            pass
        return default
