from __future__ import annotations

from typing import Any

import pandas as pd


class FirstLimitStrategy:
    """涨停板范围筛选策略。

    筛选条件：
    1. 当日涨停，且 limit_times 落在页面配置的连板范围内。
    2. 排除 ST / *ST。
    3. 排除科创板、创业板：688/689/300/301 开头。
    4. 总市值大于页面配置的下限。
    5. 指定换手率口径低于页面配置的上限。

    这个策略更偏“低换手、大市值、涨停异动”观察池，
    不等同于直接买入信号，后续还要结合板块、竞价和次日承接。
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.config = engine.config

    def log(self, message: str) -> None:
        self.engine.log(message)

    def run(self) -> dict[str, Any]:
        self.log("读取交易日历...")
        trade_dates = self.engine._trade_dates()
        trade_date = trade_dates[-1]
        self.config.end_date = trade_date
        self.log(f"使用交易日: {trade_date}")

        self.log("读取股票基础信息...")
        basic = self.engine._stock_basic()

        self.log("读取每日指标...")
        daily_basic = self.engine._daily_basic(trade_date)

        self.log("读取当日日线行情...")
        daily = self.engine._daily_window([trade_date])

        self.log("读取涨跌停数据...")
        limit_list = self.engine._limit_list(trade_date)

        rows = self._rows(basic, daily_basic, daily, limit_list)
        self.log(f"涨停板筛选完成，共 {len(rows)} 只。")
        return {
            "ok": True,
            "trade_date": trade_date,
            "rows": rows,
            "logs": self.engine.logs,
        }

    def _rows(
        self,
        basic: pd.DataFrame,
        daily_basic: pd.DataFrame,
        daily: pd.DataFrame,
        limit_list: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """按“涨停板范围”策略生成结果行。"""
        if limit_list.empty:
            self.log("涨跌停数据为空，无法筛选涨停板。")
            return []

        data = limit_list.copy()
        data["ts_code"] = data["ts_code"].astype(str)

        # 只保留涨停。不同 Tushare 版本字段可能叫 limit / limit_type，
        # 所以这里同时兼容中文和 U/D/Z 标记。
        up_mask = pd.Series(False, index=data.index)
        if "limit_type" in data.columns:
            up_mask |= data["limit_type"].astype(str).str.upper().str.contains("U")
        if "limit" in data.columns:
            up_mask |= data["limit"].astype(str).str.contains("涨|U|up", case=False, regex=True)
        if "limit_type" not in data.columns and "limit" not in data.columns:
            up_mask = pd.Series(True, index=data.index)
        data = data[up_mask].copy()

        # 连板范围：1-1 表示首板，1-2 表示首板或二板，3- 表示三板及以上。
        if "limit_times" not in data.columns:
            self.log("涨跌停数据缺少 limit_times 字段，无法严格识别连板数。")
            return []
        min_limit, max_limit = self._parse_board_range(self.config.first_limit_board_range)
        data["limit_times"] = pd.to_numeric(data["limit_times"], errors="coerce").fillna(0)
        data = data[data["limit_times"] >= min_limit].copy()
        if max_limit is not None:
            data = data[data["limit_times"] <= max_limit].copy()
        if data.empty:
            return []

        basic_cols = ["ts_code", "symbol", "name", "industry", "market", "stock_type", "list_date"]
        basic_use = basic[[col for col in basic_cols if col in basic.columns]].copy()
        data = data.merge(basic_use, on="ts_code", how="left", suffixes=("", "_basic"))

        db = daily_basic.copy()
        if "ts_code" in db.columns:
            db["ts_code"] = db["ts_code"].astype(str)
        for col in ["turnover_rate", "turnover_rate_f", "total_mv", "circ_mv", "volume_ratio"]:
            if col not in db.columns:
                db[col] = 0
            db[col] = pd.to_numeric(db[col], errors="coerce").fillna(0)

        rank_level = self.config.first_limit_industry_rank_level
        rank_industry = basic_use[["ts_code", "industry"]].copy()
        rank_industry["rank_industry"] = rank_industry["industry"].fillna("未分类").replace("", "未分类")
        rank_industry["rank_industry_code"] = ""
        if rank_level != "basic":
            sw_members = self.engine._sw_industry_members(rank_level)
            if not sw_members.empty:
                rank_industry = rank_industry.drop(columns=["rank_industry", "rank_industry_code"]).merge(
                    sw_members,
                    on="ts_code",
                    how="left",
                )
                rank_industry["rank_industry"] = rank_industry["rank_industry"].fillna(rank_industry["industry"]).replace("", "未分类")
                rank_industry["rank_industry_code"] = rank_industry["rank_industry_code"].fillna("")

        rank_base = basic_use.merge(db[["ts_code", "total_mv"]], on="ts_code", how="left")
        rank_base = rank_base.merge(rank_industry[["ts_code", "rank_industry", "rank_industry_code"]], on="ts_code", how="left")
        rank_base["rank_industry"] = rank_base["rank_industry"].fillna(rank_base["industry"]).replace("", "未分类")
        rank_base["market_cap_yi"] = pd.to_numeric(rank_base["total_mv"], errors="coerce").fillna(0) / 10000
        rank_base["industry_mv_rank"] = (
            rank_base.groupby("rank_industry", dropna=False)["market_cap_yi"]
            .rank(method="first", ascending=False)
            .fillna(9999)
            .astype(int)
        )
        industry_rank_map = rank_base.set_index("ts_code")["industry_mv_rank"].to_dict()
        rank_industry_map = rank_base.set_index("ts_code")["rank_industry"].to_dict()
        rank_industry_code_map = rank_base.set_index("ts_code")["rank_industry_code"].to_dict()

        daily_basic_cols = ["turnover_rate", "turnover_rate_f", "total_mv", "circ_mv", "volume_ratio"]
        data = data.drop(columns=[col for col in daily_basic_cols if col in data.columns])
        data = data.merge(db[["ts_code", *daily_basic_cols]], on="ts_code", how="left")

        d = daily.copy()
        if "ts_code" in d.columns:
            d["ts_code"] = d["ts_code"].astype(str)
        for col in ["close", "pct_chg", "amount", "vol"]:
            if col not in d.columns:
                d[col] = 0
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0)
        data = data.drop(columns=[col for col in ["close", "pct_chg", "amount", "vol"] if col in data.columns])
        data = data.merge(d[["ts_code", "close", "pct_chg", "amount", "vol"]], on="ts_code", how="left")

        data["market_cap_yi"] = pd.to_numeric(data["total_mv"], errors="coerce").fillna(0) / 10000
        data["circ_mv_yi"] = pd.to_numeric(data["circ_mv"], errors="coerce").fillna(0) / 10000
        data["turnover_rate"] = pd.to_numeric(data["turnover_rate"], errors="coerce").fillna(0)
        data["turnover_rate_f"] = pd.to_numeric(data["turnover_rate_f"], errors="coerce").fillna(0)
        data["amount_yi"] = pd.to_numeric(data["amount"], errors="coerce").fillna(0) / 100000
        data["industry_mv_rank"] = data["ts_code"].map(industry_rank_map).fillna(9999).astype(int)
        data["rank_industry"] = data["ts_code"].map(rank_industry_map).fillna(data["industry"]).replace("", "未分类")
        data["rank_industry_code"] = data["ts_code"].map(rank_industry_code_map).fillna("")

        name = data["name"].fillna("").astype(str)
        symbol = data.get("symbol", data["ts_code"].str.slice(0, 6)).astype(str)
        if self.config.first_limit_exclude_st:
            data = data[~name.str.upper().str.contains("ST", regex=False)].copy()
        if self.config.first_limit_exclude_star:
            data = data[~symbol.str.startswith(("688", "689"))].copy()
        if self.config.first_limit_exclude_chinext:
            data = data[~symbol.str.startswith(("300", "301"))].copy()

        min_mv = float(self.config.first_limit_min_market_cap_yi)
        max_turnover = float(self.config.first_limit_max_turnover_rate)
        turnover_field = self.config.first_limit_turnover_field
        if turnover_field not in {"turnover_rate", "turnover_rate_f"}:
            turnover_field = "turnover_rate_f"
        turnover_label = "实换手" if turnover_field == "turnover_rate_f" else "换手"
        data = data[(data["market_cap_yi"] > min_mv) & (data[turnover_field] < max_turnover)].copy()
        rank_top = int(self.config.first_limit_industry_mv_rank_top)
        if rank_top > 0:
            data = data[data["industry_mv_rank"] <= rank_top].copy()
        if data.empty:
            return []

        data = data.sort_values(["limit_times", "market_cap_yi", "amount_yi"], ascending=[False, False, False])
        rows = []
        for _, row in data.iterrows():
            rows.append(
                {
                    "ts_code": row.get("ts_code", ""),
                    "code": row.get("symbol", str(row.get("ts_code", ""))[:6]),
                    "name": row.get("name", ""),
                    "stock_type": row.get("stock_type", ""),
                    "industry": row.get("industry", ""),
                    "rank_industry": row.get("rank_industry", row.get("industry", "")),
                    "rank_industry_code": row.get("rank_industry_code", ""),
                    "market": row.get("market", ""),
                    "close": round(float(row.get("close", 0) or 0), 2),
                    "pct_chg": round(float(row.get("pct_chg", 0) or 0), 2),
                    "amount_yi": round(float(row.get("amount_yi", 0) or 0), 2),
                    "turnover_rate": round(float(row.get("turnover_rate", 0) or 0), 2),
                    "turnover_rate_f": round(float(row.get("turnover_rate_f", 0) or 0), 2),
                    "market_cap_yi": round(float(row.get("market_cap_yi", 0) or 0), 2),
                    "circ_mv_yi": round(float(row.get("circ_mv_yi", 0) or 0), 2),
                    "industry_mv_rank": int(row.get("industry_mv_rank", 0) or 0),
                    "volume_ratio": round(float(row.get("volume_ratio", 0) or 0), 2),
                    "limit_times": int(row.get("limit_times", 1) or 1),
                    "reason": f"涨停{self.config.first_limit_board_range}板；市值>{min_mv:g}亿；{turnover_label}<{max_turnover:g}%；行业市值前{rank_top if rank_top > 0 else '不限'}；行业口径={rank_level}",
                }
            )
        return rows

    def _parse_board_range(self, value: str) -> tuple[int, int | None]:
        text = (value or "1-1").strip()
        if not text:
            text = "1-1"
        if "-" not in text:
            number = self._positive_int(text, 1)
            return number, number
        left, right = text.split("-", 1)
        min_limit = max(self._positive_int(left, 1), 1)
        max_limit = self._positive_int(right, 0) if right.strip() else None
        if max_limit is not None and max_limit < min_limit:
            self.log(f"连板范围 {text} 不合法，已自动调整为 {min_limit}-{min_limit}。")
            max_limit = min_limit
        return min_limit, max_limit

    @staticmethod
    def _positive_int(value: str, default: int) -> int:
        try:
            number = int(str(value).strip())
        except (TypeError, ValueError):
            number = default
        return max(number, 0)
