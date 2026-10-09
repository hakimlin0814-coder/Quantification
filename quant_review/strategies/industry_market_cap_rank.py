from __future__ import annotations

from typing import Any

import pandas as pd


LEVEL_LABELS = {
    "sw_l1": "申万一级",
    "sw_l2": "申万二级",
}

RANK_TYPE_LABELS = {
    "industry_market_cap": "行业总市值排名",
    "stock_market_cap": "个股总市值排名",
    "stock_price": "个股股价排名",
}


def rank_industry_market_caps(
    basic: pd.DataFrame,
    daily_basic: pd.DataFrame,
    industry_members: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate total market cap by SW industry and rank descending.

    Tushare daily_basic.total_mv is in 10k CNY, so dividing by 10000 gives
    100m CNY.
    """
    columns = [
        "rank",
        "rank_industry",
        "rank_industry_code",
        "total_market_cap_yi",
        "member_count",
        "priced_member_count",
        "avg_market_cap_yi",
    ]
    required_members = {"ts_code", "rank_industry", "rank_industry_code"}
    if (
        basic.empty
        or daily_basic.empty
        or industry_members.empty
        or "ts_code" not in basic.columns
        or "ts_code" not in daily_basic.columns
        or "total_mv" not in daily_basic.columns
        or not required_members.issubset(industry_members.columns)
    ):
        return pd.DataFrame(columns=columns)

    universe = basic[["ts_code"]].copy()
    universe["ts_code"] = universe["ts_code"].astype(str)
    universe = universe.drop_duplicates("ts_code", keep="first")

    members = industry_members[["ts_code", "rank_industry", "rank_industry_code"]].copy()
    members["ts_code"] = members["ts_code"].astype(str)
    members["rank_industry"] = members["rank_industry"].fillna("").astype(str).str.strip()
    members["rank_industry_code"] = members["rank_industry_code"].fillna("").astype(str).str.strip()
    members = members[members["rank_industry"] != ""].drop_duplicates("ts_code", keep="first")

    market_cap = daily_basic[["ts_code", "total_mv"]].copy()
    market_cap["ts_code"] = market_cap["ts_code"].astype(str)
    market_cap["total_mv"] = pd.to_numeric(market_cap["total_mv"], errors="coerce").fillna(0)
    market_cap = market_cap.drop_duplicates("ts_code", keep="last")

    data = universe.merge(members, on="ts_code", how="inner")
    data = data.merge(market_cap, on="ts_code", how="left")
    data["total_mv"] = pd.to_numeric(data["total_mv"], errors="coerce").fillna(0)

    if data.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        data.groupby(["rank_industry", "rank_industry_code"], as_index=False)
        .agg(
            total_mv=("total_mv", "sum"),
            member_count=("ts_code", "nunique"),
            priced_member_count=("total_mv", lambda item: int((pd.to_numeric(item, errors="coerce").fillna(0) > 0).sum())),
        )
        .copy()
    )
    grouped = grouped[grouped["total_mv"] > 0].copy()
    if grouped.empty:
        return pd.DataFrame(columns=columns)

    grouped["total_market_cap_yi"] = grouped["total_mv"] / 10000
    grouped["avg_market_cap_yi"] = grouped["total_market_cap_yi"] / grouped["priced_member_count"].replace(0, pd.NA)
    grouped = grouped.sort_values(
        ["total_market_cap_yi", "rank_industry_code", "rank_industry"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    grouped["rank"] = grouped.index + 1

    result = grouped[columns].copy()
    result["total_market_cap_yi"] = result["total_market_cap_yi"].round(2)
    result["avg_market_cap_yi"] = pd.to_numeric(result["avg_market_cap_yi"], errors="coerce").fillna(0).round(2)
    result["member_count"] = result["member_count"].astype(int)
    result["priced_member_count"] = result["priced_member_count"].astype(int)
    return result


def rank_stock_market_caps(
    basic: pd.DataFrame,
    daily_basic: pd.DataFrame,
) -> pd.DataFrame:
    """Rank individual stocks by total market cap descending."""
    columns = [
        "rank",
        "code",
        "name",
        "stock_type",
        "industry",
        "total_market_cap_yi",
        "total_mv",
        "close",
        "pct_chg",
        "trade_date",
    ]
    if (
        basic.empty
        or daily_basic.empty
        or "ts_code" not in basic.columns
        or "ts_code" not in daily_basic.columns
        or "total_mv" not in daily_basic.columns
    ):
        return pd.DataFrame(columns=columns)

    data = _stock_universe(basic).merge(_daily_basic_use(daily_basic), on="ts_code", how="inner")
    data = data[data["total_mv"] > 0].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)

    data["total_market_cap_yi"] = data["total_mv"] / 10000
    data = data.sort_values(["total_market_cap_yi", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    data["rank"] = data.index + 1
    data["code"] = data["ts_code"].str.slice(0, 6)
    data["total_market_cap_yi"] = data["total_market_cap_yi"].round(2)
    data["close"] = pd.to_numeric(data.get("close", 0), errors="coerce").fillna(0).round(2)
    data["pct_chg"] = pd.to_numeric(data.get("pct_chg", 0), errors="coerce").fillna(0).round(2)
    return data[columns].copy()


def rank_stock_prices(
    basic: pd.DataFrame,
    price_data: pd.DataFrame,
    daily_basic: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rank individual stocks by latest price descending."""
    columns = [
        "rank",
        "code",
        "name",
        "stock_type",
        "industry",
        "close",
        "pct_chg",
        "total_market_cap_yi",
        "trade_date",
    ]
    if (
        basic.empty
        or price_data.empty
        or "ts_code" not in basic.columns
        or "ts_code" not in price_data.columns
        or "close" not in price_data.columns
    ):
        return pd.DataFrame(columns=columns)

    prices = price_data.copy()
    prices["ts_code"] = prices["ts_code"].astype(str)
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce").fillna(0)
    for col in ["pct_chg", "trade_date"]:
        if col not in prices.columns:
            prices[col] = 0 if col == "pct_chg" else ""
    prices["pct_chg"] = pd.to_numeric(prices["pct_chg"], errors="coerce").fillna(0)
    prices = prices[["ts_code", "close", "pct_chg", "trade_date"]].drop_duplicates("ts_code", keep="last")

    data = _stock_universe(basic).merge(prices, on="ts_code", how="inner")
    if daily_basic is not None and not daily_basic.empty and "total_mv" in daily_basic.columns:
        data = data.merge(_daily_basic_use(daily_basic)[["ts_code", "total_mv"]], on="ts_code", how="left")
        data["total_market_cap_yi"] = pd.to_numeric(data["total_mv"], errors="coerce").fillna(0) / 10000
    else:
        data["total_market_cap_yi"] = 0
    data = data[data["close"] > 0].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)

    data = data.sort_values(["close", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    data["rank"] = data.index + 1
    data["code"] = data["ts_code"].str.slice(0, 6)
    data["close"] = data["close"].round(2)
    data["pct_chg"] = data["pct_chg"].round(2)
    data["total_market_cap_yi"] = data["total_market_cap_yi"].round(2)
    return data[columns].copy()


def _stock_universe(basic: pd.DataFrame) -> pd.DataFrame:
    columns = ["ts_code", "symbol", "name", "industry", "stock_type"]
    data = basic.copy()
    for col in columns:
        if col not in data.columns:
            data[col] = ""
    data = data[columns].copy()
    data["ts_code"] = data["ts_code"].astype(str)
    data["symbol"] = data["symbol"].fillna(data["ts_code"].str.slice(0, 6)).astype(str).str.zfill(6)
    data["name"] = data["name"].fillna("").astype(str)
    data["industry"] = data["industry"].fillna("未分类").replace("", "未分类")
    data["stock_type"] = data["stock_type"].fillna("").astype(str)
    return data.drop_duplicates("ts_code", keep="first")


def _daily_basic_use(daily_basic: pd.DataFrame) -> pd.DataFrame:
    data = daily_basic.copy()
    data["ts_code"] = data["ts_code"].astype(str)
    data["total_mv"] = pd.to_numeric(data["total_mv"], errors="coerce").fillna(0)
    for col in ["close", "pct_chg", "trade_date"]:
        if col not in data.columns:
            data[col] = 0 if col in {"close", "pct_chg"} else ""
    data["close"] = pd.to_numeric(data["close"], errors="coerce").fillna(0)
    data["pct_chg"] = pd.to_numeric(data["pct_chg"], errors="coerce").fillna(0)
    data["trade_date"] = data["trade_date"].fillna("").astype(str)
    return data[["ts_code", "total_mv", "close", "pct_chg", "trade_date"]].drop_duplicates("ts_code", keep="last")


class IndustryMarketCapRankStrategy:
    """Market ranking tool for industries, stock market cap, and stock price."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.config = engine.config

    def log(self, message: str) -> None:
        self.engine.log(message)

    def run(self) -> dict[str, Any]:
        rank_type = str(
            getattr(self.config, "industry_market_cap_rank_type", "industry_market_cap")
            or "industry_market_cap"
        ).strip()
        if rank_type not in RANK_TYPE_LABELS:
            rank_type = "industry_market_cap"
        rank_type_label = RANK_TYPE_LABELS[rank_type]

        level = str(getattr(self.config, "industry_market_cap_rank_level", "sw_l2") or "sw_l2").strip()
        if level not in LEVEL_LABELS:
            level = "sw_l2"
        level_label = LEVEL_LABELS[level]

        self.log("读取交易日历...")
        trade_dates = self.engine._trade_dates()
        trade_date = trade_dates[-1]
        self.config.end_date = trade_date
        self.log(f"使用交易日: {trade_date}")

        self.log("读取股票基础信息...")
        basic = self.engine._stock_basic()

        daily_basic, market_cap_trade_date = self._latest_daily_basic(trade_dates)
        price_trade_date = market_cap_trade_date
        data_source = "盘后日线"

        if rank_type == "industry_market_cap":
            self.log(f"读取{level_label}行业成分...")
            members = self.engine._sw_industry_members(level)
            required = {"ts_code", "rank_industry", "rank_industry_code"}
            if members.empty or not required.issubset(members.columns):
                raise RuntimeError(f"{level_label}行业成分为空，无法计算行业总市值排名。")
            rows_df = rank_industry_market_caps(basic, daily_basic, members)
            count_label = "行业"
        else:
            quote_data, price_trade_date, data_source = self._stock_quote_data(
                basic,
                daily_basic,
                market_cap_trade_date,
                trade_date,
            )
            if rank_type == "stock_market_cap":
                rows_df = rank_stock_market_caps(basic, quote_data)
            else:
                rows_df = rank_stock_prices(basic, quote_data, daily_basic)
            count_label = "股票"

        rows = rows_df.to_dict("records")
        total_market_cap_yi = self._sum_column(rows_df, "total_market_cap_yi")
        self.log(f"{rank_type_label}完成，共 {len(rows)} 个{count_label}。")
        return {
            "ok": True,
            "trade_date": trade_date,
            "market_cap_trade_date": market_cap_trade_date,
            "price_trade_date": price_trade_date,
            "data_source": data_source,
            "rank_type": rank_type,
            "rank_type_label": rank_type_label,
            "rank_level": level,
            "rank_level_label": level_label,
            "count": len(rows),
            "total_market_cap_yi": total_market_cap_yi,
            "rows": rows,
            "logs": self.engine.logs,
        }

    def _latest_daily_basic(self, trade_dates: list[str]) -> tuple[pd.DataFrame, str]:
        for trade_date in reversed(trade_dates[-5:]):
            daily_basic = self.engine._daily_basic(trade_date)
            if not self.engine._daily_basic_is_empty(daily_basic):
                if trade_date != trade_dates[-1]:
                    self.log(f"{trade_dates[-1]} 市值数据为空，已回退使用 {trade_date}。")
                return daily_basic, trade_date
        raise RuntimeError("最近交易日总市值数据为空，无法计算行业总市值排名。")

    def _stock_quote_data(
        self,
        basic: pd.DataFrame,
        daily_basic: pd.DataFrame,
        market_cap_trade_date: str,
        trade_date: str,
    ) -> tuple[pd.DataFrame, str, str]:
        if self.config.review_mode == "intraday":
            realtime = self.engine._realtime_list()
            if not realtime.empty:
                normalized = self.engine._normalize_realtime(realtime, basic, trade_date)
                if not normalized.empty:
                    quote_data = self._apply_realtime_market_caps(daily_basic, normalized)
                    self.log(f"盘中模式：已读取 {len(normalized)} 条实时行情用于个股排名。")
                    return quote_data, trade_date, "盘中实时行情"
            self.log("盘中实时行情为空，退回最近日线收盘价排名。")

        daily = self.engine._daily_window([market_cap_trade_date])
        quote_data = daily_basic.merge(
            daily[["ts_code", "close", "pct_chg", "trade_date"]],
            on="ts_code",
            how="left",
            suffixes=("", "_daily"),
        )
        quote_data["close"] = pd.to_numeric(quote_data.get("close"), errors="coerce").fillna(0)
        quote_data["pct_chg"] = pd.to_numeric(quote_data.get("pct_chg"), errors="coerce").fillna(0)
        quote_data["trade_date"] = quote_data.get("trade_date", market_cap_trade_date)
        return quote_data, market_cap_trade_date, "盘后日线"

    @staticmethod
    def _apply_realtime_market_caps(daily_basic: pd.DataFrame, realtime: pd.DataFrame) -> pd.DataFrame:
        base = daily_basic.copy()
        base["ts_code"] = base["ts_code"].astype(str)
        base["total_mv"] = pd.to_numeric(base["total_mv"], errors="coerce").fillna(0)
        rt = realtime[["ts_code", "close", "pre_close", "pct_chg", "trade_date"]].copy()
        rt["ts_code"] = rt["ts_code"].astype(str)
        for col in ["close", "pre_close", "pct_chg"]:
            rt[col] = pd.to_numeric(rt[col], errors="coerce").fillna(0)
        data = base.merge(rt, on="ts_code", how="inner", suffixes=("", "_rt"))
        ratio = data["close"] / data["pre_close"].replace(0, pd.NA)
        data["total_mv"] = (data["total_mv"] * ratio).fillna(data["total_mv"])
        if "trade_date_rt" in data.columns:
            data["trade_date"] = data["trade_date_rt"]
        return data

    @staticmethod
    def _sum_column(data: pd.DataFrame, column: str) -> float:
        return round(
            float(pd.to_numeric(data.get(column, pd.Series(dtype="float64")), errors="coerce").fillna(0).sum()),
            2,
        )
