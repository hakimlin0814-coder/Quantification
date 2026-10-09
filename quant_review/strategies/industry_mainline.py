from __future__ import annotations

import math
from typing import Any

import pandas as pd


MAINLINE_CONFIRMED = "主线确认"
MAINLINE_CANDIDATE = "主线候选"
MAINLINE_NONE = "非主线"


def score_industry_mainlines(
    basic: pd.DataFrame,
    daily: pd.DataFrame,
    trade_dates: list[str],
    industry_members: pd.DataFrame | None = None,
    candidate_score: float = 60.0,
    confirmed_score: float = 75.0,
    min_members: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score industries using multi-period RPS, breadth, and persistence.

    Returns:
    - industry rows, sorted by mainline score.
    - stock-to-industry rows ready to merge into a stock strategy.
    """
    empty_industries = pd.DataFrame(
        columns=[
            "mainline_industry",
            "mainline_industry_code",
            "mainline_industry_source",
            "mainline_score",
            "mainline_state",
        ]
    )
    empty_stocks = pd.DataFrame(columns=["ts_code"])
    if basic.empty or daily.empty or not trade_dates:
        return empty_industries, empty_stocks

    stock_map = _industry_stock_map(basic, industry_members)
    if stock_map.empty:
        return empty_industries, empty_stocks

    close = (
        daily.pivot_table(index="ts_code", columns="trade_date", values="close", aggfunc="last")
        .reindex(columns=trade_dates)
        .apply(pd.to_numeric, errors="coerce")
    )
    close.index = close.index.astype(str)
    stock_map = stock_map[stock_map["ts_code"].isin(close.index)].drop_duplicates("ts_code", keep="first")
    if stock_map.empty:
        return empty_industries, empty_stocks

    industry_by_code = stock_map.set_index("ts_code")["mainline_industry"]
    current = _industry_rps_snapshot(close, industry_by_code, end_offset=0)
    previous_5 = _industry_rps_snapshot(close, industry_by_code, end_offset=5)
    previous_10 = _industry_rps_snapshot(close, industry_by_code, end_offset=10)
    if current.empty:
        return empty_industries, empty_stocks

    current = current.rename(
        columns={
            "industry_rps20": "industry_rps20",
            "industry_rps60": "industry_rps60",
            "industry_rps120": "industry_rps120",
            "industry_rps_combo": "industry_rps_combo",
        }
    )
    previous_5 = previous_5.rename(
        columns={
            "industry_rps20": "industry_rps20_prev5",
            "industry_rps60": "industry_rps60_prev5",
            "industry_rps120": "industry_rps120_prev5",
            "industry_rps_combo": "industry_rps_combo_prev5",
        }
    )
    previous_10 = previous_10.rename(
        columns={
            "industry_rps20": "industry_rps20_prev10",
            "industry_rps60": "industry_rps60_prev10",
            "industry_rps120": "industry_rps120_prev10",
            "industry_rps_combo": "industry_rps_combo_prev10",
        }
    )

    industries = current.merge(previous_5, on="mainline_industry", how="left")
    industries = industries.merge(previous_10, on="mainline_industry", how="left")
    industries = industries.merge(_industry_breadth(close, industry_by_code), on="mainline_industry", how="left")

    meta = (
        stock_map.groupby("mainline_industry", as_index=False)
        .agg(
            mainline_industry_code=("mainline_industry_code", "first"),
            mainline_industry_source=("mainline_industry_source", "first"),
            member_count=("ts_code", "nunique"),
        )
    )
    industries = industries.merge(meta, on="mainline_industry", how="left")
    industries = industries[industries["member_count"].fillna(0) >= max(1, int(min_members))].copy()
    if industries.empty:
        return empty_industries, empty_stocks

    for col in [
        "industry_rps_combo",
        "industry_rps_combo_prev5",
        "industry_rps_combo_prev10",
        "industry_breadth_score",
    ]:
        industries[col] = pd.to_numeric(industries[col], errors="coerce").fillna(0)

    industries["industry_persistence_score"] = (
        industries["industry_rps_combo"] * 0.20
        + industries["industry_rps_combo_prev5"] * 0.45
        + industries["industry_rps_combo_prev10"] * 0.35
    )
    industries["mainline_score"] = (
        industries["industry_rps_combo"] * 0.50
        + industries["industry_breadth_score"] * 0.30
        + industries["industry_persistence_score"] * 0.20
    ).clip(0, 100)

    industries["industry_rank"] = industries["mainline_score"].rank(method="min", ascending=False).astype(int)
    industries["industry_rank_prev5"] = (
        industries["industry_rps_combo_prev5"].rank(method="min", ascending=False).astype(int)
    )
    industries["industry_rank_change_5"] = industries["industry_rank_prev5"] - industries["industry_rank"]
    industries["mainline_state"] = industries.apply(
        lambda row: _mainline_state(row, candidate_score, confirmed_score),
        axis=1,
    )
    industries["mainline_reason"] = industries.apply(_mainline_reason, axis=1)
    industries = industries.sort_values(["mainline_score", "industry_rps_combo"], ascending=False).reset_index(drop=True)

    merge_cols = [
        "mainline_industry",
        "mainline_score",
        "mainline_state",
        "mainline_reason",
        "industry_rank",
        "industry_rank_prev5",
        "industry_rank_change_5",
        "industry_rps20",
        "industry_rps60",
        "industry_rps120",
        "industry_rps_combo",
        "industry_rps_combo_prev5",
        "industry_breadth_score",
        "industry_persistence_score",
        "industry_above_ma20_ratio",
        "industry_above_ma50_ratio",
        "industry_strong_ratio",
        "member_count",
    ]
    stock_scores = stock_map.merge(industries[merge_cols], on="mainline_industry", how="left")
    return industries, stock_scores


def _industry_stock_map(
    basic: pd.DataFrame,
    industry_members: pd.DataFrame | None,
) -> pd.DataFrame:
    base = basic.copy()
    if "ts_code" not in base.columns:
        return pd.DataFrame(columns=["ts_code"])
    base["ts_code"] = base["ts_code"].astype(str)
    if "industry" not in base.columns:
        base["industry"] = "未分类"
    base["industry"] = base["industry"].fillna("未分类").replace("", "未分类").astype(str)
    base = base[["ts_code", "industry"]].drop_duplicates("ts_code", keep="first")

    members = industry_members.copy() if industry_members is not None else pd.DataFrame()
    required = {"ts_code", "rank_industry"}
    if members.empty or not required.issubset(members.columns):
        result = base.rename(columns={"industry": "mainline_industry"})
        result["mainline_industry_code"] = result["mainline_industry"]
        result["mainline_industry_source"] = "基础行业"
        return result

    members["ts_code"] = members["ts_code"].astype(str)
    members["rank_industry"] = members["rank_industry"].fillna("").astype(str)
    if "rank_industry_code" not in members.columns:
        members["rank_industry_code"] = ""
    members["rank_industry_code"] = members["rank_industry_code"].fillna("").astype(str)
    members = members[["ts_code", "rank_industry", "rank_industry_code"]].drop_duplicates("ts_code", keep="first")

    result = base.merge(members, on="ts_code", how="left")
    matched = result["rank_industry"].fillna("").str.strip() != ""
    result["mainline_industry"] = result["rank_industry"].where(matched, result["industry"])
    result["mainline_industry_code"] = result["rank_industry_code"].where(matched, result["industry"])
    result["mainline_industry_source"] = matched.map({True: "申万行业", False: "基础行业"})
    return result[
        ["ts_code", "mainline_industry", "mainline_industry_code", "mainline_industry_source"]
    ]


def _industry_rps_snapshot(
    close: pd.DataFrame,
    industry_by_code: pd.Series,
    end_offset: int,
) -> pd.DataFrame:
    stock_returns = pd.DataFrame(index=close.index)
    for window in (20, 60, 120):
        stock_returns[f"ret{window}"] = _window_return(close, window, end_offset)
    stock_returns["mainline_industry"] = stock_returns.index.to_series().map(industry_by_code)
    stock_returns = stock_returns.dropna(subset=["mainline_industry"])
    if stock_returns.empty:
        return pd.DataFrame(columns=["mainline_industry"])

    grouped = stock_returns.groupby("mainline_industry", as_index=False).agg(
        industry_ret20=("ret20", "mean"),
        industry_ret60=("ret60", "mean"),
        industry_ret120=("ret120", "mean"),
    )
    for window in (20, 60, 120):
        grouped[f"industry_rps{window}"] = (
            grouped[f"industry_ret{window}"].rank(pct=True, method="average") * 100
        )
    grouped["industry_rps_combo"] = (
        grouped["industry_rps20"] * 0.40
        + grouped["industry_rps60"] * 0.35
        + grouped["industry_rps120"] * 0.25
    )
    return grouped[
        [
            "mainline_industry",
            "industry_rps20",
            "industry_rps60",
            "industry_rps120",
            "industry_rps_combo",
        ]
    ]


def _industry_breadth(close: pd.DataFrame, industry_by_code: pd.Series) -> pd.DataFrame:
    latest = close.iloc[:, -1]
    ma20 = close.iloc[:, -min(20, len(close.columns)) :].mean(axis=1)
    ma50 = close.iloc[:, -min(50, len(close.columns)) :].mean(axis=1)
    stock_rps20 = _window_return(close, 20, 0).rank(pct=True, method="average") * 100

    data = pd.DataFrame(
        {
            "mainline_industry": close.index.to_series().map(industry_by_code),
            "above_ma20": latest > ma20,
            "above_ma50": latest > ma50,
            "strong": stock_rps20 >= 80,
        },
        index=close.index,
    ).dropna(subset=["mainline_industry"])
    if data.empty:
        return pd.DataFrame(columns=["mainline_industry"])

    grouped = data.groupby("mainline_industry", as_index=False).agg(
        industry_above_ma20_ratio=("above_ma20", "mean"),
        industry_above_ma50_ratio=("above_ma50", "mean"),
        industry_strong_ratio=("strong", "mean"),
    )
    grouped["industry_breadth_score"] = (
        grouped["industry_above_ma20_ratio"] * 100 * 0.40
        + grouped["industry_above_ma50_ratio"] * 100 * 0.35
        + grouped["industry_strong_ratio"] * 100 * 0.25
    )
    return grouped


def _window_return(close: pd.DataFrame, window: int, end_offset: int) -> pd.Series:
    if close.empty or len(close.columns) <= end_offset:
        return pd.Series(float("nan"), index=close.index, dtype="float64")
    end_pos = len(close.columns) - 1 - end_offset
    base_pos = max(0, end_pos - window)
    end_price = pd.to_numeric(close.iloc[:, end_pos], errors="coerce")
    base_price = pd.to_numeric(close.iloc[:, base_pos], errors="coerce")
    result = (end_price / base_price - 1) * 100
    valid = (end_price > 0) & (base_price > 0)
    return result.where(valid).replace([math.inf, -math.inf], pd.NA)


def _mainline_state(row: pd.Series, candidate_score: float, confirmed_score: float) -> str:
    score = _number(row.get("mainline_score"))
    breadth = _number(row.get("industry_breadth_score"))
    previous = _number(row.get("industry_rps_combo_prev5"))
    if score >= confirmed_score and breadth >= 55 and previous >= 65:
        return MAINLINE_CONFIRMED
    if score >= candidate_score:
        return MAINLINE_CANDIDATE
    return MAINLINE_NONE


def _mainline_reason(row: pd.Series) -> str:
    parts = [
        f"行业RPS{_number(row.get('industry_rps_combo')):.1f}",
        f"广度{_number(row.get('industry_breadth_score')):.1f}",
        f"持续{_number(row.get('industry_persistence_score')):.1f}",
    ]
    change = int(_number(row.get("industry_rank_change_5")))
    if change > 0:
        parts.append(f"5日排名升{change}")
    elif change < 0:
        parts.append(f"5日排名降{abs(change)}")
    else:
        parts.append("5日排名持平")
    return "；".join(parts)


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(number) else number
