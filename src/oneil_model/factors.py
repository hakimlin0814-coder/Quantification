"""Point-in-time C/A/N/S/L/M factors specified by 09 §4 and §8–§15."""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from .config import OneilResearchConfig


def _date8(value: Any) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    return digits[:8] if len(digits) >= 8 else ""


def _number(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        result = float(value)
        return None if math.isnan(result) or math.isinf(result) else result
    except (TypeError, ValueError):
        return None


def filter_financials_as_of(financials: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    """Keep only records announced by the research cut-off.

    Source: A-share adaptation, 09 §4. Missing announcement dates are excluded;
    report period is never used as a substitute for availability time.
    """
    if financials is None or financials.empty:
        return pd.DataFrame()
    data = financials.copy()
    if "ann_date" not in data.columns:
        return pd.DataFrame(columns=data.columns)
    data["ann_date"] = data["ann_date"].map(_date8)
    data["end_date"] = data.get("end_date", "").map(_date8)
    data = data[(data["ann_date"] != "") & (data["ann_date"] <= _date8(as_of_date))]
    if data.empty:
        return data
    # Later corrections only become visible from their own announcement date.
    return (
        data.sort_values(["ts_code", "end_date", "ann_date"])
        .groupby(["ts_code", "end_date"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def compute_financial_factors(
    financials: pd.DataFrame,
    as_of_date: str,
    config: OneilResearchConfig,
) -> pd.DataFrame:
    data = filter_financials_as_of(financials, as_of_date)
    columns = [
        "ts_code",
        "report_date",
        "max_ann_date",
        "eps_yoy_q",
        "sales_yoy_q",
        "latest_eps",
        "annual_eps_up",
        "annual_eps_cagr",
        "annual_roe",
        "financial_complete",
        "financial_source",
        "financial_point_in_time_risk",
    ]
    if data.empty or "ts_code" not in data.columns:
        return pd.DataFrame(columns=columns)

    numeric = [
        "eps",
        "q_eps",
        "basic_eps_yoy",
        "dt_eps_yoy",
        "q_sales_yoy",
        "or_yoy",
        "roe",
        "roe_dt",
    ]
    for col in numeric:
        if col not in data.columns:
            data[col] = pd.NA
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if "data_source" not in data.columns:
        data["data_source"] = "unknown"
    if "point_in_time_risk" not in data.columns:
        data["point_in_time_risk"] = False

    rows: list[dict[str, Any]] = []
    for ts_code, group in data.groupby("ts_code", sort=False):
        group = group.sort_values(["end_date", "ann_date"])
        latest = group.iloc[-1]
        latest_eps = _number(latest.get("q_eps"))
        if latest_eps is None:
            latest_eps = _number(latest.get("eps"))
        eps_yoy = _number(latest.get("basic_eps_yoy"))
        if eps_yoy is None:
            eps_yoy = _number(latest.get("dt_eps_yoy"))
        sales_yoy = _number(latest.get("q_sales_yoy"))
        if sales_yoy is None:
            sales_yoy = _number(latest.get("or_yoy"))

        annual = group[group["end_date"].astype(str).str.endswith("1231")].copy()
        annual = annual.sort_values(["end_date", "ann_date"]).groupby("end_date", as_index=False).tail(1)
        annual = annual.tail(4)
        annual_eps = [_number(value) for value in annual["eps"].tolist()]
        annual_eps_valid = len(annual_eps) == 4 and all(value is not None and value > 0 for value in annual_eps)
        annual_up: bool | None = None
        annual_cagr: float | None = None
        if annual_eps_valid:
            values = [float(value) for value in annual_eps if value is not None]
            annual_up = all(right > left for left, right in zip(values, values[1:]))
            annual_cagr = ((values[-1] / values[0]) ** (1 / 3) - 1) * 100
        latest_annual = annual.iloc[-1] if not annual.empty else None
        annual_roe = _number(latest_annual.get("roe")) if latest_annual is not None else None
        if annual_roe is None and latest_annual is not None:
            annual_roe = _number(latest_annual.get("roe_dt"))

        positive_ok = latest_eps is not None and (latest_eps > 0 or not config.require_positive_eps)
        complete = all(value is not None for value in [eps_yoy, sales_yoy, annual_cagr, annual_roe]) and annual_up is not None and positive_ok
        sources = sorted({str(value) for value in group["data_source"].dropna().tolist() if str(value)})
        point_in_time_risk = bool(group["point_in_time_risk"].fillna(False).astype(bool).any())
        if point_in_time_risk:
            # AkShare/Eastmoney fallback can contain subsequently revised history.
            complete = False
        rows.append(
            {
                "ts_code": str(ts_code),
                "report_date": str(latest.get("end_date", "")),
                "max_ann_date": str(group["ann_date"].max()),
                "eps_yoy_q": eps_yoy,
                "sales_yoy_q": sales_yoy,
                "latest_eps": latest_eps,
                "annual_eps_up": annual_up,
                "annual_eps_cagr": annual_cagr,
                "annual_roe": annual_roe,
                "financial_complete": bool(complete),
                "financial_source": "+".join(sources) or "unknown",
                "financial_point_in_time_risk": point_in_time_risk,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _synthetic_adjusted_path(group: pd.DataFrame) -> pd.DataFrame:
    """Build a point-in-time continuity index from Tushare daily pct_chg.

    This is an A-share implementation hypothesis. It avoids treating an ex-right
    raw-price jump as momentum while preserving raw OHLC for execution checks.
    """
    data = group.sort_values("trade_date").copy()
    pct = pd.to_numeric(data.get("pct_chg"), errors="coerce")
    close = pd.to_numeric(data.get("close"), errors="coerce")
    pre_close = pd.to_numeric(data.get("pre_close"), errors="coerce")
    fallback = (close / pre_close.replace(0, pd.NA) - 1) * 100
    pct = pct.combine_first(fallback).fillna(0).clip(lower=-99.0)
    synthetic_close = (1 + pct / 100).cumprod()
    raw_high = pd.to_numeric(data.get("high"), errors="coerce")
    high_ratio = (raw_high / close.replace(0, pd.NA)).fillna(1.0).clip(lower=1.0)
    data["research_close"] = synthetic_close
    data["research_high"] = synthetic_close * high_ratio
    return data


def compute_price_factors(daily: pd.DataFrame, config: OneilResearchConfig) -> pd.DataFrame:
    columns = [
        "ts_code",
        "price_date",
        "close",
        "high",
        "low",
        "pct_chg",
        "vol",
        "amount",
        "history_days",
        "return_rs_window",
        "rs_percentile",
        "near_high_ratio",
        "new_high_breakout",
        "breakout_rvol",
        "price_complete",
        "suspended_latest",
        "research_price_method",
    ]
    if daily is None or daily.empty or "ts_code" not in daily.columns:
        return pd.DataFrame(columns=columns)
    data = daily.copy()
    data["trade_date"] = data["trade_date"].astype(str)
    latest_date = str(data["trade_date"].max())
    rows: list[dict[str, Any]] = []
    for ts_code, raw_group in data.groupby("ts_code", sort=False):
        group = _synthetic_adjusted_path(raw_group)
        latest = group.iloc[-1]
        history_days = int(group["research_close"].notna().sum())
        lookback = min(config.rs_lookback, history_days)
        window = group.tail(lookback)
        first = _number(window.iloc[0].get("research_close")) if not window.empty else None
        last = _number(window.iloc[-1].get("research_close")) if not window.empty else None
        rs_return = ((last / first) - 1) * 100 if first and last is not None and lookback >= config.min_history_days else None
        rolling_high = _number(window["research_high"].max()) if not window.empty else None
        near_high = last / rolling_high if last is not None and rolling_high else None
        prior = group.iloc[:-1].tail(config.rs_lookback)
        prior_high = _number(prior["research_high"].max()) if not prior.empty else None
        new_high = bool(last is not None and prior_high is not None and last >= prior_high)
        volume = pd.to_numeric(group.get("vol"), errors="coerce")
        prior_volume = volume.iloc[:-1].tail(config.rvol_lookback)
        mean_volume = _number(prior_volume.mean()) if len(prior_volume) >= config.rvol_lookback else None
        latest_volume = _number(volume.iloc[-1]) if len(volume) else None
        rvol = latest_volume / mean_volume if latest_volume is not None and mean_volume else None
        suspended = str(latest.get("trade_date", "")) != latest_date
        complete = bool(
            not suspended
            and history_days >= config.min_history_days
            and rs_return is not None
            and near_high is not None
            and rvol is not None
        )
        rows.append(
            {
                "ts_code": str(ts_code),
                "price_date": str(latest.get("trade_date", "")),
                "close": _number(latest.get("close")),
                "high": _number(latest.get("high")),
                "low": _number(latest.get("low")),
                "pct_chg": _number(latest.get("pct_chg")),
                "vol": latest_volume,
                "amount": _number(latest.get("amount")),
                "history_days": history_days,
                "return_rs_window": rs_return,
                "rs_percentile": None,
                "near_high_ratio": near_high,
                "new_high_breakout": new_high,
                "breakout_rvol": rvol,
                "price_complete": complete,
                "suspended_latest": suspended,
                "research_price_method": "tushare_pct_chg_continuity_index",
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    eligible = result["return_rs_window"].notna()
    if eligible.any():
        result.loc[eligible, "rs_percentile"] = (
            result.loc[eligible, "return_rs_window"].rank(method="average", pct=True) * 100
        )
    return result


def compute_market_context(
    market_indices: dict[str, pd.DataFrame],
    daily: pd.DataFrame,
    limit_list: pd.DataFrame | None,
) -> dict[str, Any]:
    """Descriptive M and A-share sentiment; neither gates entries in Q0 v1."""
    index_rows: list[dict[str, Any]] = []
    for code, frame in market_indices.items():
        if frame is None or frame.empty:
            index_rows.append({"index_code": code, "status": "INCOMPLETE", "distribution_count_25": None})
            continue
        data = frame.sort_values("trade_date").copy()
        for col in ["pct_chg", "vol", "amount"]:
            data[col] = pd.to_numeric(data.get(col), errors="coerce")
        activity_col = "amount" if data["amount"].notna().any() else "vol"
        distribution = (data["pct_chg"] <= -0.2) & (data[activity_col] > data[activity_col].shift(1))
        index_rows.append(
            {
                "index_code": code,
                "status": "DISPLAY_ONLY",
                "activity_field": activity_col,
                "distribution_count_25": int(distribution.tail(25).sum()),
                "last_date": str(data.iloc[-1].get("trade_date", "")),
                "last_pct_chg": _number(data.iloc[-1].get("pct_chg")),
                "follow_through": "NOT_EVALUATED",
            }
        )

    breadth = None
    latest_date = ""
    if daily is not None and not daily.empty:
        latest_date = str(daily["trade_date"].astype(str).max())
        latest = daily[daily["trade_date"].astype(str) == latest_date]
        pct = pd.to_numeric(latest.get("pct_chg"), errors="coerce").dropna()
        if len(pct):
            breadth = float((pct > 0).mean() * 100)
    up_count = down_count = 0
    if limit_list is not None and not limit_list.empty:
        limit_col = "limit" if "limit" in limit_list.columns else "limit_type" if "limit_type" in limit_list.columns else ""
        if limit_col:
            values = limit_list[limit_col].astype(str).str.upper()
            up_count = int(values.isin(["U", "UP", "涨停"]).sum())
            down_count = int(values.isin(["D", "DOWN", "跌停"]).sum())
    return {
        "m_source": "O/P/A",
        "m_gate": "NOT_EVALUATED",
        "indices": index_rows,
        "sentiment": {
            "source": "A",
            "mode": "DISPLAY_ONLY",
            "trade_date": latest_date,
            "advance_ratio_pct": breadth,
            "limit_up_count": up_count,
            "limit_down_count": down_count,
        },
    }
