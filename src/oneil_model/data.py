"""Tushare/AkShare data adaptation for 05 §17 and 09 §4–§6."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from typing import Any

import pandas as pd

from .config import OneilResearchConfig
from .models import DataSnapshot


INDEX_CODES = ("000001.SH", "399001.SZ", "399006.SZ", "000688.SH")


def report_periods(as_of_date: str, count: int = 12) -> list[str]:
    year = int(as_of_date[:4])
    periods = [
        f"{candidate_year}{suffix}"
        for candidate_year in range(year - 5, year + 1)
        for suffix in ("0331", "0630", "0930", "1231")
        if f"{candidate_year}{suffix}" <= as_of_date
    ]
    return sorted(periods, reverse=True)[:count]


class OneilDataGateway:
    """Tushare-first, AkShare-fallback gateway.

    Tushare supplies calendar, security master, daily bars, daily basics,
    financial indicators, limits and indices. AkShare/Eastmoney is only a
    financial/basic fallback and is marked with point-in-time revision risk.
    """

    def __init__(self, engine: Any, config: OneilResearchConfig) -> None:
        self.engine = engine
        self.config = config
        self.notes: list[str] = []

    def load(self) -> DataSnapshot:
        old_lookback = int(self.engine.config.lookback_days)
        self.engine.config.lookback_days = max(old_lookback, self.config.rs_lookback + 5)
        try:
            trade_dates = self.engine._trade_dates()
        finally:
            self.engine.config.lookback_days = old_lookback
        as_of_date = trade_dates[-1]
        self.engine.config.end_date = as_of_date
        basic, basic_source, survivor_risk = self._stock_basic_as_of(as_of_date)
        daily = self.engine._daily_window(trade_dates)
        daily_basic = self.engine._daily_basic(as_of_date)
        financials, financial_source = self._financials(as_of_date)
        market_indices = self._market_indices(trade_dates)
        limit_list = self.engine._limit_list(as_of_date)
        return DataSnapshot(
            as_of_date=as_of_date,
            trade_dates=trade_dates,
            basic=basic,
            daily=daily,
            daily_basic=daily_basic,
            financials=financials,
            market_indices=market_indices,
            limit_list=limit_list,
            metadata={
                "primary_source": "Tushare",
                "fallback_source": "AkShare/Eastmoney",
                "basic_source": basic_source,
                "financial_source": financial_source,
                "survivorship_risk": survivor_risk,
                "notes": list(self.notes),
            },
        )

    def _stock_basic_as_of(self, as_of_date: str) -> tuple[pd.DataFrame, str, bool]:
        fields = "ts_code,symbol,name,area,industry,market,list_date,delist_date"
        frames: list[pd.DataFrame] = []
        retrieval_date = datetime.now().strftime("%Y%m%d")
        for status in ("L", "D", "P"):
            try:
                frame = self.engine._cached(
                    f"oneil_stock_basic_{status}_{retrieval_date}",
                    lambda status=status: self.engine.pro.stock_basic(
                        exchange="", list_status=status, fields=fields
                    ),
                )
            except Exception as exc:
                self.notes.append(f"Tushare stock_basic {status}失败: {exc}")
                continue
            if frame is not None and not frame.empty and "ts_code" in frame.columns:
                frame = frame.copy()
                frame["list_status"] = status
                frames.append(frame)
        if frames:
            data = pd.concat(frames, ignore_index=True)
            for col in ["list_date", "delist_date"]:
                if col not in data.columns:
                    data[col] = ""
                data[col] = data[col].fillna("").astype(str).str.replace(r"\D", "", regex=True).str[:8]
            data = data[
                (data["list_date"] != "")
                & (data["list_date"] <= as_of_date)
                & ((data["delist_date"] == "") | (data["delist_date"] > as_of_date))
            ].copy()
            data["ts_code"] = data["ts_code"].astype(str)
            data["symbol"] = data.get("symbol", data["ts_code"].str[:6]).astype(str).str.zfill(6)
            data["industry"] = data.get("industry", "未分类").fillna("未分类").replace("", "未分类")
            data["stock_type"] = self.engine._stock_type_series(data)
            data = data.drop_duplicates("ts_code", keep="last")
            return self._apply_universe_rules(data), "Tushare point-in-time listing/delisting", False

        fallback = self._akshare_stock_basic()
        if fallback.empty:
            raise RuntimeError("Tushare与AkShare均未返回股票基础信息")
        self.notes.append("AkShare基础名单通常只含当前股票，存在生存者偏差，仅用于展示")
        return self._apply_universe_rules(fallback), "AkShare current-list fallback", True

    def _akshare_stock_basic(self) -> pd.DataFrame:
        try:
            import akshare as ak
        except Exception as exc:
            self.notes.append(f"AkShare导入失败: {exc}")
            return pd.DataFrame()
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                frame = ak.stock_info_a_code_name()
        except Exception as exc:
            self.notes.append(f"AkShare股票名单失败: {exc}")
            return pd.DataFrame()
        if frame is None or frame.empty or "code" not in frame.columns:
            return pd.DataFrame()
        data = frame.rename(columns={"code": "symbol", "name": "name"}).copy()
        data["symbol"] = data["symbol"].astype(str).str.zfill(6)
        data["ts_code"] = data["symbol"].map(self._symbol_to_ts_code)
        data["industry"] = "未分类"
        data["market"] = ""
        data["list_date"] = ""
        data["delist_date"] = ""
        data["stock_type"] = self.engine._stock_type_series(data)
        return data

    def _apply_universe_rules(self, basic: pd.DataFrame) -> pd.DataFrame:
        data = basic.copy()
        symbol = data["symbol"].astype(str).str.zfill(6)
        allowed = ~symbol.str.startswith(("8", "4", "92")) if not self.config.include_bj else pd.Series(True, index=data.index)
        if not self.config.include_chinext:
            allowed &= ~symbol.str.startswith(("300", "301"))
        if not self.config.include_star:
            allowed &= ~symbol.str.startswith(("688", "689"))
        if self.config.exclude_st:
            allowed &= ~data["name"].fillna("").astype(str).str.upper().str.contains("ST")
        data = data[allowed].copy()
        data["universe_source"] = "A"
        return data

    def _financials(self, as_of_date: str) -> tuple[pd.DataFrame, str]:
        frames: list[pd.DataFrame] = []
        for period in report_periods(as_of_date):
            try:
                frame = self.engine._fina_indicator(period)
            except Exception as exc:
                self.notes.append(f"Tushare {period}财务失败: {exc}")
                continue
            if frame is None or frame.empty or "ts_code" not in frame.columns:
                continue
            frame = frame.copy()
            frame["data_source"] = "Tushare fina_indicator"
            frame["point_in_time_risk"] = False
            frames.append(frame)
        if frames:
            return pd.concat(frames, ignore_index=True), "Tushare"
        fallback = self._akshare_financials(as_of_date)
        if fallback.empty:
            self.notes.append("Tushare财务无数据，AkShare兜底也无数据")
            return pd.DataFrame(), "unavailable"
        self.notes.append("财务使用AkShare/Eastmoney兜底；可能含后续修订，只展示不形成PASS")
        return fallback, "AkShare/Eastmoney fallback (revision risk)"

    def _akshare_financials(self, as_of_date: str) -> pd.DataFrame:
        try:
            import akshare as ak
        except Exception as exc:
            self.notes.append(f"AkShare财务兜底导入失败: {exc}")
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        for period in report_periods(as_of_date):
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    frame = ak.stock_yjbb_em(date=period)
            except Exception as exc:
                self.notes.append(f"AkShare {period}业绩报表失败: {exc}")
                continue
            normalized = self._normalize_akshare_financial(frame, period)
            if not normalized.empty:
                frames.append(normalized)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @classmethod
    def _normalize_akshare_financial(cls, frame: pd.DataFrame, period: str) -> pd.DataFrame:
        if frame is None or frame.empty or "股票代码" not in frame.columns:
            return pd.DataFrame()
        data = frame.copy()
        data["symbol"] = data["股票代码"].astype(str).str.zfill(6)
        data["ts_code"] = data["symbol"].map(cls._symbol_to_ts_code)
        rename = {
            "每股收益": "eps",
            "营业总收入-同比增长": "q_sales_yoy",
            "净资产收益率": "roe",
            "最新公告日期": "ann_date",
        }
        data = data.rename(columns={old: new for old, new in rename.items() if old in data.columns})
        for col in ["eps", "q_sales_yoy", "roe"]:
            if col not in data.columns:
                data[col] = pd.NA
            data[col] = pd.to_numeric(data[col], errors="coerce")
        if "ann_date" not in data.columns:
            data["ann_date"] = ""
        data["end_date"] = period
        data["q_eps"] = data["eps"]
        data["basic_eps_yoy"] = pd.NA
        data["data_source"] = "AkShare stock_yjbb_em"
        data["point_in_time_risk"] = True
        return data[
            [
                "ts_code",
                "ann_date",
                "end_date",
                "eps",
                "q_eps",
                "basic_eps_yoy",
                "q_sales_yoy",
                "roe",
                "data_source",
                "point_in_time_risk",
            ]
        ]

    def _market_indices(self, trade_dates: list[str]) -> dict[str, pd.DataFrame]:
        start_date, end_date = trade_dates[0], trade_dates[-1]
        result: dict[str, pd.DataFrame] = {}
        for code in INDEX_CODES:
            try:
                result[code] = self.engine._cached(
                    f"oneil_index_{code}_{start_date}_{end_date}",
                    lambda code=code: self.engine.pro.index_daily(
                        ts_code=code, start_date=start_date, end_date=end_date
                    ),
                )
            except Exception as exc:
                self.notes.append(f"指数{code}读取失败: {exc}")
                result[code] = pd.DataFrame()
        return result

    @staticmethod
    def _symbol_to_ts_code(symbol: str) -> str:
        value = str(symbol).zfill(6)
        if value.startswith(("6", "5", "9")):
            return f"{value}.SH"
        if value.startswith(("8", "4")):
            return f"{value}.BJ"
        return f"{value}.SZ"
