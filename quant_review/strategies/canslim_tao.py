from __future__ import annotations

import math
import io
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from typing import Any

import pandas as pd

from quant_review.strategies.industry_mainline import (
    MAINLINE_CANDIDATE,
    MAINLINE_CONFIRMED,
    score_industry_mainlines,
)


class CanslimTaoStrategy:
    """CANSLIM + Tao-style RPS trend growth strategy.

    The strategy is an observation pool, not an automatic buy signal. It blends:
    - CANSLIM growth, new high, supply-demand, leadership, sponsorship proxy, market direction.
    - Tao-style RPS50/RPS120/RPS250 strength and 250-day high proximity.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.config = engine.config
        self.fundamental_status = "未读取"
        self.fundamental_note = ""
        self.fundamental_rows = 0
        self.institution_status = "未读取"
        self.institution_note = ""
        self.institution_rows = 0

    def log(self, message: str) -> None:
        self.engine.log(message)

    def run(self) -> dict[str, Any]:
        old_lookback = int(self.config.lookback_days)
        self.config.lookback_days = max(old_lookback, 270)
        try:
            self.log("读取交易日历，CANSLIM/RPS策略需要约270个交易日...")
            trade_dates = self.engine._trade_dates()
        finally:
            self.config.lookback_days = old_lookback

        trade_date = trade_dates[-1]
        self.config.end_date = trade_date
        self.log(f"使用交易日: {trade_date}")

        self.log("读取股票基础信息...")
        basic = self.engine._stock_basic()

        self.log("读取近一年日线行情...")
        daily = self.engine._daily_window(trade_dates)
        if self.config.review_mode == "intraday":
            self.log("盘中模式：读取实时行情并合成临时日K...")
            daily = self.engine._apply_intraday_snapshot(daily, basic, trade_date)

        self.log("读取每日指标...")
        daily_basic = self.engine._daily_basic(trade_date)
        if self.config.review_mode == "intraday" and self.engine._daily_basic_is_empty(daily_basic) and len(trade_dates) >= 2:
            prev_date = trade_dates[-2]
            self.log(f"盘中模式：当日每日指标未更新，回退到 {prev_date}。")
            daily_basic = self.engine._daily_basic(prev_date)

        self.log("计算市场方向...")
        market = self._market_direction(daily, trade_dates)

        fundamentals = pd.DataFrame()
        if self.config.canslim_include_fundamental:
            self.log("读取财务成长指标...")
            fundamentals = self._fundamental_scores(trade_date)
        else:
            self.fundamental_status = "已关闭"
            self.fundamental_note = "页面关闭了财务指标读取"
            self.log("已关闭财务指标读取，C/A按中性分处理。")

        self.log("计算RPS、均线结构和买点形态...")
        features = self._price_features(daily, trade_dates)

        self.log("计算行业主线强度...")
        mainline_industries, mainline_stocks = self._industry_mainline_scores(
            basic,
            daily,
            trade_dates,
        )

        self.log("读取陶博士通道机构持仓条件...")
        institutions = self._institutional_scores(trade_date, features)

        self.log("合成CANSLIM+RPS候选池...")
        rows = self._score_rows(
            basic,
            daily_basic,
            features,
            fundamentals,
            institutions,
            market,
            mainline_stocks,
        )
        probe = self._probe(getattr(self, "_last_canslim_data", pd.DataFrame()))

        self.log(f"CANSLIM+RPS策略完成，共 {len(rows)} 只。")
        return {
            "ok": True,
            "trade_date": trade_date,
            "market": market,
            "financial": {
                "status": self.fundamental_status,
                "note": self.fundamental_note,
                "rows": self.fundamental_rows,
            },
            "institutional": {
                "status": self.institution_status,
                "note": self.institution_note,
                "rows": self.institution_rows,
            },
            "mainline": self._mainline_payload(mainline_industries),
            "rows": rows,
            "probe": probe,
            "logs": self.engine.logs,
        }

    def _price_features(self, daily: pd.DataFrame, trade_dates: list[str]) -> pd.DataFrame:
        close = daily.pivot_table(index="ts_code", columns="trade_date", values="close", aggfunc="last").reindex(columns=trade_dates)
        high = daily.pivot_table(index="ts_code", columns="trade_date", values="high", aggfunc="last").reindex(columns=trade_dates)
        low = daily.pivot_table(index="ts_code", columns="trade_date", values="low", aggfunc="last").reindex(columns=trade_dates)
        open_ = daily.pivot_table(index="ts_code", columns="trade_date", values="open", aggfunc="last").reindex(columns=trade_dates)
        amount = daily.pivot_table(index="ts_code", columns="trade_date", values="amount", aggfunc="last").reindex(columns=trade_dates)
        pct = daily.pivot_table(index="ts_code", columns="trade_date", values="pct_chg", aggfunc="last").reindex(columns=trade_dates)

        latest = trade_dates[-1]
        features = pd.DataFrame(index=close.index)
        features["ts_code"] = features.index.astype(str)
        features["close"] = close[latest]
        features["open"] = open_[latest]
        features["high"] = high[latest]
        features["low"] = low[latest]
        features["pct_chg"] = pct[latest].fillna(0)
        features["amount_yi"] = amount[latest].fillna(0) / 100000

        for window in [5, 10, 20, 50, 120, 200, 250]:
            take = min(window, len(trade_dates))
            features[f"ma{window}"] = close.iloc[:, -take:].mean(axis=1)

        for window in [20, 50, 120, 250]:
            base = close.iloc[:, -(window + 1)] if len(trade_dates) > window else close.iloc[:, 0]
            ret = (features["close"] / base - 1) * 100
            features[f"ret{window}"] = ret.replace([math.inf, -math.inf], pd.NA)
            features[f"rps{window}"] = features[f"ret{window}"].rank(pct=True) * 100

        features["rps_combo"] = (
            features["rps50"].fillna(0) * 0.25
            + features["rps120"].fillna(0) * 0.35
            + features["rps250"].fillna(0) * 0.40
        )
        features["high250"] = high.iloc[:, -min(250, len(trade_dates)) :].max(axis=1)
        features["low250"] = low.iloc[:, -min(250, len(trade_dates)) :].min(axis=1)
        features["high_position_250"] = (features["close"] / features["high250"]).replace([math.inf, -math.inf], pd.NA)
        features["new_high_5"] = high.iloc[:, -min(5, len(trade_dates)) :].max(axis=1) >= features["high250"] * 0.995
        features["amount_ratio"] = amount.iloc[:, -min(3, len(trade_dates)) :].mean(axis=1) / amount.iloc[:, -min(20, len(trade_dates)) :].mean(axis=1)

        down_amount = amount.where(close.diff(axis=1) < 0)
        max_down_amount_10 = down_amount.iloc[:, -min(10, len(trade_dates)) :].max(axis=1).fillna(0)
        near_ma = (
            (features["low"] <= features["ma10"] * 1.03)
            | (features["low"] <= features["ma20"] * 1.03)
            | (features["low"] <= features["ma50"] * 1.03)
        )
        features["pocket_pivot"] = (
            (features["pct_chg"] >= 3)
            & (features["close"] >= features["open"])
            & (amount[latest].fillna(0) > max_down_amount_10)
            & near_ma
        )
        features["ma_stack"] = (
            (features["close"] > features["ma50"])
            & (features["ma50"] > features["ma120"])
            & (features["ma120"] > features["ma200"])
        )
        features["distance_ma5"] = (features["close"] / features["ma5"] - 1) * 100
        features["distance_ma10"] = (features["close"] / features["ma10"] - 1) * 100
        features["distance_ma20"] = (features["close"] / features["ma20"] - 1) * 100
        features["distance_ma50"] = (features["close"] / features["ma50"] - 1) * 100
        features["breakout_20"] = features["close"] >= high.iloc[:, -min(20, len(trade_dates)) :].max(axis=1) * 0.995
        return features.reset_index(drop=True)

    def _industry_mainline_scores(
        self,
        basic: pd.DataFrame,
        daily: pd.DataFrame,
        trade_dates: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not bool(getattr(self.config, "canslim_mainline_enabled", True)):
            self.log("行业主线评分已关闭。")
            return pd.DataFrame(), pd.DataFrame()

        level = str(getattr(self.config, "canslim_mainline_industry_level", "sw_l2") or "sw_l2").strip()
        members = pd.DataFrame()
        if level in {"sw_l1", "sw_l2", "sw_l3"}:
            members = self.engine._sw_industry_members(level)
        try:
            industries, stocks = score_industry_mainlines(
                basic=basic,
                daily=daily,
                trade_dates=trade_dates,
                industry_members=members,
            )
        except Exception as exc:
            self.log(f"行业主线评分失败，按中性分处理: {exc}")
            return pd.DataFrame(), pd.DataFrame()
        confirmed = int((industries.get("mainline_state", pd.Series(dtype=str)) == MAINLINE_CONFIRMED).sum())
        candidate = int((industries.get("mainline_state", pd.Series(dtype=str)) == MAINLINE_CANDIDATE).sum())
        self.log(f"行业主线评分完成：确认 {confirmed} 个，候选 {candidate} 个。")
        return industries, stocks

    def _mainline_payload(self, industries: pd.DataFrame) -> dict[str, Any]:
        enabled = bool(getattr(self.config, "canslim_mainline_enabled", True))
        if not enabled:
            return {
                "enabled": False,
                "industry_level": str(getattr(self.config, "canslim_mainline_industry_level", "sw_l2")),
                "confirmed_count": 0,
                "candidate_count": 0,
                "rows": [],
            }
        if industries.empty:
            return {
                "enabled": True,
                "industry_level": str(getattr(self.config, "canslim_mainline_industry_level", "sw_l2")),
                "confirmed_count": 0,
                "candidate_count": 0,
                "rows": [],
            }
        return {
            "enabled": True,
            "industry_level": str(getattr(self.config, "canslim_mainline_industry_level", "sw_l2")),
            "confirmed_count": int((industries["mainline_state"] == MAINLINE_CONFIRMED).sum()),
            "candidate_count": int((industries["mainline_state"] == MAINLINE_CANDIDATE).sum()),
            "rows": industries.head(15).fillna("").to_dict("records"),
        }

    def _fundamental_scores(self, trade_date: str) -> pd.DataFrame:
        periods = self._report_periods(trade_date)
        if not periods:
            self.fundamental_status = "无报告期"
            self.fundamental_note = "未推导出可读取的报告期"
            return pd.DataFrame()

        field_sets = [
            "ts_code,ann_date,end_date,eps,dt_eps,q_eps,basic_eps_yoy,dt_eps_yoy,roe,roe_dt,or_yoy,q_sales_yoy,netprofit_yoy,dt_netprofit_yoy,q_profit_yoy,q_netprofit_yoy,grossprofit_margin",
            "ts_code,ann_date,end_date,eps,dt_eps,basic_eps_yoy,dt_eps_yoy,roe,or_yoy,q_sales_yoy,netprofit_yoy,q_profit_yoy",
            "",
        ]
        frames = []
        failures = []
        for period in periods:
            frame = self._fetch_fina_indicator_period(period, field_sets, failures)
            if not frame.empty and "ts_code" in frame.columns:
                frames.append(frame)

        if not frames:
            fallback = self._eastmoney_financial_scores(trade_date, failures)
            if not fallback.empty:
                return fallback
            self.fundamental_status = "读取失败"
            self.fundamental_note = "fina_indicator无数据或无权限"
            if failures:
                self.fundamental_note = failures[-1][:120]
            self.log(f"财务指标为空，C/A按中性处理。原因: {self.fundamental_note}")
            self.log("财务指标为空，C/A/I按中性处理。")
            return pd.DataFrame()

        data = pd.concat(frames, ignore_index=True)
        data["ts_code"] = data["ts_code"].astype(str)
        data["end_date"] = data["end_date"].astype(str)
        numeric_cols = [
            "eps",
            "dt_eps",
            "q_eps",
            "basic_eps_yoy",
            "dt_eps_yoy",
            "roe",
            "roe_dt",
            "or_yoy",
            "q_sales_yoy",
            "netprofit_yoy",
            "dt_netprofit_yoy",
            "q_profit_yoy",
            "q_netprofit_yoy",
            "grossprofit_margin",
        ]
        for col in numeric_cols:
            if col not in data.columns:
                data[col] = pd.NA
            data[col] = pd.to_numeric(data[col], errors="coerce")

        latest = data.sort_values("end_date").groupby("ts_code", as_index=False).tail(1).copy()
        latest["sales_yoy"] = latest["q_sales_yoy"].combine_first(latest["or_yoy"])
        latest["profit_yoy"] = latest["q_netprofit_yoy"].combine_first(latest["q_profit_yoy"]).combine_first(latest["dt_netprofit_yoy"]).combine_first(latest["netprofit_yoy"])
        latest["eps_yoy"] = latest["dt_eps_yoy"].combine_first(latest["basic_eps_yoy"])
        latest["finance_status"] = latest.apply(self._finance_status, axis=1)
        latest["finance_note"] = latest.apply(self._finance_note, axis=1)
        latest["c_score"] = latest.apply(lambda row: self._current_growth_score(row["eps_yoy"], row["profit_yoy"], row["sales_yoy"]), axis=1)

        annual = data[data["end_date"].str.endswith("1231")].sort_values("end_date").groupby("ts_code", as_index=False).tail(1)
        annual = annual[["ts_code", "eps", "basic_eps_yoy", "dt_eps_yoy", "roe", "or_yoy", "netprofit_yoy"]].rename(
            columns={
                "eps": "annual_eps",
                "basic_eps_yoy": "annual_basic_eps_yoy",
                "dt_eps_yoy": "annual_dt_eps_yoy",
                "roe": "annual_roe",
                "or_yoy": "annual_sales_yoy",
                "netprofit_yoy": "annual_profit_yoy",
            }
        )
        annual["annual_eps_yoy"] = annual["annual_dt_eps_yoy"].combine_first(annual["annual_basic_eps_yoy"])

        result = latest[
            [
                "ts_code",
                "end_date",
                "eps_yoy",
                "basic_eps_yoy",
                "dt_eps_yoy",
                "q_eps",
                "sales_yoy",
                "profit_yoy",
                "grossprofit_margin",
                "finance_status",
                "finance_note",
                "c_score",
            ]
        ].merge(
            annual,
            on="ts_code",
            how="left",
        )
        result["a_score"] = result.apply(
            lambda row: self._annual_quality_score(
                row.get("annual_eps_yoy"),
                row.get("annual_profit_yoy"),
                row.get("annual_sales_yoy"),
                row.get("annual_roe"),
                row.get("annual_eps"),
            ),
            axis=1,
        )
        self.fundamental_rows = len(result)
        self.fundamental_status = "已读取"
        self.fundamental_note = f"读取{len(frames)}个报告期，覆盖{len(result)}只股票"
        self.log(self.fundamental_note)
        return result

    def _eastmoney_financial_scores(self, trade_date: str, failures: list[str]) -> pd.DataFrame:
        try:
            import akshare as ak
        except Exception as exc:
            failures.append(f"东财兜底不可用: akshare未安装或导入失败: {exc}")
            return pd.DataFrame()

        periods = sorted(self._report_periods(trade_date), reverse=True)
        frames = []
        for period in periods:
            frame = self._eastmoney_yjbb(ak, period)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            failures.append("东财业绩报表无数据")
            return pd.DataFrame()

        # Eastmoney reports can be partially disclosed. Combine several candidate
        # periods and keep the latest available report for each stock.
        all_reports = pd.concat(frames, ignore_index=True)
        all_reports["end_date"] = all_reports["end_date"].astype(str)
        result = all_reports.sort_values("end_date").groupby("ts_code", as_index=False).tail(1).copy()

        current_periods = sorted(result["end_date"].dropna().astype(str).unique().tolist(), reverse=True)
        prev_periods = sorted({f"{int(period[:4]) - 1}{period[4:]}" for period in current_periods if len(period) == 8}, reverse=True)
        prev_frames = []
        for period in prev_periods:
            frame = self._eastmoney_yjbb(ak, period)
            if not frame.empty:
                prev_frames.append(frame)
        if prev_frames:
            prev_all = pd.concat(prev_frames, ignore_index=True)
            prev_all["end_date"] = prev_all["end_date"].astype(str)
            prev_eps = prev_all[["ts_code", "end_date", "eps"]].rename(columns={"end_date": "prev_period", "eps": "prev_eps"})
            result["prev_period"] = result["end_date"].apply(lambda value: f"{int(str(value)[:4]) - 1}{str(value)[4:]}" if len(str(value)) == 8 else "")
            result = result.merge(prev_eps, on=["ts_code", "prev_period"], how="left")
            result["eps_yoy"] = result.apply(lambda row: self._yoy_from_values(row.get("eps"), row.get("prev_eps")), axis=1)
        else:
            result["eps_yoy"] = pd.NA
        result["basic_eps_yoy"] = result["eps_yoy"]
        result["dt_eps_yoy"] = pd.NA
        result["q_eps"] = result["eps"]
        result["finance_status"] = result.apply(self._eastmoney_finance_status, axis=1)
        result["finance_note"] = result.apply(self._finance_note, axis=1)
        result["c_score"] = result.apply(lambda row: self._current_growth_score(row["eps_yoy"], row["profit_yoy"], row["sales_yoy"]), axis=1)

        annual_use = pd.DataFrame(columns=["ts_code", "annual_eps", "annual_eps_yoy", "annual_roe", "annual_sales_yoy", "annual_profit_yoy"])
        annual_all = all_reports[all_reports["end_date"].str.endswith("1231")].copy()
        if not annual_all.empty:
            annual_latest = annual_all.sort_values("end_date").groupby("ts_code", as_index=False).tail(1).copy()
            annual_use = annual_latest[["ts_code", "end_date", "eps", "roe", "sales_yoy", "profit_yoy"]].rename(
                columns={
                    "end_date": "annual_period",
                    "eps": "annual_eps",
                    "roe": "annual_roe",
                    "sales_yoy": "annual_sales_yoy",
                    "profit_yoy": "annual_profit_yoy",
                }
            )
            annual_use["prev_annual_period"] = annual_use["annual_period"].apply(lambda value: f"{int(str(value)[:4]) - 1}{str(value)[4:]}" if len(str(value)) == 8 else "")
            prev_annual = annual_all[["ts_code", "end_date", "eps"]].rename(columns={"end_date": "prev_annual_period", "eps": "prev_annual_eps"})
            annual_use = annual_use.merge(prev_annual, on=["ts_code", "prev_annual_period"], how="left")
            annual_use["annual_eps_yoy"] = annual_use.apply(
                lambda row: self._yoy_from_values(row.get("annual_eps"), row.get("prev_annual_eps")),
                axis=1,
            )
        result = result.merge(annual_use, on="ts_code", how="left")
        result["a_score"] = result.apply(
            lambda row: self._annual_quality_score(
                row.get("annual_eps_yoy"),
                row.get("annual_profit_yoy"),
                row.get("annual_sales_yoy"),
                row.get("annual_roe"),
                row.get("annual_eps"),
            ),
            axis=1,
        )
        result["grossprofit_margin"] = result.get("grossprofit_margin", pd.NA)
        self.fundamental_rows = len(result)
        self.fundamental_status = "东财兜底"
        period_hint = "、".join(current_periods[:4])
        self.fundamental_note = f"Tushare财务失败，已用东财逐股最近财报覆盖{len(result)}只股票，报告期含{period_hint}"
        self.log(self.fundamental_note)
        cols = [
            "ts_code",
            "end_date",
            "eps_yoy",
            "basic_eps_yoy",
            "dt_eps_yoy",
            "q_eps",
            "sales_yoy",
            "profit_yoy",
            "grossprofit_margin",
            "finance_status",
            "finance_note",
            "c_score",
            "annual_eps",
            "annual_eps_yoy",
            "annual_roe",
            "annual_sales_yoy",
            "annual_profit_yoy",
            "a_score",
        ]
        return result[[col for col in cols if col in result.columns]]

    def _eastmoney_yjbb(self, ak: Any, period: str) -> pd.DataFrame:
        cache_name = f"eastmoney_yjbb_{period}"
        path = self.engine._cache_path(cache_name)
        if path.exists():
            try:
                cached = pd.read_csv(path, dtype=str)
            except pd.errors.EmptyDataError:
                self.log(f"{path.name} 缓存为空，已删除并重新读取。")
                path.unlink(missing_ok=True)
                cached = pd.DataFrame()
            if cached.empty:
                return pd.DataFrame()
            return self._normalize_eastmoney_yjbb(cached, period)
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                frame = ak.stock_yjbb_em(date=period)
        except Exception as exc:
            self.log(f"东财{period}业绩报表读取失败: {exc}")
            return pd.DataFrame()
        if frame is not None and not frame.empty:
            frame.to_csv(path, index=False, encoding="utf-8-sig")
        return self._normalize_eastmoney_yjbb(frame, period)

    def _normalize_eastmoney_yjbb(self, frame: pd.DataFrame, period: str) -> pd.DataFrame:
        if frame is None or frame.empty or "股票代码" not in frame.columns:
            return pd.DataFrame()
        data = frame.copy()
        data["symbol"] = data["股票代码"].astype(str).str.zfill(6)
        data["ts_code"] = data["symbol"].apply(self._symbol_to_ts_code)
        rename = {
            "每股收益": "eps",
            "营业总收入-同比增长": "sales_yoy",
            "净利润-同比增长": "profit_yoy",
            "净资产收益率": "roe",
            "销售毛利率": "grossprofit_margin",
            "最新公告日期": "ann_date",
        }
        data = data.rename(columns={old: new for old, new in rename.items() if old in data.columns})
        for col in ["eps", "sales_yoy", "profit_yoy", "roe", "grossprofit_margin"]:
            if col not in data.columns:
                data[col] = pd.NA
            data[col] = pd.to_numeric(data[col], errors="coerce")
        if "ann_date" not in data.columns:
            data["ann_date"] = ""
        data["end_date"] = period
        return data[["ts_code", "end_date", "eps", "sales_yoy", "profit_yoy", "roe", "grossprofit_margin", "ann_date"]]

    def _fetch_fina_indicator_period(self, period: str, field_sets: list[str], failures: list[str]) -> pd.DataFrame:
        try:
            frame = self.engine._fina_indicator(period)
        except Exception as exc:
            failures.append(f"{period} 财务指标读取失败: {exc}")
            return pd.DataFrame()
        if not frame.empty and "ts_code" in frame.columns:
            return frame
        failures.append(f"{period} 财务指标为空")
        return pd.DataFrame()

    def _call_fina_indicator(self, period: str, fields: str) -> pd.DataFrame:
        if fields:
            return self.engine.pro.fina_indicator(period=period, fields=fields)
        return self.engine.pro.fina_indicator(period=period)

    def _cached_financial(self, name: str, fetcher) -> pd.DataFrame:
        path = self.engine._cache_path(name)
        if path.exists():
            try:
                cached = pd.read_csv(path, dtype=str)
            except pd.errors.EmptyDataError:
                self.log(f"{path.name} 缓存为空，已删除并重新读取。")
                path.unlink(missing_ok=True)
                cached = pd.DataFrame()
            if not cached.empty and "ts_code" in cached.columns:
                return cached
        frame = fetcher()
        if frame is not None and not frame.empty and "ts_code" in frame.columns:
            frame.to_csv(path, index=False, encoding="utf-8-sig")
            return frame
        return pd.DataFrame()

    def _institutional_scores(self, trade_date: str, features: pd.DataFrame) -> pd.DataFrame:
        if str(getattr(self.config, "canslim_channel_mode", "all") or "all").strip().lower() == "canslim":
            self.institution_status = "已跳过"
            self.institution_note = "当前只看CANSLIM，陶博士机构条件未参与"
            return pd.DataFrame()
        if not bool(getattr(self.config, "tao_institution_enabled", True)):
            self.institution_status = "已关闭"
            self.institution_note = "陶博士机构条件未启用"
            return pd.DataFrame()

        periods = self._report_periods(trade_date)
        if not periods:
            self.institution_status = "无报告期"
            self.institution_note = "未推导出可读取的机构持仓报告期"
            return pd.DataFrame()

        fund = pd.DataFrame()
        fund_period = ""
        if bool(getattr(self.config, "tao_fund_hold_enabled", True)):
            for period in periods:
                frame = self._fund_holding_period(period)
                if not frame.empty:
                    fund = frame
                    fund_period = period
                    break

        north = pd.DataFrame()
        north_period = ""
        if bool(getattr(self.config, "tao_north_hold_enabled", True)):
            for period in periods:
                frame = self._north_holding_period(period, features)
                if not frame.empty:
                    north = frame
                    north_period = period
                    break

        result = pd.DataFrame({"ts_code": features["ts_code"].astype(str).unique()})
        if fund.empty:
            result["fund_hold_ratio"] = pd.NA
            result["fund_hold_period"] = ""
        else:
            result = result.merge(fund, on="ts_code", how="left")
            result["fund_hold_period"] = result["fund_hold_period"].fillna("")

        if north.empty:
            result["north_hold_value_wan"] = pd.NA
            result["north_hold_ratio"] = pd.NA
            result["north_hold_period"] = ""
        else:
            result = result.merge(north, on="ts_code", how="left")
            result["north_hold_period"] = result["north_hold_period"].fillna("")

        for col in ["fund_hold_ratio", "north_hold_value_wan", "north_hold_ratio"]:
            if col not in result.columns:
                result[col] = pd.NA
            result[col] = pd.to_numeric(result[col], errors="coerce")

        notes = []
        if bool(getattr(self.config, "tao_fund_hold_enabled", True)):
            notes.append(f"基金{fund_period or '无数据'}覆盖{0 if fund.empty else len(fund)}只")
        if bool(getattr(self.config, "tao_north_hold_enabled", True)):
            notes.append(f"北向{north_period or '无数据'}覆盖{0 if north.empty else len(north)}只")
        self.institution_rows = len(result)
        self.institution_status = "已读取" if (not fund.empty or not north.empty) else "无数据"
        self.institution_note = "；".join(notes) if notes else "未启用机构子条件"
        self.log(self.institution_note)
        return result

    def _fund_holding_period(self, period: str) -> pd.DataFrame:
        def fetch() -> pd.DataFrame:
            frames = []
            limit = 8000
            for page in range(50):
                offset = page * limit
                frame = self.engine.pro.fund_portfolio(
                    period=period,
                    offset=offset,
                    limit=limit,
                    fields="ts_code,symbol,mkv,amount,stk_mkv_ratio,stk_float_ratio",
                )
                if frame is None or frame.empty:
                    break
                frames.append(frame)
                if len(frame) < limit:
                    break
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        try:
            data = self.engine._cached(f"fund_portfolio_{period}", fetch)
        except Exception as exc:
            self.log(f"{period} 基金持仓读取失败: {exc}")
            return pd.DataFrame()
        if data.empty or "symbol" not in data.columns:
            return pd.DataFrame()
        data = data.copy()
        data["ts_code"] = data["symbol"].astype(str)
        if "stk_float_ratio" not in data.columns:
            data["stk_float_ratio"] = 0
        data["stk_float_ratio"] = pd.to_numeric(data["stk_float_ratio"], errors="coerce").fillna(0)
        grouped = data.groupby("ts_code", as_index=False)["stk_float_ratio"].sum()
        grouped = grouped.rename(columns={"stk_float_ratio": "fund_hold_ratio"})
        grouped["fund_hold_period"] = period
        return grouped

    def _north_holding_period(self, period: str, features: pd.DataFrame) -> pd.DataFrame:
        try:
            data = self.engine._cached(
                f"hk_hold_{period}",
                lambda: self.engine.pro.hk_hold(trade_date=period),
            )
        except Exception as exc:
            self.log(f"{period} 北向持仓读取失败: {exc}")
            return pd.DataFrame()
        if data.empty or "ts_code" not in data.columns or "vol" not in data.columns:
            return pd.DataFrame()
        data = data.copy()
        data["ts_code"] = data["ts_code"].astype(str)
        data["vol"] = pd.to_numeric(data["vol"], errors="coerce").fillna(0)
        if "ratio" not in data.columns:
            data["ratio"] = 0
        data["ratio"] = pd.to_numeric(data["ratio"], errors="coerce").fillna(0)
        grouped = data.groupby("ts_code", as_index=False).agg(north_hold_vol=("vol", "sum"), north_hold_ratio=("ratio", "sum"))
        close_map = features.set_index("ts_code")["close"].apply(lambda value: self._num(value, 0))
        grouped["close"] = grouped["ts_code"].map(close_map).fillna(0)
        grouped["north_hold_value_wan"] = grouped["north_hold_vol"] * grouped["close"] / 10000
        grouped["north_hold_period"] = period
        return grouped[["ts_code", "north_hold_value_wan", "north_hold_ratio", "north_hold_period"]]

    def _score_rows(
        self,
        basic: pd.DataFrame,
        daily_basic: pd.DataFrame,
        features: pd.DataFrame,
        fundamentals: pd.DataFrame,
        institutions: pd.DataFrame,
        market: dict[str, Any],
        mainline_stocks: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        data = features.copy()
        basic_cols = ["ts_code", "symbol", "name", "industry", "market", "stock_type", "list_date"]
        data = data.merge(basic[[col for col in basic_cols if col in basic.columns]], on="ts_code", how="left")

        db = daily_basic.copy()
        if "ts_code" in db.columns:
            db["ts_code"] = db["ts_code"].astype(str)
        else:
            db["ts_code"] = pd.Series(dtype=str)
        for col in ["turnover_rate", "turnover_rate_f", "volume_ratio", "total_mv", "circ_mv"]:
            if col not in db.columns:
                db[col] = 0
            db[col] = pd.to_numeric(db[col], errors="coerce").fillna(0)
        data = data.merge(db[["ts_code", "turnover_rate", "turnover_rate_f", "volume_ratio", "total_mv", "circ_mv"]], on="ts_code", how="left")

        if fundamentals.empty:
            data["c_score"] = 50.0
            data["a_score"] = 50.0
            data["finance_status"] = self.fundamental_status
            data["finance_note"] = self.fundamental_note
            for col in [
                "eps_yoy",
                "basic_eps_yoy",
                "dt_eps_yoy",
                "q_eps",
                "sales_yoy",
                "profit_yoy",
                "annual_eps_yoy",
                "annual_roe",
                "annual_eps",
                "grossprofit_margin",
                "end_date",
            ]:
                data[col] = pd.NA
        else:
            data = data.merge(fundamentals, on="ts_code", how="left")
            data["c_score"] = data["c_score"].fillna(50.0)
            data["a_score"] = data["a_score"].fillna(50.0)
            data["finance_status"] = data["finance_status"].fillna("无匹配财报")
            data["finance_note"] = data["finance_note"].fillna("该股票未匹配到最近报告期")

        institution_numeric_cols = ["fund_hold_ratio", "north_hold_value_wan", "north_hold_ratio"]
        institution_text_cols = ["fund_hold_period", "north_hold_period"]
        if institutions.empty:
            for col in institution_numeric_cols:
                data[col] = pd.NA
            for col in institution_text_cols:
                data[col] = ""
        else:
            inst = institutions.copy()
            inst["ts_code"] = inst["ts_code"].astype(str)
            keep_cols = ["ts_code"] + institution_numeric_cols + institution_text_cols
            for col in keep_cols:
                if col not in inst.columns:
                    inst[col] = pd.NA if col in institution_numeric_cols else ""
            data = data.merge(inst[keep_cols], on="ts_code", how="left")
            for col in institution_text_cols:
                data[col] = data[col].fillna("")
        for col in institution_numeric_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data["tao_institution_pass"] = data.apply(self._tao_institution_ok, axis=1)
        data["tao_institution_note"] = data.apply(self._tao_institution_note, axis=1)

        data["name"] = data["name"].fillna("")
        data["symbol"] = data["symbol"].fillna(data["ts_code"].str.slice(0, 6))
        data["industry"] = data["industry"].fillna("未分类").replace("", "未分类")
        self.log("读取产业题材概念映射...")
        concept_map = self.engine._stock_concept_map(str(self.config.end_date), data["ts_code"].astype(str).tolist())
        data["concepts"] = data["ts_code"].astype(str).map(concept_map).fillna("")
        data = self._attach_mainline_scores(data, mainline_stocks)
        data["market_cap_yi"] = pd.to_numeric(data["total_mv"], errors="coerce").fillna(0) / 10000
        data["circ_mv_yi"] = pd.to_numeric(data["circ_mv"], errors="coerce").fillna(0) / 10000
        data["turnover_rate_f"] = pd.to_numeric(data["turnover_rate_f"], errors="coerce").fillna(0)
        data["volume_ratio"] = pd.to_numeric(data["volume_ratio"], errors="coerce").fillna(0)
        self._last_canslim_data = self._add_scores(data.copy(), market)
        if getattr(self.config, "canslim_probe_only", False) and str(getattr(self.config, "canslim_probe_query", "") or "").strip():
            return []

        data = self._apply_base_filters(data)
        if data.empty:
            return []

        data = self._add_scores(data, market)
        # A candidate must be strong enough by score and RPS, then pass either
        # the Tao/RPS near-high route or the CANSLIM growth-and-breakout route.
        data = data[
            (data["score"] >= float(self.config.canslim_min_score))
            & (data["channel"] != "")
        ].copy()
        data = self._apply_channel_mode(data)
        data = self._apply_mainline_filter(data)
        if data.empty:
            return []

        data = data.sort_values(["score", "rps_combo", "rps120", "rps250"], ascending=False)
        rows = []
        for _, row in data.head(int(self.config.canslim_top_count)).iterrows():
            rows.append(self._to_row(row))
        return rows

    def _add_scores(self, data: pd.DataFrame, market: dict[str, Any]) -> pd.DataFrame:
        data = data.copy()
        data["l_score"] = data["rps_combo"].fillna(0).clip(0, 100)
        data["n_score"] = data.apply(self._new_factor_score, axis=1)
        data["s_score"] = data.apply(self._supply_demand_score, axis=1)
        data["i_score"] = data.apply(self._sponsorship_proxy_score, axis=1)
        data["m_score"] = float(market.get("score", 50))
        data["base_score"] = (
            data["c_score"] * 0.20
            + data["a_score"] * 0.10
            + data["n_score"] * 0.15
            + data["s_score"] * 0.10
            + data["l_score"] * 0.25
            + data["i_score"] * 0.05
            + data["m_score"] * 0.15
        )
        data["base_score"] = data["base_score"] - data.apply(self._risk_penalty, axis=1)
        if bool(getattr(self.config, "canslim_mainline_enabled", True)):
            weight = max(0.0, min(1.0, float(getattr(self.config, "canslim_mainline_weight", 0.25))))
            mainline_values = data["mainline_score"] if "mainline_score" in data.columns else pd.Series(50.0, index=data.index)
            mainline_score = pd.to_numeric(mainline_values, errors="coerce").fillna(50)
            data["score"] = data["base_score"] * (1 - weight) + mainline_score * weight
        else:
            data["score"] = data["base_score"]

        data["setup"] = data.apply(self._setup_label, axis=1)
        data["tao_pass"] = data.apply(self._tao_pass, axis=1)
        data["canslim_pass"] = data.apply(self._canslim_pass, axis=1)
        data["channel"] = data.apply(self._pass_channel, axis=1)
        return data

    def _attach_mainline_scores(
        self,
        data: pd.DataFrame,
        mainline_stocks: pd.DataFrame,
    ) -> pd.DataFrame:
        result = data.copy()
        enabled = bool(getattr(self.config, "canslim_mainline_enabled", True))
        if enabled and not mainline_stocks.empty and "ts_code" in mainline_stocks.columns:
            scores = mainline_stocks.copy()
            scores["ts_code"] = scores["ts_code"].astype(str)
            result = result.merge(scores, on="ts_code", how="left")

        result["mainline_industry"] = result.get("mainline_industry", result["industry"])
        result["mainline_industry"] = result["mainline_industry"].fillna(result["industry"])
        missing_industry = result["mainline_industry"].astype(str).str.strip() == ""
        result.loc[missing_industry, "mainline_industry"] = result.loc[missing_industry, "industry"]
        defaults: dict[str, Any] = {
            "mainline_industry_code": "",
            "mainline_industry_source": "未计算" if enabled else "已关闭",
            "mainline_score": 50.0,
            "mainline_state": "数据不足" if enabled else "未启用",
            "mainline_reason": "",
            "industry_rank": 0,
            "industry_rank_prev5": 0,
            "industry_rank_change_5": 0,
            "industry_rps20": 50.0,
            "industry_rps60": 50.0,
            "industry_rps120": 50.0,
            "industry_rps_combo": 50.0,
            "industry_rps_combo_prev5": 50.0,
            "industry_breadth_score": 50.0,
            "industry_persistence_score": 50.0,
            "industry_above_ma20_ratio": 0.0,
            "industry_above_ma50_ratio": 0.0,
            "industry_strong_ratio": 0.0,
            "member_count": 0,
        }
        for col, default in defaults.items():
            if col not in result.columns:
                result[col] = default
            else:
                result[col] = result[col].fillna(default)
        return result

    def _apply_mainline_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty or not bool(getattr(self.config, "canslim_mainline_enabled", True)):
            return data
        result = data.copy()
        minimum = float(getattr(self.config, "canslim_mainline_min_score", 0))
        if minimum > 0:
            result = result[pd.to_numeric(result["mainline_score"], errors="coerce").fillna(0) >= minimum]

        mode = str(getattr(self.config, "canslim_mainline_filter", "all") or "all").strip().lower()
        if mode in {"candidate", "candidates", "mainline"}:
            result = result[result["mainline_state"].isin({MAINLINE_CONFIRMED, MAINLINE_CANDIDATE})]
        elif mode in {"confirmed", "confirm"}:
            result = result[result["mainline_state"] == MAINLINE_CONFIRMED]
        return result

    def _probe(self, data: pd.DataFrame) -> dict[str, Any]:
        query = str(getattr(self.config, "canslim_probe_query", "") or "").strip()
        if not query:
            return {}
        if data.empty:
            return {"query": query, "found": False, "message": "没有可分析的数据"}
        match = self._match_stock(data, query)
        if match is None:
            return {"query": query, "found": False, "message": f"未找到 {query}"}

        checks = self._probe_checks(match)
        enabled = [item for item in checks if not item.get("disabled")]
        overall_pass = bool(enabled) and all(bool(item.get("pass")) for item in enabled)
        row = self._to_row(match)
        row["probe_result"] = "符合" if overall_pass else "不符合"
        failed = [item["label"] for item in enabled if not item.get("pass")]
        return {
            "query": query,
            "found": True,
            "overall_pass": overall_pass,
            "summary": "全部达标" if overall_pass else "未达标：" + "、".join(failed[:8]),
            "row": row,
            "checks": checks,
        }

    def _match_stock(self, data: pd.DataFrame, query: str) -> pd.Series | None:
        q = str(query).strip()
        q_upper = q.upper()
        symbol = "".join(ch for ch in q if ch.isdigit())
        candidates = data.copy()
        candidates["symbol"] = candidates["symbol"].astype(str)
        candidates["ts_code"] = candidates["ts_code"].astype(str)
        candidates["name"] = candidates["name"].astype(str)
        if symbol:
            symbol = symbol.zfill(6)
            exact = candidates[(candidates["symbol"] == symbol) | candidates["ts_code"].str.startswith(symbol)]
            if not exact.empty:
                return exact.iloc[0]
        exact_name = candidates[candidates["name"].str.upper() == q_upper]
        if not exact_name.empty:
            return exact_name.iloc[0]
        contains = candidates[candidates["name"].str.contains(q, case=False, na=False, regex=False)]
        if not contains.empty:
            return contains.iloc[0]
        return None

    def _probe_checks(self, row: pd.Series) -> list[dict[str, Any]]:
        checks = []
        checks.append(self._check("有效价格", row.get("close"), ">", 0, self._num(row.get("close"), 0) > 0))
        checks.append(self._check("成交额", row.get("amount_yi"), ">=", float(self.config.canslim_min_amount_yi), self._num(row.get("amount_yi"), 0) >= float(self.config.canslim_min_amount_yi), "亿"))

        min_mv = float(self.config.canslim_min_market_cap_yi)
        max_mv = float(self.config.canslim_max_market_cap_yi)
        if min_mv > 0:
            checks.append(self._check("市值下限", row.get("market_cap_yi"), ">=", min_mv, self._num(row.get("market_cap_yi"), 0) >= min_mv, "亿"))
        if max_mv > 0:
            checks.append(self._check("市值上限", row.get("market_cap_yi"), "<=", max_mv, self._num(row.get("market_cap_yi"), 0) <= max_mv, "亿"))

        turnover = self._num(row.get("turnover_rate_f"), 0)
        min_turnover = float(self.config.canslim_min_turnover_rate)
        max_turnover = float(self.config.canslim_max_turnover_rate)
        if min_turnover > 0:
            checks.append(self._check("实换手下限", turnover, ">=", min_turnover, turnover >= min_turnover, "%"))
        if max_turnover > 0:
            turnover_ok = turnover <= max_turnover if min_turnover > 0 else (turnover <= max_turnover or turnover <= 0)
            checks.append(self._check("实换手上限", turnover, "<=", max_turnover, turnover_ok, "%"))

        name = str(row.get("name", ""))
        symbol = str(row.get("symbol", ""))
        if self.config.canslim_exclude_st:
            checks.append(self._check("排除ST", name, "not contains", "ST", "ST" not in name.upper()))
        if self.config.canslim_exclude_star:
            checks.append(self._check("排除科创板", symbol, "not starts", "688/689", not symbol.startswith(("688", "689"))))
        if self.config.canslim_exclude_chinext:
            checks.append(self._check("排除创业板", symbol, "not starts", "300/301", not symbol.startswith(("300", "301"))))
        checks.append(self._check("排除北交所", symbol, "not starts", "8/4", not symbol.startswith(("8", "4"))))
        checks.append(self._check("上市满180天", row.get("list_date"), ">=", "180天", self._listed_enough(str(row.get("list_date", "")))))

        checks.append(self._check("最低总分", row.get("score"), ">=", float(self.config.canslim_min_score), self._num(row.get("score"), 0) >= float(self.config.canslim_min_score)))
        checks.append(self._check("CANSLIM最低RPS", row.get("rps_combo"), ">=", float(self.config.canslim_min_rps), self._num(row.get("rps_combo"), 0) >= float(self.config.canslim_min_rps)))
        checks.append(self._check("CANSLIM成长通道", row.get("channel"), "pass", "财务+RPS+形态", bool(row.get("canslim_pass", False))))
        checks.append(self._check("陶博士RPS通道", row.get("channel"), "pass", "RPS+新高+机构", bool(row.get("tao_pass", False))))
        if bool(getattr(self.config, "tao_institution_enabled", True)):
            checks.append(self._check("陶博士机构条件", row.get("tao_institution_note"), "pass", self._institution_threshold_label(), bool(row.get("tao_institution_pass", False))))
        checks.append(self._check("通道模式", row.get("channel"), "match", self.config.canslim_channel_mode, self._probe_channel_mode_ok(row)))
        if bool(getattr(self.config, "canslim_mainline_enabled", True)):
            minimum = float(getattr(self.config, "canslim_mainline_min_score", 0))
            if minimum > 0:
                checks.append(
                    self._check(
                        "主线最低分",
                        row.get("mainline_score"),
                        ">=",
                        minimum,
                        self._num(row.get("mainline_score"), 0) >= minimum,
                    )
                )
            mode = str(getattr(self.config, "canslim_mainline_filter", "all") or "all").strip().lower()
            state = str(row.get("mainline_state", ""))
            if mode in {"candidate", "candidates", "mainline"}:
                checks.append(
                    self._check(
                        "主线过滤",
                        state,
                        "in",
                        "主线候选及以上",
                        state in {MAINLINE_CONFIRMED, MAINLINE_CANDIDATE},
                    )
                )
            elif mode in {"confirmed", "confirm"}:
                checks.append(
                    self._check(
                        "主线过滤",
                        state,
                        "==",
                        MAINLINE_CONFIRMED,
                        state == MAINLINE_CONFIRMED,
                    )
                )
        return checks

    def _probe_channel_mode_ok(self, row: pd.Series) -> bool:
        mode = str(self.config.canslim_channel_mode or "all").strip().lower()
        canslim = bool(row.get("canslim_pass", False))
        tao = bool(row.get("tao_pass", False))
        if mode == "canslim":
            return canslim
        if mode == "tao":
            return tao
        if mode == "both":
            return canslim and tao
        return canslim or tao

    @staticmethod
    def _check(label: str, value: Any, op: str, threshold: Any, passed: bool, unit: str = "") -> dict[str, Any]:
        return {
            "label": label,
            "value": "" if value is None or pd.isna(value) else value,
            "op": op,
            "threshold": threshold,
            "unit": unit,
            "pass": bool(passed),
        }

    def _apply_base_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply hard liquidity, board, listing-age, market-cap, and turnover gates.

        These gates run before scoring so the ranking stage focuses on stocks that are
        tradable for this strategy style. Max values <= 0 are treated as disabled.
        """
        result = data.copy()
        result = result[result["close"].fillna(0) > 0]
        result = result[result["amount_yi"].fillna(0) >= float(self.config.canslim_min_amount_yi)]

        min_market_cap = float(self.config.canslim_min_market_cap_yi)
        max_market_cap = float(self.config.canslim_max_market_cap_yi)
        if min_market_cap > 0:
            result = result[result["market_cap_yi"].fillna(0) >= min_market_cap]
        if max_market_cap > 0:
            result = result[result["market_cap_yi"].fillna(0) <= max_market_cap]

        min_turnover = float(self.config.canslim_min_turnover_rate)
        max_turnover = float(self.config.canslim_max_turnover_rate)
        turnover = result["turnover_rate_f"].fillna(0)
        if min_turnover > 0:
            result = result[turnover >= min_turnover]
        if max_turnover > 0:
            turnover = result["turnover_rate_f"].fillna(0)
            if min_turnover > 0:
                result = result[turnover <= max_turnover]
            else:
                result = result[(turnover <= max_turnover) | (turnover <= 0)]

        if self.config.canslim_exclude_st:
            result = result[~result["name"].astype(str).str.upper().str.contains("ST", regex=False)]
        if self.config.canslim_exclude_star:
            result = result[~result["symbol"].astype(str).str.startswith(("688", "689"))]
        if self.config.canslim_exclude_chinext:
            result = result[~result["symbol"].astype(str).str.startswith(("300", "301"))]
        result = result[~result["symbol"].astype(str).str.startswith(("8", "4"))]
        result = result[result["list_date"].fillna("19000101").astype(str).apply(self._listed_enough)]
        return result

    def _apply_channel_mode(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter candidates by the user-selected route mode."""
        mode = str(self.config.canslim_channel_mode or "all").strip().lower()
        if mode == "canslim":
            return data[data["canslim_pass"]].copy()
        if mode == "tao":
            return data[data["tao_pass"]].copy()
        if mode == "both":
            return data[data["canslim_pass"] & data["tao_pass"]].copy()
        return data

    def _market_direction(self, daily: pd.DataFrame, trade_dates: list[str]) -> dict[str, Any]:
        by_date = daily.groupby("trade_date")["pct_chg"].mean().reindex(trade_dates).fillna(0)
        latest = daily[daily["trade_date"] == trade_dates[-1]].copy()
        up_ratio = float((latest["pct_chg"] > 0).mean()) if len(latest) else 0.0

        close = daily.pivot_table(index="ts_code", columns="trade_date", values="close", aggfunc="last").reindex(columns=trade_dates)
        ma50 = close.iloc[:, -min(50, len(trade_dates)) :].mean(axis=1)
        ma120 = close.iloc[:, -min(120, len(trade_dates)) :].mean(axis=1)
        latest_close = close[trade_dates[-1]]
        above_ma50_ratio = float((latest_close > ma50).mean())
        ma_trend_ratio = float(((latest_close > ma50) & (ma50 > ma120)).mean())

        short_sum = float(by_date.tail(5).sum())
        mid_sum = float(by_date.tail(20).sum())
        score = (
            self._clip(short_sum, -3, 5) * 0.28
            + self._clip(mid_sum, -8, 12) * 0.32
            + self._clip(up_ratio, 0.35, 0.75) * 0.18
            + self._clip(above_ma50_ratio, 0.30, 0.70) * 0.14
            + self._clip(ma_trend_ratio, 0.18, 0.55) * 0.08
        )
        if score >= 68:
            state = "上升/可进攻"
            playbook = "可重点观察RPS龙头的新高突破和首次回踩。"
        elif score >= 52:
            state = "震荡偏强"
            playbook = "只做形态最紧、RPS最强的核心票，降低追高。"
        elif score >= 38:
            state = "震荡偏弱"
            playbook = "候选池为主，等待市场重新站上强势区。"
        else:
            state = "弱势/防守"
            playbook = "CANSLIM顺势要求高，弱市以观察为主。"
        return {
            "state": state,
            "score": round(score, 1),
            "short_sum": round(short_sum, 2),
            "mid_sum": round(mid_sum, 2),
            "up_ratio": round(up_ratio, 3),
            "above_ma50_ratio": round(above_ma50_ratio, 3),
            "ma_trend_ratio": round(ma_trend_ratio, 3),
            "playbook": playbook,
        }

    def _report_periods(self, trade_date: str) -> list[str]:
        year = int(trade_date[:4])
        candidates = []
        for item_year in range(year - 4, year + 1):
            for suffix in ["0331", "0630", "0930", "1231"]:
                period = f"{item_year}{suffix}"
                if period <= trade_date:
                    candidates.append(period)
        quarter_periods = candidates[-6:]
        annual_periods = [period for period in candidates if period.endswith("1231")][-4:]
        return sorted(set(quarter_periods + annual_periods))

    def _finance_status(self, row: pd.Series) -> str:
        eps_yoy = self._num(row.get("dt_eps_yoy"), None)
        if eps_yoy is None:
            eps_yoy = self._num(row.get("basic_eps_yoy"), None)
        profit = self._num(row.get("profit_yoy"), None)
        sales = self._num(row.get("sales_yoy"), None)
        if eps_yoy is not None:
            return "EPS同比"
        if profit is not None or sales is not None:
            return "利润/营收"
        return "财务缺字段"

    def _eastmoney_finance_status(self, row: pd.Series) -> str:
        eps_yoy = self._num(row.get("eps_yoy"), None)
        if eps_yoy is not None:
            return "东财EPS同比"
        profit = self._num(row.get("profit_yoy"), None)
        sales = self._num(row.get("sales_yoy"), None)
        if profit is not None or sales is not None:
            return "东财利润/营收"
        return "东财缺字段"

    def _finance_note(self, row: pd.Series) -> str:
        parts = []
        eps_yoy = self._num(row.get("dt_eps_yoy"), None)
        if eps_yoy is not None:
            parts.append(f"扣非EPS同比{eps_yoy:.1f}%")
        else:
            basic_eps_yoy = self._num(row.get("basic_eps_yoy"), None)
            if basic_eps_yoy is not None:
                parts.append(f"基本EPS同比{basic_eps_yoy:.1f}%")
        profit = self._num(row.get("profit_yoy"), None)
        if profit is not None:
            parts.append(f"利润同比{profit:.1f}%")
        sales = self._num(row.get("sales_yoy"), None)
        if sales is not None:
            parts.append(f"营收同比{sales:.1f}%")
        return "；".join(parts) if parts else "接口返回但缺EPS/利润/营收同比"

    @staticmethod
    def _symbol_to_ts_code(symbol: str) -> str:
        code = str(symbol).zfill(6)
        if code.startswith(("6", "9")):
            return f"{code}.SH"
        if code.startswith(("8", "4")):
            return f"{code}.BJ"
        return f"{code}.SZ"

    def _yoy_from_values(self, current: Any, previous: Any) -> float | Any:
        cur = self._num(current, None)
        prev = self._num(previous, None)
        if cur is None or prev is None:
            return pd.NA
        if prev > 0:
            return (cur / prev - 1) * 100
        if prev == 0:
            if cur > 0:
                return 100.0
            if cur < 0:
                return -100.0
            return 0.0
        return (cur - prev) / abs(prev) * 100

    def _current_growth_score(self, eps_yoy: Any, profit_yoy: Any, sales_yoy: Any) -> float:
        eps = self._num(eps_yoy, None)
        profit = self._num(profit_yoy, None)
        sales = self._num(sales_yoy, None)
        if eps is None and profit is None and sales is None:
            return 50.0
        eps_score = 50.0 if eps is None else self._growth_yoy_score(eps, 25, 80)
        profit_score = 50.0 if profit is None else self._growth_yoy_score(profit, 25, 80)
        sales_score = 50.0 if sales is None else self._growth_yoy_score(sales, 20, 50)
        bonus = 0
        if eps is not None and eps >= 25:
            bonus += 5
        if eps is not None and profit is not None and sales is not None and eps >= 25 and profit >= 25 and sales >= 20:
            bonus += 5
        return min(100.0, eps_score * 0.60 + profit_score * 0.25 + sales_score * 0.15 + bonus)

    def _annual_quality_score(self, eps_yoy: Any, profit_yoy: Any, sales_yoy: Any, roe: Any, eps: Any) -> float:
        annual_eps_yoy = self._num(eps_yoy, None)
        profit = self._num(profit_yoy, None)
        sales = self._num(sales_yoy, None)
        roe_value = self._num(roe, None)
        eps_value = self._num(eps, None)
        if annual_eps_yoy is None and profit is None and sales is None and roe_value is None and eps_value is None:
            return 50.0
        score = 0.0
        score += (50.0 if annual_eps_yoy is None else self._growth_yoy_score(annual_eps_yoy, 20, 60)) * 0.35
        score += (50.0 if profit is None else self._growth_yoy_score(profit, 20, 60)) * 0.20
        score += (50.0 if sales is None else self._growth_yoy_score(sales, 15, 40)) * 0.15
        score += (50.0 if roe_value is None else self._clip(roe_value, 5, 20)) * 0.20
        score += 10 if eps_value is not None and eps_value > 0 else 4
        return min(100.0, score)

    @staticmethod
    def _growth_yoy_score(value: float, good: float, excellent: float) -> float:
        if value <= -20:
            return 0.0
        if value < 0:
            return max(0.0, (value + 20) / 20 * 35)
        if value < good:
            return 35 + value / good * 40
        if value < excellent:
            return 75 + (value - good) / (excellent - good) * 25
        return 100.0

    def _new_factor_score(self, row: pd.Series) -> float:
        high_position = self._num(row.get("high_position_250"), 0)
        base = self._clip(high_position, 0.70, 1.00) * 0.45
        if bool(row.get("new_high_5")):
            base += 28
        if bool(row.get("breakout_20")):
            base += 12
        if bool(row.get("pocket_pivot")):
            base += 15
        return min(100.0, base)

    def _supply_demand_score(self, row: pd.Series) -> float:
        amount_score = self._clip(self._num(row.get("amount_yi"), 0), self.config.canslim_min_amount_yi, 20)
        ratio_score = self._clip(self._num(row.get("amount_ratio"), 1), 0.75, 1.80)
        turnover = self._num(row.get("turnover_rate_f"), 0)
        # Healthy demand usually expands volume without forcing extreme turnover.
        turnover_score = 70.0 if turnover <= 0 else 100.0 - min(55.0, max(0.0, turnover - 8) * 5)
        return amount_score * 0.35 + ratio_score * 0.40 + turnover_score * 0.25

    def _sponsorship_proxy_score(self, row: pd.Series) -> float:
        # A-share fund-holding data is lagged, so tradability is used as a proxy.
        amount_score = self._clip(self._num(row.get("amount_yi"), 0), 2, 35)
        mv_score = self._clip(self._num(row.get("market_cap_yi"), 0), 40, 1000)
        volume_ratio_score = self._clip(self._num(row.get("volume_ratio"), 1), 0.8, 1.8)
        return amount_score * 0.45 + mv_score * 0.30 + volume_ratio_score * 0.25

    def _risk_penalty(self, row: pd.Series) -> float:
        penalty = 0.0
        distance_ma50 = self._num(row.get("distance_ma50"), 0)
        if distance_ma50 > 35:
            penalty += min(18.0, (distance_ma50 - 35) * 0.6)
        if self._num(row.get("pct_chg"), 0) > 10.5:
            penalty += 6.0
        if self._num(row.get("high_position_250"), 0) < 0.75:
            penalty += 6.0
        if not bool(row.get("ma_stack")) and self._num(row.get("rps_combo"), 0) < 95:
            penalty += 5.0
        return penalty

    def _tao_pass(self, row: pd.Series) -> bool:
        min_rps = float(self.config.tao_min_rps)
        core_rps = float(self.config.tao_core_rps)
        near_high_ratio = float(self.config.tao_near_high_ratio)
        rps50 = self._num(row.get("rps50"), 0)
        rps120 = self._num(row.get("rps120"), 0)
        rps250 = self._num(row.get("rps250"), 0)
        high_position = self._num(row.get("high_position_250"), 0)
        near_high = high_position >= near_high_ratio or bool(row.get("new_high_5"))
        rps_floor = max(rps50, rps120, rps250) >= min_rps
        tao_rps = (rps120 >= core_rps) or (rps250 >= core_rps) or (rps50 >= core_rps and rps120 >= min_rps)
        return bool(near_high and rps_floor and tao_rps and self._tao_fundamental_ok(row) and self._tao_institution_ok(row))

    def _canslim_pass(self, row: pd.Series) -> bool:
        growth_ok = self._canslim_fundamental_ok(row)
        leadership_ok = self._num(row.get("rps_combo"), 0) >= float(self.config.canslim_min_rps)
        new_ok = bool(row.get("new_high_5")) or bool(row.get("pocket_pivot")) or bool(row.get("breakout_20"))
        trend_ok = bool(row.get("ma_stack")) or (
            self._num(row.get("close"), 0) > self._num(row.get("ma50"), 0)
            and self._num(row.get("ma50"), 0) > self._num(row.get("ma120"), 0)
        )
        return bool(growth_ok and leadership_ok and new_ok and trend_ok)

    def _canslim_fundamental_ok(self, row: pd.Series) -> bool:
        """Strict CANSLIM growth gate driven by user-entered financial thresholds."""
        if not self.config.canslim_include_fundamental:
            return False

        checks = [
            self._any_metric_meets(row, ["eps_yoy", "basic_eps_yoy"], self.config.canslim_min_eps_yoy),
            self._metric_meets(row, "dt_eps_yoy", self.config.canslim_min_dt_eps_yoy),
            self._metric_meets(row, "profit_yoy", self.config.canslim_min_profit_yoy),
            self._metric_meets(row, "sales_yoy", self.config.canslim_min_sales_yoy),
            self._metric_meets(row, "annual_eps_yoy", self.config.canslim_min_annual_eps_yoy),
            self._metric_meets(row, "annual_roe", self.config.canslim_min_roe),
        ]
        enabled = [item for item in checks if item is not None]
        if not enabled:
            return bool(self._num(row.get("c_score"), 50) >= 60 or self._num(row.get("a_score"), 50) >= 60)
        return all(enabled)

    def _tao_fundamental_ok(self, row: pd.Series) -> bool:
        """Loose Tao-style quality floor; thresholds default to 0 so users opt in."""
        if not self.config.canslim_include_fundamental:
            return True
        checks = [
            self._any_metric_meets(row, ["eps_yoy", "basic_eps_yoy"], self.config.tao_min_eps_yoy),
            self._metric_meets(row, "profit_yoy", self.config.tao_min_profit_yoy),
            self._metric_meets(row, "sales_yoy", self.config.tao_min_sales_yoy),
            self._metric_meets(row, "annual_roe", self.config.tao_min_roe),
        ]
        enabled = [item for item in checks if item is not None]
        return all(enabled) if enabled else True

    def _tao_institution_ok(self, row: pd.Series) -> bool:
        if not bool(getattr(self.config, "tao_institution_enabled", True)):
            return True

        checks: list[bool] = []
        missing_pass = bool(getattr(self.config, "tao_institution_missing_pass", False))

        if bool(getattr(self.config, "tao_fund_hold_enabled", True)):
            threshold = float(getattr(self.config, "tao_fund_hold_min_ratio", 2.0))
            value = self._num(row.get("fund_hold_ratio"), None)
            checks.append(True if threshold <= 0 else (missing_pass if value is None else value >= threshold))

        if bool(getattr(self.config, "tao_north_hold_enabled", True)):
            threshold = float(getattr(self.config, "tao_north_hold_min_value_wan", 3000.0))
            value = self._num(row.get("north_hold_value_wan"), None)
            checks.append(True if threshold <= 0 else (missing_pass if value is None else value >= threshold))

        if not checks:
            return True
        mode = str(getattr(self.config, "tao_institution_mode", "all") or "all").strip().lower()
        if mode == "any":
            return any(checks)
        return all(checks)

    def _tao_institution_note(self, row: pd.Series) -> str:
        if not bool(getattr(self.config, "tao_institution_enabled", True)):
            return "未启用"

        parts = []
        if bool(getattr(self.config, "tao_fund_hold_enabled", True)):
            threshold = float(getattr(self.config, "tao_fund_hold_min_ratio", 2.0))
            value = self._num(row.get("fund_hold_ratio"), None)
            period = str(row.get("fund_hold_period", "") or "")
            if value is None:
                parts.append(f"基金无数据/{threshold:.2f}%")
            else:
                suffix = f"({period})" if period else ""
                parts.append(f"基金{value:.2f}%/{threshold:.2f}%{suffix}")

        if bool(getattr(self.config, "tao_north_hold_enabled", True)):
            threshold = float(getattr(self.config, "tao_north_hold_min_value_wan", 3000.0))
            value = self._num(row.get("north_hold_value_wan"), None)
            period = str(row.get("north_hold_period", "") or "")
            if value is None:
                parts.append(f"北向无数据/{threshold:.0f}万")
            else:
                suffix = f"({period})" if period else ""
                parts.append(f"北向{value:.0f}万/{threshold:.0f}万{suffix}")

        if not parts:
            return "未启用机构子条件"
        mode = "任一满足" if str(getattr(self.config, "tao_institution_mode", "all") or "all").strip().lower() == "any" else "同时满足"
        return f"{mode}：" + "；".join(parts)

    def _institution_threshold_label(self) -> str:
        if not bool(getattr(self.config, "tao_institution_enabled", True)):
            return "未启用"
        parts = []
        if bool(getattr(self.config, "tao_fund_hold_enabled", True)):
            parts.append(f"基金≥{float(getattr(self.config, 'tao_fund_hold_min_ratio', 2.0)):.2f}%")
        if bool(getattr(self.config, "tao_north_hold_enabled", True)):
            parts.append(f"北向≥{float(getattr(self.config, 'tao_north_hold_min_value_wan', 3000.0)):.0f}万")
        if not parts:
            return "未启用机构子条件"
        joiner = " 或 " if str(getattr(self.config, "tao_institution_mode", "all") or "all").strip().lower() == "any" else " 且 "
        return joiner.join(parts)

    def _metric_meets(self, row: pd.Series, key: str, threshold: Any) -> bool | None:
        threshold_value = float(threshold)
        if threshold_value <= 0:
            return None
        value = self._num(row.get(key), None)
        return value is not None and value >= threshold_value

    def _any_metric_meets(self, row: pd.Series, keys: list[str], threshold: Any) -> bool | None:
        threshold_value = float(threshold)
        if threshold_value <= 0:
            return None
        values = [self._num(row.get(key), None) for key in keys]
        return any(value is not None and value >= threshold_value for value in values)

    @staticmethod
    def _pass_channel(row: pd.Series) -> str:
        canslim = bool(row.get("canslim_pass", False))
        tao = bool(row.get("tao_pass", False))
        if canslim and tao:
            return "CANSLIM+陶博士"
        if canslim:
            return "CANSLIM成长"
        if tao:
            return "陶博士RPS"
        return ""

    def _setup_label(self, row: pd.Series) -> str:
        if bool(row.get("new_high_5")) and self._num(row.get("amount_ratio"), 1) >= 1.05:
            return "新高突破"
        if bool(row.get("pocket_pivot")):
            return "口袋支点"
        if self._num(row.get("low"), 0) <= self._num(row.get("ma50"), 0) * 1.03 and self._num(row.get("close"), 0) > self._num(row.get("ma50"), 0):
            return "均线回踩"
        if bool(row.get("breakout_20")):
            return "平台突破"
        return "强势观察"

    def _price_ma_label(self, row: pd.Series) -> str:
        close = self._num(row.get("close"), 0)
        low = self._num(row.get("low"), 0)
        ma5 = self._num(row.get("ma5"), 0)
        ma10 = self._num(row.get("ma10"), 0)
        ma20 = self._num(row.get("ma20"), 0)
        if close <= 0 or ma20 <= 0:
            return "均线数据不足"

        dist5 = self._num(row.get("distance_ma5"), 0)
        dist10 = self._num(row.get("distance_ma10"), 0)
        dist20 = self._num(row.get("distance_ma20"), 0)

        if close < ma20:
            relation = "跌破20日线，买点失效"
        elif ma5 > 0 and low > 0 and low <= ma5 * 1.015 and close >= ma5:
            relation = "回踩5日线后收回"
        elif ma10 > 0 and low > 0 and low <= ma10 * 1.02 and close >= ma10:
            relation = "回踩10日线后收回"
        elif low > 0 and low <= ma20 * 1.025 and close >= ma20:
            relation = "回踩20日线后收回"
        elif dist20 >= 15:
            relation = "距20日线偏离较大"
        elif dist10 >= 8:
            relation = "距10日线偏离偏高"
        elif ma5 > 0 and close < ma5:
            relation = "短线跌破5日线但守20日"
        else:
            relation = "5/10/20日线上方运行"

        return f"{relation}；距MA5 {dist5:+.1f}%，距MA10 {dist10:+.1f}%，距MA20 {dist20:+.1f}%"

    def _to_row(self, row: pd.Series) -> dict[str, Any]:
        stop = self._stop_hint(row)
        return {
            "code": row.get("symbol", ""),
            "name": row.get("name", ""),
            "stock_type": row.get("stock_type", ""),
            "industry": row.get("industry", ""),
            "concepts": row.get("concepts", ""),
            "mainline_industry": row.get("mainline_industry", row.get("industry", "")),
            "mainline_industry_source": row.get("mainline_industry_source", ""),
            "mainline_state": row.get("mainline_state", ""),
            "mainline_score": round(self._num(row.get("mainline_score"), 50), 1),
            "base_score": round(self._num(row.get("base_score"), 0), 2),
            "industry_rank": int(self._num(row.get("industry_rank"), 0)),
            "industry_rank_change_5": int(self._num(row.get("industry_rank_change_5"), 0)),
            "industry_rps20": round(self._num(row.get("industry_rps20"), 50), 1),
            "industry_rps60": round(self._num(row.get("industry_rps60"), 50), 1),
            "industry_rps120": round(self._num(row.get("industry_rps120"), 50), 1),
            "industry_rps_combo": round(self._num(row.get("industry_rps_combo"), 50), 1),
            "industry_breadth_score": round(self._num(row.get("industry_breadth_score"), 50), 1),
            "industry_persistence_score": round(self._num(row.get("industry_persistence_score"), 50), 1),
            "industry_above_ma20_ratio": round(self._num(row.get("industry_above_ma20_ratio"), 0) * 100, 1),
            "industry_above_ma50_ratio": round(self._num(row.get("industry_above_ma50_ratio"), 0) * 100, 1),
            "industry_strong_ratio": round(self._num(row.get("industry_strong_ratio"), 0) * 100, 1),
            "mainline_reason": row.get("mainline_reason", ""),
            "setup": row.get("setup", ""),
            "channel": row.get("channel", ""),
            "score": round(self._num(row.get("score"), 0), 2),
            "c_score": round(self._num(row.get("c_score"), 50), 1),
            "a_score": round(self._num(row.get("a_score"), 50), 1),
            "n_score": round(self._num(row.get("n_score"), 0), 1),
            "s_score": round(self._num(row.get("s_score"), 0), 1),
            "l_score": round(self._num(row.get("l_score"), 0), 1),
            "i_score": round(self._num(row.get("i_score"), 0), 1),
            "m_score": round(self._num(row.get("m_score"), 0), 1),
            "close": round(self._num(row.get("close"), 0), 2),
            "pct_chg": round(self._num(row.get("pct_chg"), 0), 2),
            "amount_yi": round(self._num(row.get("amount_yi"), 0), 2),
            "amount_ratio": round(self._num(row.get("amount_ratio"), 1), 2),
            "turnover_rate_f": round(self._num(row.get("turnover_rate_f"), 0), 2),
            "market_cap_yi": round(self._num(row.get("market_cap_yi"), 0), 2),
            "rps50": round(self._num(row.get("rps50"), 0), 1),
            "rps120": round(self._num(row.get("rps120"), 0), 1),
            "rps250": round(self._num(row.get("rps250"), 0), 1),
            "rps_combo": round(self._num(row.get("rps_combo"), 0), 1),
            "ret50": round(self._num(row.get("ret50"), 0), 2),
            "ret120": round(self._num(row.get("ret120"), 0), 2),
            "ret250": round(self._num(row.get("ret250"), 0), 2),
            "high_position_250": round(self._num(row.get("high_position_250"), 0), 3),
            "ma5": round(self._num(row.get("ma5"), 0), 2),
            "ma10": round(self._num(row.get("ma10"), 0), 2),
            "ma20": round(self._num(row.get("ma20"), 0), 2),
            "distance_ma5": round(self._num(row.get("distance_ma5"), 0), 2),
            "distance_ma10": round(self._num(row.get("distance_ma10"), 0), 2),
            "distance_ma20": round(self._num(row.get("distance_ma20"), 0), 2),
            "distance_ma50": round(self._num(row.get("distance_ma50"), 0), 2),
            "ma_pullback": self._price_ma_label(row),
            "ma_stack": bool(row.get("ma_stack", False)),
            "new_high_5": bool(row.get("new_high_5", False)),
            "pocket_pivot": bool(row.get("pocket_pivot", False)),
            "eps_yoy": self._round_or_blank(row.get("eps_yoy")),
            "basic_eps_yoy": self._round_or_blank(row.get("basic_eps_yoy")),
            "dt_eps_yoy": self._round_or_blank(row.get("dt_eps_yoy")),
            "q_eps": self._round_or_blank(row.get("q_eps")),
            "profit_yoy": self._round_or_blank(row.get("profit_yoy")),
            "sales_yoy": self._round_or_blank(row.get("sales_yoy")),
            "annual_eps_yoy": self._round_or_blank(row.get("annual_eps_yoy")),
            "annual_roe": self._round_or_blank(row.get("annual_roe")),
            "report_date": "" if pd.isna(row.get("end_date", pd.NA)) else str(row.get("end_date")),
            "finance_status": row.get("finance_status", ""),
            "finance_note": row.get("finance_note", ""),
            "tao_institution_pass": "是" if bool(row.get("tao_institution_pass", False)) else "否",
            "fund_hold_ratio": self._round_or_blank(row.get("fund_hold_ratio")),
            "fund_hold_period": "" if pd.isna(row.get("fund_hold_period", pd.NA)) else str(row.get("fund_hold_period")),
            "north_hold_value_wan": self._round_or_blank(row.get("north_hold_value_wan")),
            "north_hold_ratio": self._round_or_blank(row.get("north_hold_ratio")),
            "north_hold_period": "" if pd.isna(row.get("north_hold_period", pd.NA)) else str(row.get("north_hold_period")),
            "tao_institution_note": row.get("tao_institution_note", ""),
            "stop_hint": stop,
            "reason": self._reason(row),
        }

    def _reason(self, row: pd.Series) -> str:
        parts = []
        channel = str(row.get("channel", "")).strip()
        if channel:
            parts.append(channel)
        mainline_state = str(row.get("mainline_state", ""))
        if mainline_state in {MAINLINE_CONFIRMED, MAINLINE_CANDIDATE}:
            parts.append(f"{mainline_state}:{row.get('mainline_industry', row.get('industry', ''))}")
        if self._canslim_pass(row):
            parts.append("CANSLIM财务/形态达标")
        if self._tao_pass(row):
            if bool(getattr(self.config, "tao_institution_enabled", True)):
                parts.append("RPS强且机构条件通过")
            else:
                parts.append("RPS强且接近一年新高")
        if bool(row.get("new_high_5")):
            parts.append("5日内新高")
        if bool(row.get("pocket_pivot")):
            parts.append("口袋支点")
        if bool(row.get("ma_stack")):
            parts.append("均线多头")
        ma_pullback = self._price_ma_label(row)
        if ma_pullback and ma_pullback != "均线数据不足":
            parts.append(ma_pullback.split("；", 1)[0])
        eps_yoy = self._num(row.get("eps_yoy"), None)
        if eps_yoy is not None and eps_yoy >= float(self.config.canslim_min_eps_yoy):
            parts.append(f"EPS同比{eps_yoy:.1f}%")
        if self._num(row.get("c_score"), 50) >= 70:
            parts.append("近期成长高")
        if self._num(row.get("m_score"), 50) >= 60:
            parts.append("市场环境支持")
        return "；".join(parts[:5])

    def _stop_hint(self, row: pd.Series) -> str:
        setup = str(row.get("setup", ""))
        ma20 = self._num(row.get("ma20"), 0)
        ma50 = self._num(row.get("ma50"), 0)
        low = self._num(row.get("low"), 0)
        if setup in {"新高突破", "平台突破"}:
            return f"收盘跌破20日线 {ma20:.2f} 或突破日低点 {low:.2f}"
        if setup == "均线回踩":
            return f"收盘跌破50日线 {ma50:.2f}"
        if setup == "口袋支点":
            return f"收盘跌回支点日低点 {low:.2f}"
        return f"跌破20日线 {ma20:.2f} 后降级观察"

    def _listed_enough(self, list_date: str) -> bool:
        try:
            listed = datetime.strptime(str(list_date)[:8], "%Y%m%d").date()
            end_date = datetime.strptime(str(self.config.end_date)[:8], "%Y%m%d").date()
        except (TypeError, ValueError):
            return True
        return (end_date - listed).days >= 180

    @staticmethod
    def _clip(value: Any, low: float, high: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(number) or high <= low:
            return 0.0
        return max(0.0, min(100.0, (number - low) / (high - low) * 100))

    @staticmethod
    def _num(value: Any, default: float | None = 0.0) -> float | None:
        try:
            if pd.isna(value):
                return default
            number = float(value)
            if math.isnan(number):
                return default
            return number
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _round_or_blank(value: Any) -> float | str:
        if pd.isna(value):
            return ""
        try:
            return round(float(value), 2)
        except (TypeError, ValueError):
            return ""
