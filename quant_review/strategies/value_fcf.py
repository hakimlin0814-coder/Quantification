from __future__ import annotations

import math
import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import pandas as pd


class ValueFcfStrategy:
    """Low valuation, high ROE, and free-cash-flow yield strategy."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.config = engine.config
        self.financial_status = "未读取"
        self.financial_note = ""
        self.financial_rows = 0
        self.roe_rows = 0
        self.cashflow_rows = 0

    def log(self, message: str) -> None:
        self.engine.log(message)

    def run(self) -> dict[str, Any]:
        old_lookback = int(self.config.lookback_days)
        self.config.lookback_days = max(old_lookback, 70)
        try:
            self.log("读取交易日历，低估值现金流策略需要最近约70个交易日...")
            trade_dates = self.engine._trade_dates()
        finally:
            self.config.lookback_days = old_lookback

        trade_date = trade_dates[-1]
        self.config.end_date = trade_date
        self.log(f"使用交易日 {trade_date}")

        self.log("读取股票基础信息...")
        basic = self.engine._stock_basic()

        self.log("读取近期日线行情...")
        daily = self.engine._daily_window(trade_dates)
        if self.config.review_mode == "intraday":
            self.log("盘中模式：读取实时行情并合成临时日K...")
            daily = self.engine._apply_intraday_snapshot(daily, basic, trade_date)

        self.log("读取估值指标 PB / PE_TTM...")
        daily_basic = self.engine._daily_basic(trade_date)
        if self.config.review_mode == "intraday" and self._daily_basic_value_is_empty(daily_basic) and len(trade_dates) >= 2:
            prev_date = trade_dates[-2]
            self.log(f"盘中模式：当日估值指标未更新，回退到 {prev_date}。")
            daily_basic = self.engine._daily_basic(prev_date)

        self.log("读取 ROE 和现金流量表，计算 FCF率...")
        fundamentals = self._financial_metrics(trade_date)

        latest_daily = self._latest_daily(daily, trade_date)
        self.log("合成低估值现金流候选池...")
        rows = self._rows(basic, latest_daily, daily_basic, fundamentals)
        probe = self._probe(getattr(self, "_last_value_data", pd.DataFrame()))

        self.log(f"低估值现金流策略完成，共 {len(rows)} 只。")
        return {
            "ok": True,
            "trade_date": trade_date,
            "financial": {
                "status": self.financial_status,
                "note": self.financial_note,
                "rows": self.financial_rows,
                "roe_rows": self.roe_rows,
                "cashflow_rows": self.cashflow_rows,
            },
            "rows": rows,
            "probe": probe,
            "logs": self.engine.logs,
        }

    def calculate_single_fcf(self) -> dict[str, Any]:
        query = str(getattr(self.config, "value_probe_query", "") or "").strip()
        if not query:
            raise RuntimeError("请输入股票代码或名称")

        old_lookback = int(self.config.lookback_days)
        self.config.lookback_days = max(old_lookback, 70)
        try:
            self.log("读取交易日历，用最近可交易日计算单股 FCF...")
            trade_dates = self.engine._trade_dates()
        finally:
            self.config.lookback_days = old_lookback

        trade_date = trade_dates[-1]
        self.config.end_date = trade_date
        self.log(f"使用交易日 {trade_date}")

        self.log(f"读取股票基础信息，查找 {query}...")
        basic = self.engine._stock_basic()
        match = self._match_stock(basic, query)
        if match is None:
            return {
                "ok": True,
                "query": query,
                "found": False,
                "calculated": False,
                "message": f"未找到 {query}",
                "trade_date": trade_date,
                "logs": self.engine.logs,
            }

        ts_code = str(match.get("ts_code", ""))
        symbol = str(match.get("symbol", ts_code[:6])).zfill(6)
        name = str(match.get("name", ""))
        self.log(f"找到 {name}({symbol})，读取估值和现金流...")

        daily_basic = self.engine._daily_basic(trade_date)
        valuation_trade_date = trade_date
        if self._daily_basic_value_is_empty(daily_basic) and len(trade_dates) >= 2:
            valuation_trade_date = trade_dates[-2]
            self.log(f"{trade_date} 估值数据未更新，回退到 {valuation_trade_date}。")
            daily_basic = self.engine._daily_basic(valuation_trade_date)

        cashflow = self._cashflow_metrics(self._report_periods(trade_date))
        self.cashflow_rows = len(cashflow)
        cashflow_match = self._row_by_ts_code(cashflow, ts_code)
        db_row = self._row_by_ts_code(daily_basic, ts_code)

        stock = {
            "code": symbol,
            "ts_code": ts_code,
            "name": name,
            "industry": str(match.get("industry", "")),
            "market": str(match.get("market", "")),
            "stock_type": str(match.get("stock_type", "")),
        }

        if cashflow_match is None:
            message = f"{name}({symbol}) 暂无可用现金流数据"
            self.log(message)
            return {
                "ok": True,
                "query": query,
                "found": True,
                "calculated": False,
                "message": message,
                "trade_date": trade_date,
                "stock": stock,
                "logs": self.engine.logs,
            }

        total_mv = self._num(db_row.get("total_mv") if db_row is not None else None, None)
        market_cap_yi = total_mv / 10000 if total_mv and total_mv > 0 else None
        fcf_ttm = self._num(cashflow_match.get("fcf_ttm"), None)
        latest_fcf = self._num(cashflow_match.get("fcf"), None)
        fcf_yield = fcf_ttm / (total_mv * 10000) * 100 if fcf_ttm is not None and total_mv and total_mv > 0 else None
        pe_ttm = self._num(db_row.get("pe_ttm") if db_row is not None else None, None)
        pe = self._num(db_row.get("pe") if db_row is not None else None, None)
        pe_value = pe_ttm if pe_ttm is not None and pe_ttm > 0 else pe
        pb = self._num(db_row.get("pb") if db_row is not None else None, None)

        result = {
            **stock,
            "trade_date": trade_date,
            "valuation_trade_date": valuation_trade_date if db_row is not None else "",
            "cashflow_report_date": str(cashflow_match.get("cashflow_report_date", "")),
            "ann_date": "" if pd.isna(cashflow_match.get("ann_date", pd.NA)) else str(cashflow_match.get("ann_date", "")),
            "market_cap_yi": self._round_or_blank(market_cap_yi),
            "pb": self._round_or_blank(pb),
            "pe_ttm": self._round_or_blank(pe_value),
            "fcf_ttm_yi": self._round_or_blank(fcf_ttm / 100000000 if fcf_ttm is not None else pd.NA),
            "latest_fcf_yi": self._round_or_blank(latest_fcf / 100000000 if latest_fcf is not None else pd.NA),
            "fcf_yield": self._round_or_blank(fcf_yield),
            "fcf_source": "" if pd.isna(cashflow_match.get("fcf_source", pd.NA)) else str(cashflow_match.get("fcf_source", "")),
            "fcf_note": "" if pd.isna(cashflow_match.get("fcf_note", pd.NA)) else str(cashflow_match.get("fcf_note", "")),
        }
        result["formula"] = self._single_fcf_formula(result)
        self.log(f"{name}({symbol}) FCF计算完成。")
        return {
            "ok": True,
            "query": query,
            "found": True,
            "calculated": True,
            "trade_date": trade_date,
            "result": result,
            "logs": self.engine.logs,
        }

    @staticmethod
    def _daily_basic_value_is_empty(daily_basic: pd.DataFrame) -> bool:
        if daily_basic.empty or "total_mv" not in daily_basic.columns:
            return True
        total_mv = pd.to_numeric(daily_basic["total_mv"], errors="coerce").fillna(0)
        return bool(total_mv.sum() <= 0)

    def _latest_daily(self, daily: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        latest = daily[daily["trade_date"].astype(str) == str(trade_date)].copy()
        if latest.empty:
            return pd.DataFrame(columns=["ts_code", "close", "pct_chg", "amount_yi"])
        for col in ["close", "pct_chg", "amount"]:
            latest[col] = pd.to_numeric(latest[col], errors="coerce")
        latest["amount_yi"] = latest["amount"].fillna(0) / 100000
        return latest[["ts_code", "close", "pct_chg", "amount_yi"]]

    def _financial_metrics(self, trade_date: str) -> pd.DataFrame:
        periods = self._report_periods(trade_date)
        if not periods:
            self.financial_status = "无报告期"
            self.financial_note = "未推导出可读取的报告期"
            return pd.DataFrame()

        roe = self._roe_metrics(periods)
        cashflow = self._cashflow_metrics(periods)
        self.roe_rows = len(roe)
        self.cashflow_rows = len(cashflow)

        if roe.empty and cashflow.empty:
            self.financial_status = "读取失败"
            self.financial_note = "ROE 和现金流量表均为空，可能是接口权限或字段限制"
            self.log(self.financial_note)
            return pd.DataFrame()
        if roe.empty:
            result = cashflow.copy()
        elif cashflow.empty:
            result = roe.copy()
        else:
            result = roe.merge(cashflow, on="ts_code", how="outer")

        self.financial_rows = len(result)
        if not roe.empty and not cashflow.empty:
            self.financial_status = "已读取"
            self.financial_note = f"ROE覆盖{len(roe)}只，现金流覆盖{len(cashflow)}只"
        elif not roe.empty:
            self.financial_status = "缺现金流"
            self.financial_note = f"仅ROE覆盖{len(roe)}只，现金流量表为空"
        else:
            self.financial_status = "缺ROE"
            self.financial_note = f"仅现金流覆盖{len(cashflow)}只，ROE为空"
        self.log(self.financial_note)
        return result

    def _roe_metrics(self, periods: list[str]) -> pd.DataFrame:
        frames = []
        failures = []
        for period in periods:
            try:
                frame = self.engine._fina_indicator(period)
            except Exception as exc:
                failures.append(f"{period} ROE读取失败: {exc}")
                continue
            if not frame.empty and "ts_code" in frame.columns:
                frames.append(frame)
        if not frames:
            fallback = self._eastmoney_roe_metrics(periods, failures)
            if not fallback.empty:
                return fallback
            if failures:
                self.log(f"ROE读取失败: {failures[-1]}")
            return pd.DataFrame()

        data = pd.concat(frames, ignore_index=True)
        if "end_date" not in data.columns:
            return pd.DataFrame()
        data["ts_code"] = data["ts_code"].astype(str)
        data["end_date"] = data["end_date"].astype(str)
        if "ann_date" not in data.columns:
            data["ann_date"] = ""
        for col in ["roe", "roe_dt"]:
            if col not in data.columns:
                data[col] = pd.NA
            data[col] = pd.to_numeric(data[col], errors="coerce")

        latest = data.sort_values(["ts_code", "end_date", "ann_date"]).drop_duplicates(["ts_code", "end_date"], keep="last")
        latest = latest.sort_values("end_date").groupby("ts_code", as_index=False).tail(1).copy()
        latest["roe_value"] = latest["roe"].combine_first(latest["roe_dt"])
        latest["roe_source"] = latest.apply(lambda row: "roe" if not pd.isna(row.get("roe")) else "roe_dt", axis=1)
        return latest[["ts_code", "end_date", "ann_date", "roe", "roe_dt", "roe_value", "roe_source"]].rename(columns={"end_date": "report_date"})

    def _cashflow_metrics(self, periods: list[str]) -> pd.DataFrame:
        frames = []
        failures = []
        for period in periods:
            try:
                frame = self.engine._cashflow(period)
            except Exception as exc:
                failures.append(f"{period} 现金流读取失败: {exc}")
                continue
            if not frame.empty and "ts_code" in frame.columns:
                frames.append(frame)
        if not frames:
            fallback = self._eastmoney_cashflow_metrics(periods, failures)
            if not fallback.empty:
                return fallback
            if failures:
                self.log(f"现金流读取失败: {failures[-1]}")
            return pd.DataFrame()

        data = pd.concat(frames, ignore_index=True)
        if "end_date" not in data.columns:
            return pd.DataFrame()
        data["ts_code"] = data["ts_code"].astype(str)
        data["end_date"] = data["end_date"].astype(str)
        if "ann_date" not in data.columns:
            data["ann_date"] = ""
        for col in ["n_cashflow_act", "c_pay_acq_const_fiolta", "free_cashflow"]:
            if col not in data.columns:
                data[col] = pd.NA
            data[col] = pd.to_numeric(data[col], errors="coerce")

        computed_fcf = data["n_cashflow_act"] - data["c_pay_acq_const_fiolta"]
        data["fcf_source"] = "经营现金流-购建固定资产等现金支出"
        data.loc[~data["free_cashflow"].isna(), "fcf_source"] = "Tushare free_cashflow"
        data["fcf"] = data["free_cashflow"].combine_first(computed_fcf)
        data = data.sort_values(["ts_code", "end_date", "ann_date"]).drop_duplicates(["ts_code", "end_date"], keep="last")
        data = data[~data["fcf"].isna()].copy()
        if data.empty:
            return pd.DataFrame()

        lookup = {
            (str(row.ts_code), str(row.end_date)): self._num(row.fcf, None)
            for row in data.itertuples(index=False)
        }
        latest = data.sort_values("end_date").groupby("ts_code", as_index=False).tail(1).copy()
        values = latest.apply(lambda row: self._ttm_fcf(row, lookup), axis=1)
        latest["fcf_ttm"] = [item[0] for item in values]
        latest["fcf_note"] = [item[1] for item in values]
        latest = latest[~latest["fcf_ttm"].isna()].copy()
        return latest[["ts_code", "end_date", "ann_date", "fcf_ttm", "fcf", "fcf_source", "fcf_note"]].rename(columns={"end_date": "cashflow_report_date"})

    def _ttm_fcf(self, row: pd.Series, lookup: dict[tuple[str, str], float | None]) -> tuple[float | Any, str]:
        ts_code = str(row.get("ts_code", ""))
        period = str(row.get("end_date", ""))
        current = self._num(row.get("fcf"), None)
        if current is None or len(period) != 8:
            return pd.NA, "缺本期FCF"
        suffix = period[4:]
        if suffix == "1231":
            return current, "年报FCF"

        year = int(period[:4])
        prev_annual = lookup.get((ts_code, f"{year - 1}1231"))
        prev_same = lookup.get((ts_code, f"{year - 1}{suffix}"))
        if prev_annual is not None and prev_same is not None:
            return current + prev_annual - prev_same, "TTM=本期累计+上年年报-上年同期"
        return current, "最新累计FCF，缺少TTM所需同期数据"

    def _rows(
        self,
        basic: pd.DataFrame,
        latest_daily: pd.DataFrame,
        daily_basic: pd.DataFrame,
        fundamentals: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        basic_cols = ["ts_code", "symbol", "name", "industry", "market", "stock_type"]
        data = basic[[col for col in basic_cols if col in basic.columns]].copy()
        data = data.merge(latest_daily, on="ts_code", how="left")

        db = daily_basic.copy()
        if "ts_code" in db.columns:
            db["ts_code"] = db["ts_code"].astype(str)
        else:
            db["ts_code"] = pd.Series(dtype=str)
        for col in ["pe", "pe_ttm", "pb", "total_mv", "circ_mv", "turnover_rate", "turnover_rate_f", "volume_ratio"]:
            if col not in db.columns:
                db[col] = pd.NA
            db[col] = pd.to_numeric(db[col], errors="coerce")
        data = data.merge(
            db[["ts_code", "pe", "pe_ttm", "pb", "total_mv", "circ_mv", "turnover_rate", "turnover_rate_f", "volume_ratio"]],
            on="ts_code",
            how="left",
        )

        if not fundamentals.empty:
            data = data.merge(fundamentals, on="ts_code", how="left")
        else:
            data["roe_value"] = pd.NA
            data["fcf_ttm"] = pd.NA
            data["fcf_note"] = ""
            data["report_date"] = ""
            data["cashflow_report_date"] = ""
        for col in ["roe_value", "fcf_ttm", "fcf_note", "report_date", "cashflow_report_date"]:
            if col not in data.columns:
                data[col] = "" if col in {"fcf_note", "report_date", "cashflow_report_date"} else pd.NA

        data["name"] = data["name"].fillna("")
        data["symbol"] = data["symbol"].fillna(data["ts_code"].str.slice(0, 6))
        data["industry"] = data["industry"].fillna("未分类").replace("", "未分类")
        data["pe_value"] = pd.to_numeric(data["pe_ttm"], errors="coerce")
        pe = pd.to_numeric(data["pe"], errors="coerce")
        data.loc[(data["pe_value"].isna() | (data["pe_value"] <= 0)) & (pe > 0), "pe_value"] = pe
        data["pb"] = pd.to_numeric(data["pb"], errors="coerce")
        data["roe_value"] = pd.to_numeric(data["roe_value"], errors="coerce")
        data["total_mv"] = pd.to_numeric(data["total_mv"], errors="coerce")
        data["market_cap_yi"] = data["total_mv"] / 10000
        data["fcf_ttm"] = pd.to_numeric(data["fcf_ttm"], errors="coerce")
        data["fcf_yield"] = data["fcf_ttm"] / (data["total_mv"] * 10000) * 100
        data["value_score"] = data.apply(self._value_score, axis=1)
        self._last_value_data = data.copy()

        data = self._apply_filters(data)
        if data.empty:
            return []

        data = data.sort_values(["fcf_yield", "roe_value", "pe_value", "pb"], ascending=[False, False, True, True])
        return [self._to_row(row) for _, row in data.head(int(self.config.value_top_count)).iterrows()]

    def _probe(self, data: pd.DataFrame) -> dict[str, Any]:
        query = str(getattr(self.config, "value_probe_query", "") or "").strip()
        if not query:
            return {}
        if data.empty:
            return {"query": query, "found": False, "message": "没有可分析的数据"}
        match = self._match_stock(data, query)
        if match is None:
            return {"query": query, "found": False, "message": f"未找到 {query}"}

        checks = self._value_checks(match)
        enabled = [item for item in checks if not item.get("disabled")]
        overall_pass = bool(enabled) and all(bool(item.get("pass")) for item in enabled)
        row = self._to_row(match)
        row["probe_result"] = "符合" if overall_pass else "不符合"
        failed = [item["label"] for item in enabled if not item.get("pass")]
        return {
            "query": query,
            "found": True,
            "overall_pass": overall_pass,
            "summary": "全部达标" if overall_pass else "未达标：" + "、".join(failed),
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

    @staticmethod
    def _row_by_ts_code(data: pd.DataFrame, ts_code: str) -> pd.Series | None:
        if data is None or data.empty or "ts_code" not in data.columns:
            return None
        matched = data[data["ts_code"].astype(str) == str(ts_code)]
        if matched.empty:
            return None
        return matched.iloc[0]

    @staticmethod
    def _single_fcf_formula(result: dict[str, Any]) -> str:
        parts = []
        source = str(result.get("fcf_source", "") or "")
        note = str(result.get("fcf_note", "") or "")
        if source:
            parts.append(f"FCF={source}")
        if note:
            parts.append(note)
        if result.get("fcf_yield") != "" and result.get("market_cap_yi") != "":
            parts.append("FCF率=TTM FCF/总市值")
        return "；".join(parts)

    def _value_checks(self, row: pd.Series) -> list[dict[str, Any]]:
        checks = []
        checks.append(self._check("总市值", row.get("market_cap_yi"), ">", 0, self._num(row.get("market_cap_yi"), 0) > 0, "亿"))

        max_pb = float(self.config.value_max_pb)
        max_pe = float(self.config.value_max_pe_ttm)
        min_roe = float(self.config.value_min_roe)
        min_fcf = float(self.config.value_min_fcf_yield)
        if max_pb > 0:
            pb = self._num(row.get("pb"), None)
            checks.append(self._check("PB", pb, "<", max_pb, pb is not None and pb > 0 and pb < max_pb))
        else:
            checks.append(self._disabled_check("PB"))
        if max_pe > 0:
            pe = self._num(row.get("pe_value"), None)
            checks.append(self._check("PE_TTM", pe, "<", max_pe, pe is not None and pe > 0 and pe < max_pe))
        else:
            checks.append(self._disabled_check("PE_TTM"))
        if min_roe > 0:
            roe = self._num(row.get("roe_value"), None)
            checks.append(self._check("ROE", roe, ">", min_roe, roe is not None and roe > min_roe, "%"))
        else:
            checks.append(self._disabled_check("ROE"))
        if min_fcf > 0:
            fcf = self._num(row.get("fcf_yield"), None)
            checks.append(self._check("FCF率", fcf, ">", min_fcf, fcf is not None and fcf > min_fcf, "%"))
        else:
            checks.append(self._disabled_check("FCF率"))

        name = str(row.get("name", ""))
        symbol = str(row.get("symbol", ""))
        if self.config.value_exclude_st:
            checks.append(self._check("排除ST", name, "not contains", "ST", "ST" not in name.upper()))
        if self.config.value_exclude_star:
            checks.append(self._check("排除科创板", symbol, "not starts", "688/689", not symbol.startswith(("688", "689"))))
        if self.config.value_exclude_chinext:
            checks.append(self._check("排除创业板", symbol, "not starts", "300/301", not symbol.startswith(("300", "301"))))
        checks.append(self._check("排除北交所", symbol, "not starts", "8/4", not symbol.startswith(("8", "4"))))
        return checks

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

    @staticmethod
    def _disabled_check(label: str) -> dict[str, Any]:
        return {"label": label, "value": "", "op": "", "threshold": "不限制", "unit": "", "pass": True, "disabled": True}

    def _apply_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result = result[result["market_cap_yi"].fillna(0) > 0]

        max_pb = float(self.config.value_max_pb)
        max_pe = float(self.config.value_max_pe_ttm)
        min_roe = float(self.config.value_min_roe)
        min_fcf = float(self.config.value_min_fcf_yield)

        if max_pb > 0:
            result = result[(result["pb"] > 0) & (result["pb"] < max_pb)]
        if max_pe > 0:
            result = result[(result["pe_value"] > 0) & (result["pe_value"] < max_pe)]
        if min_roe > 0:
            result = result[result["roe_value"] > min_roe]
        if min_fcf > 0:
            result = result[result["fcf_yield"] > min_fcf]

        if self.config.value_exclude_st:
            result = result[~result["name"].astype(str).str.upper().str.contains("ST", regex=False)]
        if self.config.value_exclude_star:
            result = result[~result["symbol"].astype(str).str.startswith(("688", "689"))]
        if self.config.value_exclude_chinext:
            result = result[~result["symbol"].astype(str).str.startswith(("300", "301"))]
        result = result[~result["symbol"].astype(str).str.startswith(("8", "4"))]
        return result

    def _value_score(self, row: pd.Series) -> float:
        pb_score = self._inverse_score(row.get("pb"), float(self.config.value_max_pb))
        pe_score = self._inverse_score(row.get("pe_value"), float(self.config.value_max_pe_ttm))
        roe_score = self._scale(row.get("roe_value"), float(self.config.value_min_roe), max(float(self.config.value_min_roe) + 15, 30))
        fcf_score = self._scale(row.get("fcf_yield"), float(self.config.value_min_fcf_yield), max(float(self.config.value_min_fcf_yield) + 8, 12))
        return round(pb_score * 0.20 + pe_score * 0.20 + roe_score * 0.30 + fcf_score * 0.30, 2)

    def _to_row(self, row: pd.Series) -> dict[str, Any]:
        return {
            "code": row.get("symbol", ""),
            "name": row.get("name", ""),
            "stock_type": row.get("stock_type", ""),
            "industry": row.get("industry", ""),
            "close": self._round_or_blank(row.get("close")),
            "pct_chg": self._round_or_blank(row.get("pct_chg")),
            "amount_yi": self._round_or_blank(row.get("amount_yi")),
            "market_cap_yi": self._round_or_blank(row.get("market_cap_yi")),
            "pb": self._round_or_blank(row.get("pb")),
            "pe_ttm": self._round_or_blank(row.get("pe_value")),
            "roe": self._round_or_blank(row.get("roe_value")),
            "fcf_yield": self._round_or_blank(row.get("fcf_yield")),
            "fcf_ttm_yi": self._round_or_blank(self._num(row.get("fcf_ttm"), 0) / 100000000),
            "value_score": self._round_or_blank(row.get("value_score")),
            "report_date": "" if pd.isna(row.get("report_date", pd.NA)) else str(row.get("report_date")),
            "cashflow_report_date": "" if pd.isna(row.get("cashflow_report_date", pd.NA)) else str(row.get("cashflow_report_date")),
            "fcf_note": "" if pd.isna(row.get("fcf_note", pd.NA)) else str(row.get("fcf_note")),
            "reason": self._reason(row),
        }

    def _reason(self, row: pd.Series) -> str:
        parts = []
        pb = self._num(row.get("pb"), None)
        pe = self._num(row.get("pe_value"), None)
        roe = self._num(row.get("roe_value"), None)
        fcf_yield = self._num(row.get("fcf_yield"), None)
        if pb is not None and float(self.config.value_max_pb) > 0:
            parts.append(f"PB {pb:.2f}<{float(self.config.value_max_pb):g}")
        if pe is not None and float(self.config.value_max_pe_ttm) > 0:
            parts.append(f"PE_TTM {pe:.2f}<{float(self.config.value_max_pe_ttm):g}")
        if roe is not None and float(self.config.value_min_roe) > 0:
            parts.append(f"ROE {roe:.1f}%>{float(self.config.value_min_roe):g}%")
        if fcf_yield is not None and float(self.config.value_min_fcf_yield) > 0:
            parts.append(f"FCF率 {fcf_yield:.1f}%>{float(self.config.value_min_fcf_yield):g}%")
        return "；".join(parts)

    def _report_periods(self, trade_date: str) -> list[str]:
        year = int(trade_date[:4])
        candidates = []
        for item_year in range(year - 3, year + 1):
            for suffix in ["0331", "0630", "0930", "1231"]:
                period = f"{item_year}{suffix}"
                if period <= trade_date:
                    candidates.append(period)
        quarter_periods = candidates[-10:]
        annual_periods = [period for period in candidates if period.endswith("1231")][-4:]
        return sorted(set(quarter_periods + annual_periods))

    def _eastmoney_roe_metrics(self, periods: list[str], failures: list[str]) -> pd.DataFrame:
        try:
            import akshare as ak
        except Exception as exc:
            failures.append(f"东财ROE兜底不可用: akshare导入失败: {exc}")
            return pd.DataFrame()

        frames = []
        for period in periods:
            frame = self._eastmoney_yjbb(ak, period)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            failures.append("东财业绩报表无ROE数据")
            return pd.DataFrame()

        data = pd.concat(frames, ignore_index=True)
        data["report_date"] = data["report_date"].astype(str)
        latest = data.sort_values("report_date").groupby("ts_code", as_index=False).tail(1).copy()
        latest["roe"] = latest["roe_value"]
        latest["roe_dt"] = pd.NA
        latest["roe_source"] = "eastmoney_yjbb"
        self.log(f"ROE使用东财业绩报表兜底，覆盖{len(latest)}只。")
        return latest[["ts_code", "report_date", "ann_date", "roe", "roe_dt", "roe_value", "roe_source"]]

    def _eastmoney_yjbb(self, ak: Any, period: str) -> pd.DataFrame:
        path = self.engine._cache_path(f"eastmoney_yjbb_{period}")
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
        data["roe_value"] = pd.to_numeric(data.get("净资产收益率", pd.NA), errors="coerce")
        data["ann_date"] = data.get("最新公告日期", "")
        data["report_date"] = period
        data = data[~data["roe_value"].isna()].copy()
        return data[["ts_code", "report_date", "ann_date", "roe_value"]]

    def _eastmoney_cashflow_metrics(self, periods: list[str], failures: list[str]) -> pd.DataFrame:
        try:
            import akshare as ak
        except Exception as exc:
            failures.append(f"东财现金流兜底不可用: akshare导入失败: {exc}")
            return pd.DataFrame()

        frames = []
        for period in periods:
            frame = self._eastmoney_xjll(ak, period)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            failures.append("东财现金流量表无数据")
            return pd.DataFrame()

        data = pd.concat(frames, ignore_index=True)
        data["cashflow_report_date"] = data["cashflow_report_date"].astype(str)
        lookup = {
            (str(row.ts_code), str(row.cashflow_report_date)): self._num(row.fcf, None)
            for row in data.itertuples(index=False)
        }
        latest = data.sort_values("cashflow_report_date").groupby("ts_code", as_index=False).tail(1).copy()
        values = latest.apply(
            lambda row: self._ttm_fcf(pd.Series({"ts_code": row["ts_code"], "end_date": row["cashflow_report_date"], "fcf": row["fcf"]}), lookup),
            axis=1,
        )
        latest["fcf_ttm"] = [item[0] for item in values]
        latest["fcf_source"] = "东财近似FCF=经营现金流+投资现金流"
        latest["fcf_note"] = [f"东财近似FCF=经营现金流+投资现金流；{item[1]}" for item in values]
        latest = latest[~latest["fcf_ttm"].isna()].copy()
        self.log(f"现金流使用东财现金流量表兜底，覆盖{len(latest)}只。")
        return latest[["ts_code", "cashflow_report_date", "ann_date", "fcf_ttm", "fcf", "fcf_source", "fcf_note"]]

    def _eastmoney_xjll(self, ak: Any, period: str) -> pd.DataFrame:
        path = self.engine._cache_path(f"eastmoney_xjll_{period}")
        if path.exists():
            try:
                cached = pd.read_csv(path, dtype=str)
            except pd.errors.EmptyDataError:
                self.log(f"{path.name} 缓存为空，已删除并重新读取。")
                path.unlink(missing_ok=True)
                cached = pd.DataFrame()
            if cached.empty:
                return pd.DataFrame()
            return self._normalize_eastmoney_xjll(cached, period)
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                frame = ak.stock_xjll_em(date=period)
        except Exception as exc:
            self.log(f"东财{period}现金流量表读取失败: {exc}")
            return pd.DataFrame()
        if frame is not None and not frame.empty:
            frame.to_csv(path, index=False, encoding="utf-8-sig")
        return self._normalize_eastmoney_xjll(frame, period)

    def _normalize_eastmoney_xjll(self, frame: pd.DataFrame, period: str) -> pd.DataFrame:
        required = ["股票代码", "经营性现金流-现金流量净额", "投资性现金流-现金流量净额"]
        if frame is None or frame.empty or any(col not in frame.columns for col in required):
            return pd.DataFrame()
        data = frame.copy()
        data["symbol"] = data["股票代码"].astype(str).str.zfill(6)
        data["ts_code"] = data["symbol"].apply(self._symbol_to_ts_code)
        operating = pd.to_numeric(data["经营性现金流-现金流量净额"], errors="coerce")
        investing = pd.to_numeric(data["投资性现金流-现金流量净额"], errors="coerce")
        data["fcf"] = operating + investing
        data["ann_date"] = data.get("公告日期", "")
        data["cashflow_report_date"] = period
        data = data[~data["fcf"].isna()].copy()
        return data[["ts_code", "cashflow_report_date", "ann_date", "fcf"]]

    @staticmethod
    def _symbol_to_ts_code(symbol: str) -> str:
        code = str(symbol).zfill(6)
        if code.startswith(("6", "9")):
            return f"{code}.SH"
        if code.startswith(("8", "4")):
            return f"{code}.BJ"
        return f"{code}.SZ"

    @staticmethod
    def _inverse_score(value: Any, high: float) -> float:
        if high <= 0:
            return 50.0
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(number) or number <= 0:
            return 0.0
        return max(0.0, min(100.0, (high - number) / high * 100))

    @staticmethod
    def _scale(value: Any, low: float, high: float) -> float:
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
