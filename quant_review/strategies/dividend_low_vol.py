from __future__ import annotations

import math
from typing import Any

import pandas as pd

from quant_review.strategies.value_fcf import ValueFcfStrategy


class DividendLowVolStrategy(ValueFcfStrategy):
    """High-dividend and low-volatility defensive strategy for A-share drawdowns."""

    DEFENSIVE_KEYWORDS = (
        "银行",
        "电力",
        "水电",
        "火电",
        "核电",
        "公用",
        "燃气",
        "水务",
        "高速",
        "港口",
        "铁路",
        "机场",
        "煤炭",
        "石油",
        "石化",
        "通信",
        "运营",
        "保险",
    )
    CYCLICAL_KEYWORDS = ("地产", "房地产", "钢铁", "有色", "化工", "建材")

    def run(self) -> dict[str, Any]:
        old_lookback = int(self.config.lookback_days)
        self.config.lookback_days = max(old_lookback, 90)
        try:
            self.log("读取交易日历，红利低波策略需要约90个交易日计算波动和Beta...")
            trade_dates = self.engine._trade_dates()
        finally:
            self.config.lookback_days = old_lookback

        trade_date = trade_dates[-1]
        self.config.end_date = trade_date
        self.log(f"使用交易日 {trade_date}")

        self.log("读取股票基础信息...")
        basic = self.engine._stock_basic()

        self.log("读取近期日线行情，计算年化波动率、Beta、最大回撤和均线位置...")
        daily = self.engine._daily_window(trade_dates)
        if self.config.review_mode == "intraday":
            self.log("盘中模式：读取实时行情并合成临时日K...")
            daily = self.engine._apply_intraday_snapshot(daily, basic, trade_date)

        self.log("读取红利和估值指标：股息率、PB、PE_TTM、市值、成交额...")
        daily_basic = self._daily_basic_with_dividend(trade_date)
        if self.config.review_mode == "intraday" and self._daily_basic_dividend_is_empty(daily_basic) and len(trade_dates) >= 2:
            prev_date = trade_dates[-2]
            self.log(f"盘中模式：当日红利估值指标未更新，回退到 {prev_date}。")
            daily_basic = self._daily_basic_with_dividend(prev_date)

        self.log("读取ROE和现金流，作为高股息可持续性的质量过滤...")
        fundamentals = self._financial_metrics(trade_date)

        technical = self._technical_metrics(daily)
        self.log("合成红利低波防守候选池...")
        rows = self._rows(basic, technical, daily_basic, fundamentals)
        probe = self._probe(getattr(self, "_last_dividend_data", pd.DataFrame()))

        self.log(f"红利低波防守策略完成，共 {len(rows)} 只。")
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

    def _daily_basic_with_dividend(self, trade_date: str) -> pd.DataFrame:
        fields = (
            "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,total_mv,circ_mv,"
            "pe,pe_ttm,pb,dv_ratio,dv_ttm"
        )
        try:
            df = self.engine._cached(
                f"daily_basic_dividend_{trade_date}",
                lambda: self.engine.pro.daily_basic(trade_date=trade_date, fields=fields),
            )
        except Exception as exc:
            self.log(f"红利估值指标读取失败，回退共享daily_basic: {exc}")
            df = self.engine._daily_basic(trade_date)
        if "ts_code" in df.columns:
            df["ts_code"] = df["ts_code"].astype(str)
        for col in [
            "turnover_rate",
            "turnover_rate_f",
            "volume_ratio",
            "total_mv",
            "circ_mv",
            "pe",
            "pe_ttm",
            "pb",
            "dv_ratio",
            "dv_ttm",
        ]:
            if col not in df.columns:
                df[col] = pd.NA
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def _daily_basic_dividend_is_empty(daily_basic: pd.DataFrame) -> bool:
        if daily_basic.empty or "total_mv" not in daily_basic.columns:
            return True
        total_mv = pd.to_numeric(daily_basic["total_mv"], errors="coerce").fillna(0)
        return bool(total_mv.sum() <= 0)

    def _technical_metrics(self, daily: pd.DataFrame) -> pd.DataFrame:
        data = daily.copy()
        data["trade_date"] = data["trade_date"].astype(str)
        data["ts_code"] = data["ts_code"].astype(str)
        for col in ["close", "pct_chg", "amount"]:
            if col not in data.columns:
                data[col] = 0
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
        data = data.sort_values(["ts_code", "trade_date"])
        market_ret = data.groupby("trade_date")["pct_chg"].mean().rename("market_pct_chg").reset_index()
        data = data.merge(market_ret, on="trade_date", how="left")

        rows = []
        for ts_code, frame in data.groupby("ts_code", sort=False):
            frame = frame.tail(90).copy()
            if frame.empty:
                continue
            last = frame.iloc[-1]
            close = float(last.get("close", 0) or 0)
            returns = pd.to_numeric(frame["pct_chg"], errors="coerce").fillna(0) / 100
            market = pd.to_numeric(frame["market_pct_chg"], errors="coerce").fillna(0) / 100
            volatility = float(returns.tail(60).std(ddof=0) * math.sqrt(244) * 100) if len(returns) >= 20 else pd.NA
            beta = self._beta(returns.tail(60), market.tail(60))
            close_series = pd.to_numeric(frame["close"], errors="coerce").fillna(0)
            recent_close = close_series.tail(60)
            running_high = recent_close.cummax().replace(0, pd.NA)
            max_drawdown = float(((recent_close / running_high - 1) * 100).min() * -1) if len(recent_close) >= 10 else pd.NA
            ma20 = float(close_series.tail(20).mean()) if len(close_series) >= 20 else pd.NA
            ma60 = float(close_series.tail(60).mean()) if len(close_series) >= 60 else pd.NA
            ma60_prev = float(close_series.iloc[-80:-20].mean()) if len(close_series) >= 80 else pd.NA
            ret20 = self._period_return(close_series, 20)
            ret60 = self._period_return(close_series, 60)
            market_ret20 = self._compound_return(market, 20)
            market_ret60 = self._compound_return(market, 60)
            rs20 = ret20 - market_ret20 if not pd.isna(ret20) and not pd.isna(market_ret20) else pd.NA
            rs60 = ret60 - market_ret60 if not pd.isna(ret60) and not pd.isna(market_ret60) else pd.NA
            distance_ma60 = (close / ma60 - 1) * 100 if not pd.isna(ma60) and ma60 > 0 else pd.NA
            ma20_ma60_gap = (ma20 / ma60 - 1) * 100 if not pd.isna(ma20) and not pd.isna(ma60) and ma60 > 0 else pd.NA
            ma60_slope = (ma60 / ma60_prev - 1) * 100 if not pd.isna(ma60) and not pd.isna(ma60_prev) and ma60_prev > 0 else pd.NA
            rows.append(
                {
                    "ts_code": ts_code,
                    "trade_date": str(last.get("trade_date", "")),
                    "close": close,
                    "pct_chg": float(last.get("pct_chg", 0) or 0),
                    "amount_yi": float(last.get("amount", 0) or 0) / 100000,
                    "annual_volatility": volatility,
                    "beta": beta,
                    "max_drawdown": max_drawdown,
                    "ma20": ma20,
                    "ma60": ma60,
                    "ret20": ret20,
                    "ret60": ret60,
                    "market_ret20": market_ret20,
                    "market_ret60": market_ret60,
                    "rs20": rs20,
                    "rs60": rs60,
                    "distance_ma60": distance_ma60,
                    "ma20_ma60_gap": ma20_ma60_gap,
                    "ma60_slope": ma60_slope,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _beta(returns: pd.Series, market: pd.Series) -> float | Any:
        data = pd.concat([returns.reset_index(drop=True), market.reset_index(drop=True)], axis=1).dropna()
        if len(data) < 20:
            return pd.NA
        market_var = float(data.iloc[:, 1].var(ddof=0))
        if market_var <= 0:
            return pd.NA
        return float(data.iloc[:, 0].cov(data.iloc[:, 1], ddof=0) / market_var)

    @staticmethod
    def _period_return(close: pd.Series, days: int) -> float | Any:
        if len(close) <= days:
            return pd.NA
        current = float(close.iloc[-1] or 0)
        base = float(close.iloc[-days - 1] or 0)
        if base <= 0:
            return pd.NA
        return (current / base - 1) * 100

    @staticmethod
    def _compound_return(returns: pd.Series, days: int) -> float | Any:
        if len(returns) < days:
            return pd.NA
        data = pd.to_numeric(returns.tail(days), errors="coerce").fillna(0)
        return float(((1 + data).prod() - 1) * 100)

    def _rows(
        self,
        basic: pd.DataFrame,
        technical: pd.DataFrame,
        daily_basic: pd.DataFrame,
        fundamentals: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        basic_cols = ["ts_code", "symbol", "name", "industry", "market", "stock_type"]
        data = basic[[col for col in basic_cols if col in basic.columns]].copy()
        data = data.merge(technical, on="ts_code", how="left")

        db = daily_basic.copy()
        if "ts_code" in db.columns:
            db["ts_code"] = db["ts_code"].astype(str)
        else:
            db["ts_code"] = pd.Series(dtype=str)
        for col in ["pe", "pe_ttm", "pb", "dv_ratio", "dv_ttm", "total_mv", "circ_mv", "turnover_rate", "turnover_rate_f", "volume_ratio"]:
            if col not in db.columns:
                db[col] = pd.NA
            db[col] = pd.to_numeric(db[col], errors="coerce")
        data = data.merge(
            db[["ts_code", "pe", "pe_ttm", "pb", "dv_ratio", "dv_ttm", "total_mv", "circ_mv", "turnover_rate", "turnover_rate_f", "volume_ratio"]],
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
        data["market_cap_yi"] = pd.to_numeric(data["total_mv"], errors="coerce") / 10000
        data["pe_value"] = pd.to_numeric(data["pe_ttm"], errors="coerce")
        pe = pd.to_numeric(data["pe"], errors="coerce")
        data.loc[(data["pe_value"].isna() | (data["pe_value"] <= 0)) & (pe > 0), "pe_value"] = pe
        for col in [
            "pb",
            "dv_ratio",
            "dv_ttm",
            "roe_value",
            "fcf_ttm",
            "annual_volatility",
            "beta",
            "max_drawdown",
            "distance_ma60",
            "ret20",
            "ret60",
            "rs20",
            "rs60",
            "ma20_ma60_gap",
            "ma60_slope",
            "amount_yi",
        ]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data["dividend_yield"] = data["dv_ttm"].combine_first(data["dv_ratio"])
        data["roe_quality"] = data.apply(self._annualized_roe, axis=1)
        data["fcf_yield"] = data["fcf_ttm"] / (pd.to_numeric(data["total_mv"], errors="coerce") * 10000) * 100
        data["defense_bucket"] = data.apply(self._defense_bucket, axis=1)
        data["defense_score"] = data["defense_bucket"].map({"核心防守": 100, "周期红利": 70, "普通红利": 45}).fillna(45)
        data["trend_score"] = data.apply(self._trend_score, axis=1)
        data["trend_state"] = data.apply(self._trend_state, axis=1)
        data["dividend_score"] = data.apply(self._dividend_low_vol_score, axis=1)
        self._last_dividend_data = data.copy()

        data = self._apply_dividend_filters(data)
        if data.empty:
            return []
        data = data.sort_values(
            ["dividend_score", "dividend_yield", "annual_volatility", "beta"],
            ascending=[False, False, True, True],
        )
        return [self._to_dividend_row(row) for _, row in data.head(int(self.config.dividend_top_count)).iterrows()]

    def _probe(self, data: pd.DataFrame) -> dict[str, Any]:
        query = str(getattr(self.config, "dividend_probe_query", "") or "").strip()
        if not query:
            return {}
        if data.empty:
            return {"query": query, "found": False, "message": "没有可分析的数据"}
        match = self._match_stock(data, query)
        if match is None:
            return {"query": query, "found": False, "message": f"未找到 {query}"}
        checks = self._dividend_checks(match)
        enabled = [item for item in checks if not item.get("disabled")]
        overall_pass = bool(enabled) and all(bool(item.get("pass")) for item in enabled)
        row = self._to_dividend_row(match)
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

    def _apply_dividend_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        for check in self._filter_columns():
            result = result[~result[check].isna()]
        result = result[result["market_cap_yi"].fillna(0) >= float(self.config.dividend_min_market_cap_yi)]
        result = result[result["amount_yi"].fillna(0) >= float(self.config.dividend_min_amount_yi)]

        min_yield = float(self.config.dividend_min_yield)
        max_yield = float(self.config.dividend_max_yield)
        if min_yield > 0:
            result = result[result["dividend_yield"] >= min_yield]
        if max_yield > 0:
            result = result[result["dividend_yield"] <= max_yield]

        max_vol = float(self.config.dividend_max_volatility)
        max_beta = float(self.config.dividend_max_beta)
        max_dd = float(self.config.dividend_max_drawdown)
        max_dist = float(self.config.dividend_max_distance_ma60)
        min_rs20 = float(self.config.dividend_min_rs20)
        min_ma60_slope = float(self.config.dividend_min_ma60_slope)
        max_ret20 = float(self.config.dividend_max_ret20)
        min_roe = float(self.config.dividend_min_roe)
        max_pb = float(self.config.dividend_max_pb)
        max_pe = float(self.config.dividend_max_pe_ttm)
        if max_vol > 0:
            result = result[result["annual_volatility"] <= max_vol]
        if max_beta > 0:
            result = result[result["beta"] <= max_beta]
        if max_dd > 0:
            result = result[result["max_drawdown"] <= max_dd]
        if max_dist > 0:
            result = result[result["distance_ma60"].abs() <= max_dist]
        result = result[result["rs20"] >= min_rs20]
        result = result[result["ma60_slope"] >= min_ma60_slope]
        if max_ret20 > 0:
            result = result[result["ret20"] <= max_ret20]
        if min_roe > 0:
            result = result[result["roe_quality"] >= min_roe]
        if max_pb > 0:
            result = result[(result["pb"] > 0) & (result["pb"] <= max_pb)]
        if max_pe > 0:
            result = result[(result["pe_value"] > 0) & (result["pe_value"] <= max_pe)]

        if self.config.dividend_exclude_st:
            result = result[~result["name"].astype(str).str.upper().str.contains("ST", regex=False)]
        if self.config.dividend_exclude_star:
            result = result[~result["symbol"].astype(str).str.startswith(("688", "689"))]
        if self.config.dividend_exclude_chinext:
            result = result[~result["symbol"].astype(str).str.startswith(("300", "301"))]
        result = result[~result["symbol"].astype(str).str.startswith(("8", "4", "92"))]
        return result

    @staticmethod
    def _filter_columns() -> list[str]:
        return ["dividend_yield", "annual_volatility", "beta", "max_drawdown", "distance_ma60", "ret20", "rs20", "ma60_slope"]

    def _dividend_checks(self, row: pd.Series) -> list[dict[str, Any]]:
        checks = []
        checks.append(self._check("总市值", row.get("market_cap_yi"), ">=", self.config.dividend_min_market_cap_yi, self._num(row.get("market_cap_yi"), 0) >= float(self.config.dividend_min_market_cap_yi), "亿"))
        checks.append(self._check("成交额", row.get("amount_yi"), ">=", self.config.dividend_min_amount_yi, self._num(row.get("amount_yi"), 0) >= float(self.config.dividend_min_amount_yi), "亿"))
        checks.append(self._check("股息率", row.get("dividend_yield"), ">=", self.config.dividend_min_yield, self._num(row.get("dividend_yield"), 0) >= float(self.config.dividend_min_yield), "%"))
        if float(self.config.dividend_max_yield) > 0:
            checks.append(self._check("股息率陷阱上限", row.get("dividend_yield"), "<=", self.config.dividend_max_yield, self._num(row.get("dividend_yield"), 999) <= float(self.config.dividend_max_yield), "%"))
        checks.append(self._check("年化波动", row.get("annual_volatility"), "<=", self.config.dividend_max_volatility, self._num(row.get("annual_volatility"), 999) <= float(self.config.dividend_max_volatility), "%"))
        checks.append(self._check("Beta", row.get("beta"), "<=", self.config.dividend_max_beta, self._num(row.get("beta"), 999) <= float(self.config.dividend_max_beta)))
        checks.append(self._check("最大回撤", row.get("max_drawdown"), "<=", self.config.dividend_max_drawdown, self._num(row.get("max_drawdown"), 999) <= float(self.config.dividend_max_drawdown), "%"))
        checks.append(self._check("距MA60", row.get("distance_ma60"), "abs <=", self.config.dividend_max_distance_ma60, abs(self._num(row.get("distance_ma60"), 999)) <= float(self.config.dividend_max_distance_ma60), "%"))
        checks.append(self._check("20日相对强弱", row.get("rs20"), ">=", self.config.dividend_min_rs20, self._num(row.get("rs20"), -999) >= float(self.config.dividend_min_rs20), "%"))
        checks.append(self._check("MA60斜率", row.get("ma60_slope"), ">=", self.config.dividend_min_ma60_slope, self._num(row.get("ma60_slope"), -999) >= float(self.config.dividend_min_ma60_slope), "%"))
        if float(self.config.dividend_max_ret20) > 0:
            checks.append(self._check("20日涨幅过热", row.get("ret20"), "<=", self.config.dividend_max_ret20, self._num(row.get("ret20"), 999) <= float(self.config.dividend_max_ret20), "%"))
        if float(self.config.dividend_min_roe) > 0:
            checks.append(self._check("近似年化ROE", row.get("roe_quality"), ">=", self.config.dividend_min_roe, self._num(row.get("roe_quality"), -999) >= float(self.config.dividend_min_roe), "%"))
        if float(self.config.dividend_max_pb) > 0:
            checks.append(self._check("PB", row.get("pb"), "<=", self.config.dividend_max_pb, 0 < self._num(row.get("pb"), 999) <= float(self.config.dividend_max_pb)))
        if float(self.config.dividend_max_pe_ttm) > 0:
            checks.append(self._check("PE_TTM", row.get("pe_value"), "<=", self.config.dividend_max_pe_ttm, 0 < self._num(row.get("pe_value"), 999) <= float(self.config.dividend_max_pe_ttm)))

        name = str(row.get("name", ""))
        symbol = str(row.get("symbol", ""))
        if self.config.dividend_exclude_st:
            checks.append(self._check("排除ST", name, "not contains", "ST", "ST" not in name.upper()))
        if self.config.dividend_exclude_star:
            checks.append(self._check("排除科创板", symbol, "not starts", "688/689", not symbol.startswith(("688", "689"))))
        if self.config.dividend_exclude_chinext:
            checks.append(self._check("排除创业板", symbol, "not starts", "300/301", not symbol.startswith(("300", "301"))))
        checks.append(self._check("排除北交所", symbol, "not starts", "8/4/92", not symbol.startswith(("8", "4", "92"))))
        return checks

    def _dividend_low_vol_score(self, row: pd.Series) -> float:
        min_yield = float(self.config.dividend_min_yield)
        max_yield = float(self.config.dividend_max_yield)
        yield_high = max(min_yield + 4, 8)
        dividend_score = self._scale(row.get("dividend_yield"), min_yield, yield_high)
        if max_yield > 0 and self._num(row.get("dividend_yield"), 0) > max_yield:
            dividend_score *= 0.55

        vol_score = 100 - self._scale(row.get("annual_volatility"), 10, max(float(self.config.dividend_max_volatility), 20))
        beta_score = 100 - self._scale(row.get("beta"), 0.25, max(float(self.config.dividend_max_beta), 0.8))
        drawdown_score = 100 - self._scale(row.get("max_drawdown"), 8, max(float(self.config.dividend_max_drawdown), 18))
        low_vol_score = max(0, min(100, vol_score * 0.55 + beta_score * 0.25 + drawdown_score * 0.20))

        min_roe = float(self.config.dividend_min_roe)
        roe_score = self._scale(row.get("roe_quality"), min_roe, max(min_roe + 10, 16)) if min_roe > 0 else 55
        fcf_score = 50 if self._cashflow_not_comparable(row) else self._scale(row.get("fcf_yield"), 0, 6)
        quality_score = roe_score * 0.70 + fcf_score * 0.30

        pb_score = self._inverse_score(row.get("pb"), max(float(self.config.dividend_max_pb), 2))
        pe_score = self._inverse_score(row.get("pe_value"), max(float(self.config.dividend_max_pe_ttm), 20))
        valuation_score = pb_score * 0.45 + pe_score * 0.55
        trend_score = self._num(row.get("trend_score"), 40)
        defense_score = self._num(row.get("defense_score"), 45)
        score = (
            dividend_score * 0.28
            + low_vol_score * 0.23
            + quality_score * 0.18
            + valuation_score * 0.09
            + defense_score * 0.07
            + trend_score * 0.15
        )
        return round(max(0, min(100, score)), 2)

    def _trend_score(self, row: pd.Series) -> float:
        dist = self._num(row.get("distance_ma60"), None)
        ret20 = self._num(row.get("ret20"), 0)
        rs20 = self._num(row.get("rs20"), 0)
        ma60_slope = self._num(row.get("ma60_slope"), 0)
        ma_gap = self._num(row.get("ma20_ma60_gap"), 0)
        if dist is None:
            return 40

        if -5 <= dist <= 8:
            base = 82
        elif -12 <= dist <= 12:
            base = 66
        elif -18 <= dist <= 18:
            base = 48
        else:
            base = 25

        if rs20 >= 5:
            base += 14
        elif rs20 >= 0:
            base += 8
        elif rs20 < -10:
            base -= 24
        elif rs20 < -5:
            base -= 12

        if ma60_slope >= 2:
            base += 10
        elif ma60_slope >= 0:
            base += 5
        elif ma60_slope < -8:
            base -= 22
        elif ma60_slope < -3:
            base -= 10

        if 0 <= ret20 <= 10:
            base += 8
        elif -5 <= ret20 < 0:
            base += 4
        elif ret20 > float(self.config.dividend_max_ret20) > 0:
            base -= 28
        elif ret20 > 12:
            base -= 12
        elif ret20 < -12:
            base -= 20
        elif ret20 < -6:
            base -= 10

        if ma_gap >= 0:
            base += 5
        elif ma_gap < -6:
            base -= 8
        return max(0, min(100, base))

    def _trend_state(self, row: pd.Series) -> str:
        ret20 = self._num(row.get("ret20"), 0)
        rs20 = self._num(row.get("rs20"), 0)
        dist = self._num(row.get("distance_ma60"), 0)
        slope = self._num(row.get("ma60_slope"), 0)
        max_ret20 = float(self.config.dividend_max_ret20)
        if max_ret20 > 0 and ret20 > max_ret20:
            return "短期过热"
        if dist < -12 or slope < -8:
            return "破位走弱"
        if rs20 >= 5 and slope >= 0 and -5 <= dist <= 12:
            return "温和强势"
        if rs20 >= 0 and dist >= -8:
            return "抗跌修复"
        return "中性防守"

    def _defense_bucket(self, row: pd.Series) -> str:
        text = f"{row.get('industry', '')}{row.get('name', '')}"
        if any(keyword in text for keyword in self.DEFENSIVE_KEYWORDS):
            return "核心防守"
        if any(keyword in text for keyword in self.CYCLICAL_KEYWORDS):
            return "周期红利"
        return "普通红利"

    @staticmethod
    def _cashflow_not_comparable(row: pd.Series) -> bool:
        text = f"{row.get('industry', '')}{row.get('name', '')}"
        return any(keyword in text for keyword in ("银行", "保险", "证券", "信托", "多元金融"))

    def _annualized_roe(self, row: pd.Series) -> float | Any:
        roe = self._num(row.get("roe_value"), None)
        if roe is None:
            return pd.NA
        report_date = str(row.get("report_date", "") or "")
        suffix = report_date[4:] if len(report_date) == 8 else ""
        multiplier = {"0331": 4.0, "0630": 2.0, "0930": 4.0 / 3.0, "1231": 1.0}.get(suffix, 1.0)
        return roe * multiplier

    def _to_dividend_row(self, row: pd.Series) -> dict[str, Any]:
        return {
            "ts_code": row.get("ts_code", ""),
            "code": row.get("symbol", ""),
            "name": row.get("name", ""),
            "stock_type": row.get("stock_type", ""),
            "industry": row.get("industry", ""),
            "defense_bucket": row.get("defense_bucket", ""),
            "close": self._round_or_blank(row.get("close")),
            "pct_chg": self._round_or_blank(row.get("pct_chg")),
            "amount_yi": self._round_or_blank(row.get("amount_yi")),
            "market_cap_yi": self._round_or_blank(row.get("market_cap_yi")),
            "dividend_yield": self._round_or_blank(row.get("dividend_yield")),
            "annual_volatility": self._round_or_blank(row.get("annual_volatility")),
            "beta": self._round_or_blank(row.get("beta")),
            "max_drawdown": self._round_or_blank(row.get("max_drawdown")),
            "distance_ma60": self._round_or_blank(row.get("distance_ma60")),
            "ret20": self._round_or_blank(row.get("ret20")),
            "rs20": self._round_or_blank(row.get("rs20")),
            "rs60": self._round_or_blank(row.get("rs60")),
            "ma60_slope": self._round_or_blank(row.get("ma60_slope")),
            "trend_state": row.get("trend_state", ""),
            "trend_score": self._round_or_blank(row.get("trend_score")),
            "pb": self._round_or_blank(row.get("pb")),
            "pe_ttm": self._round_or_blank(row.get("pe_value")),
            "roe": self._round_or_blank(row.get("roe_quality")),
            "fcf_yield": "" if self._cashflow_not_comparable(row) else self._round_or_blank(row.get("fcf_yield")),
            "dividend_score": self._round_or_blank(row.get("dividend_score")),
            "report_date": "" if pd.isna(row.get("report_date", pd.NA)) else str(row.get("report_date")),
            "cashflow_report_date": "" if pd.isna(row.get("cashflow_report_date", pd.NA)) else str(row.get("cashflow_report_date")),
            "reason": self._dividend_reason(row),
        }

    def _dividend_reason(self, row: pd.Series) -> str:
        parts = []
        dy = self._num(row.get("dividend_yield"), None)
        vol = self._num(row.get("annual_volatility"), None)
        beta = self._num(row.get("beta"), None)
        dd = self._num(row.get("max_drawdown"), None)
        roe = self._num(row.get("roe_quality"), None)
        dist = self._num(row.get("distance_ma60"), None)
        rs20 = self._num(row.get("rs20"), None)
        slope = self._num(row.get("ma60_slope"), None)
        if dy is not None:
            parts.append(f"股息率{dy:.1f}%")
        if vol is not None:
            parts.append(f"年波动{vol:.1f}%")
        if beta is not None:
            parts.append(f"Beta {beta:.2f}")
        if dd is not None:
            parts.append(f"近60日回撤{dd:.1f}%")
        if roe is not None:
            parts.append(f"年化ROE {roe:.1f}%")
        if dist is not None:
            parts.append(f"距MA60 {dist:.1f}%")
        if rs20 is not None:
            parts.append(f"20日相对强弱{rs20:.1f}%")
        if slope is not None:
            parts.append(f"MA60斜率{slope:.1f}%")
        bucket = str(row.get("defense_bucket", ""))
        if bucket:
            parts.append(bucket)
        return "；".join(parts)
