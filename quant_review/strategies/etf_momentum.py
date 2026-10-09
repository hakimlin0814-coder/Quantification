from __future__ import annotations

"""ETF 实时强度策略。

本模块刻意只使用 ETF 自身可交易数据：价格、成交、趋势、流动性和产品基础信息。
它不读取个股成分或财务数据。实时源没有返回 ETF 时，会明确回退到最近收盘快照，
而不是把历史行情伪装成实时行情。
"""

import json
import math
from datetime import datetime, time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import requests


class EtfMomentumStrategy:
    WINDOWS = (1, 3, 5, 10, 20, 60)

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.config = engine.config

    def run(self) -> dict[str, Any]:
        old_lookback = int(self.config.lookback_days)
        self.config.lookback_days = max(old_lookback, 75)
        try:
            trade_dates = self.engine._trade_dates()
        finally:
            self.config.lookback_days = old_lookback

        self.engine.log("读取ETF基础信息与最近日线，计算全市场统一强度评分...")
        basic = self._fund_basic()
        daily = self._fund_daily_window(trade_dates)
        if basic.empty or daily.empty:
            raise RuntimeError("ETF基础信息或日线为空，请确认 Tushare 的 fund_basic、fund_daily 权限。")

        data = self._prepare(daily, basic)
        now = self._now()
        mode, target_date = self._snapshot_mode(now, trade_dates)
        data, realtime_count, quote_at = self._apply_realtime_snapshot(
            data, basic, target_date, mode
        )
        latest_date = str(data["trade_date"].max())
        features = self._features(data, latest_date)
        rows = self._rows(features)
        query = str(getattr(self.config, "etf_probe_query", "") or "").strip()
        probe = self._probe(rows, query) if query else None
        if mode == "live":
            snapshot_state = "盘中实时行情" if realtime_count else "盘中实时行情缺失，使用最近收盘行情"
        elif mode == "lunch":
            snapshot_state = "午间休盘最新行情" if realtime_count else "午间实时行情缺失，使用最近收盘行情"
        elif mode == "close":
            snapshot_state = "当日15:00最终行情" if latest_date == target_date else "最近交易日收盘行情"
        else:
            snapshot_state = "最近交易日收盘行情"
        market_open = mode == "live" and bool(realtime_count)
        snapshot_at = quote_at or self._close_snapshot_at(latest_date)
        return {
            "ok": True,
            "trade_date": latest_date,
            "snapshot_at": snapshot_at,
            "snapshot_state": snapshot_state,
            "market_open": market_open,
            "realtime_count": realtime_count,
            "realtime_source": "Tushare rt_k / 东方财富兜底" if realtime_count else "",
            "summary": self._summary(rows, snapshot_state),
            "rows": rows,
            "probe": probe,
            "logs": self.engine.logs,
        }

    def _fund_basic(self) -> pd.DataFrame:
        fields = (
            "ts_code,name,management,custodian,fund_type,found_date,due_date,"
            "list_date,issue_date,delist_date,invest_type,type,benchmark,status,market"
        )
        try:
            data = self.engine._cached(
                "fund_basic_etf_active",
                lambda: self.engine.pro.fund_basic(market="E", status="L", fields=fields),
            )
        except Exception:
            data = self.engine._cached(
                "fund_basic_etf_active_fallback",
                lambda: self.engine.pro.fund_basic(market="E"),
            )
        if data.empty or "ts_code" not in data.columns:
            return pd.DataFrame()
        data = data.copy()
        data["ts_code"] = data["ts_code"].astype(str).str.upper()
        data["code"] = data["ts_code"].str.split(".").str[0]
        for col in ["name", "management", "custodian", "fund_type", "benchmark", "list_date", "status", "market", "invest_type", "type"]:
            if col not in data.columns:
                data[col] = ""
            data[col] = data[col].fillna("").astype(str)
        # fund_basic 的 name 是基金登记全称。行情软件常用的是更短的交易简称，
        # 这里优先用公开行情列表的简称；请求失败时仍保留原始全称，不伪造简称。
        quote_names = self._quote_name_map()
        data["display_name"] = data["code"].map(quote_names).fillna(data["name"])
        return data.drop_duplicates("ts_code", keep="first")

    def _quote_name_map(self) -> dict[str, str]:
        params = {
            "pn": 1, "pz": 5000, "po": 1, "np": 1, "fltt": 2, "invt": 2,
            "fid": "f3", "fs": "b:MK0021,b:MK0022,b:MK0023,b:MK0024", "fields": "f12,f14",
        }
        url = "https://push2.eastmoney.com/api/qt/clist/get?" + urlencode(params)
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            items = ((payload.get("data") or {}).get("diff") or [])
            result = {
                str(item.get("f12", "")).zfill(6): str(item.get("f14", "")).strip()
                for item in items
                if str(item.get("f12", "")).strip() and str(item.get("f14", "")).strip()
            }
            if result:
                self.engine.log(f"已读取 {len(result)} 条ETF行情简称，优先用于页面显示。")
            return result
        except Exception as exc:
            self.engine.log(f"ETF行情简称读取失败，回退基金登记全称: {exc}")
            return {}

    def _fund_daily_window(self, trade_dates: list[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for trade_date in trade_dates:
            try:
                frame = self.engine._cached(
                    f"fund_daily_{trade_date}",
                    lambda trade_date=trade_date: self.engine.pro.fund_daily(trade_date=trade_date),
                )
            except Exception as exc:
                self.engine.log(f"ETF日线 {trade_date} 读取失败: {exc}")
                continue
            if frame is not None and not frame.empty and "ts_code" in frame.columns:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        data = pd.concat(frames, ignore_index=True)
        for col in ["open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]:
            if col not in data.columns:
                data[col] = 0
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
        data["ts_code"] = data["ts_code"].astype(str).str.upper()
        data["trade_date"] = data["trade_date"].astype(str)
        return data[data["close"] > 0].sort_values(["ts_code", "trade_date"])

    def _prepare(self, daily: pd.DataFrame, basic: pd.DataFrame) -> pd.DataFrame:
        columns = ["ts_code", "code", "name", "display_name", "management", "custodian", "fund_type", "benchmark", "list_date", "status", "market", "invest_type", "type"]
        data = daily.merge(basic[[col for col in columns if col in basic.columns]], on="ts_code", how="inner")
        data["name"] = data["name"].fillna("")
        data["fund_type"] = data["fund_type"].fillna("").replace("", "未分类")
        data["amount_yi"] = data["amount"] / 100000
        grouped = data.groupby("ts_code", group_keys=False)
        data["history_days"] = grouped["close"].transform("count")
        for window in self.WINDOWS:
            data[f"ret{window}"] = data["close"] / grouped["close"].shift(window) - 1
            data[f"rps{window}"] = data[f"ret{window}"].groupby(data["trade_date"]).rank(pct=True) * 100
        for window in (5, 10, 20):
            data[f"ma{window}"] = grouped["close"].transform(lambda values, w=window: values.rolling(w, min_periods=w).mean())
        data["avg_amount20"] = grouped["amount_yi"].transform(lambda values: values.rolling(20, min_periods=15).mean())
        data["avg_amount5"] = grouped["amount_yi"].transform(lambda values: values.rolling(5, min_periods=3).mean())
        data["amount_ratio"] = data["avg_amount5"] / data["avg_amount20"].replace(0, pd.NA)
        data["high20"] = grouped["high"].transform(lambda values: values.rolling(20, min_periods=15).max())
        data["distance_ma20"] = (data["close"] / data["ma20"].replace(0, pd.NA) - 1) * 100
        return data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    def _apply_realtime_snapshot(
        self,
        data: pd.DataFrame,
        basic: pd.DataFrame,
        trade_date: str,
        mode: str,
    ) -> tuple[pd.DataFrame, int, str]:
        """按市场状态读取行情，且绝不把过期报价标成今日实时行情。"""
        if mode == "history":
            self.engine.log(f"今日休市或尚未开盘，使用最近交易日 {trade_date} 收盘行情。")
            return data, 0, ""

        codes = basic["ts_code"].astype(str).str.upper().tolist()
        rt = self._tushare_rt_k(codes)
        if rt.empty:
            self.engine.log("Tushare rt_k 未返回ETF行情，使用东财实时行情兜底。")
            rt = self._eastmoney_quotes(codes)
        if rt.empty:
            self.engine.log("实时行情源均无返回，使用最近可用收盘行情。")
            return data, 0, ""
        rt, quote_at = self._quotes_for_trade_date(rt, trade_date)
        if rt.empty:
            self.engine.log(f"行情源返回的不是 {trade_date} 数据，拒绝作为实时行情，回退最近收盘。")
            return data, 0, ""
        if mode == "close":
            quote_at = self._close_snapshot_at(trade_date)
        for col in ("close", "open", "high", "low", "pre_close", "pct_chg", "vol", "amount"):
            if col not in rt.columns:
                rt[col] = 0
            rt[col] = pd.to_numeric(rt[col], errors="coerce").fillna(0)
        rt = rt[rt["close"] > 0].copy()
        if rt.empty:
            return data, 0, ""
        rt["open"] = rt["open"].where(rt["open"] > 0, rt["close"])
        rt["high"] = rt["high"].where(rt["high"] > 0, rt[["open", "close"]].max(axis=1))
        rt["low"] = rt["low"].where(rt["low"] > 0, rt[["open", "close"]].min(axis=1))
        rt["pct_chg"] = rt["pct_chg"].where(rt["pct_chg"].ne(0), (rt["close"] / rt["pre_close"].replace(0, pd.NA) - 1) * 100).fillna(0)
        rt["trade_date"] = trade_date
        fields = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]
        daily_fields = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]
        merged = data.loc[data["trade_date"] != trade_date, daily_fields].copy()
        merged = pd.concat([merged, rt[fields]], ignore_index=True)
        # _prepare 会重新计算所有窗口；这里直接重新生成，避免盘中值与历史因子断裂。
        label = "盘中实时" if mode == "live" else ("午间休盘" if mode == "lunch" else "15:00收盘")
        self.engine.log(f"{label}行情已覆盖 {len(rt)} 只ETF，按真实快照重算强度。")
        return self._prepare(merged, basic), len(rt), quote_at

    def _tushare_rt_k(self, codes: list[str]) -> pd.DataFrame:
        """优先使用高级会员的 Tushare Pro rt_k，不使用 SDK 有兼容问题的 realtime_list。"""
        frames: list[pd.DataFrame] = []
        for start in range(0, len(codes), 500):
            chunk = codes[start : start + 500]
            try:
                frame = self.engine.pro.rt_k(ts_code=",".join(chunk))
            except Exception as exc:
                self.engine.log(f"Tushare rt_k 第 {start // 500 + 1} 批读取失败: {exc}")
                continue
            if frame is not None and not frame.empty and "ts_code" in frame.columns:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        data = pd.concat(frames, ignore_index=True).copy()
        data.columns = [str(col).lower() for col in data.columns]
        rename = {"price": "close", "vol": "vol", "volume": "vol"}
        data = data.rename(columns={key: value for key, value in rename.items() if key in data.columns})
        if "close" not in data.columns or "ts_code" not in data.columns:
            return pd.DataFrame()
        data["ts_code"] = data["ts_code"].astype(str).str.upper()
        data["quote_at"] = self._quote_time_series(data)
        if data["quote_at"].eq("").all():
            # rt_k 是即时请求，但部分 Tushare 版本不返回行情时间字段。
            # 只会在已经确认“今天是交易日且处于盘中/收盘后”时调用此接口。
            data["quote_at"] = self._now().strftime("%Y-%m-%d %H:%M:%S")
        if "pct_chg" not in data.columns:
            data["pct_chg"] = 0
        return data.drop_duplicates("ts_code", keep="last")

    def _eastmoney_quotes(self, codes: list[str]) -> pd.DataFrame:
        """只在盘中 Tushare rt_k 无返回时的公开兜底，金额统一为千元。"""
        frames: list[pd.DataFrame] = []
        for start in range(0, len(codes), 80):
            chunk = codes[start : start + 80]
            secids = ",".join(self._eastmoney_secid(code) for code in chunk)
            try:
                response = requests.get(
                    "https://push2.eastmoney.com/api/qt/ulist.np/get",
                    params={"fltt": "2", "invt": "2", "fields": "f12,f2,f17,f15,f16,f18,f5,f6,f124", "secids": secids},
                    timeout=10,
                )
                response.raise_for_status()
                items = (response.json().get("data") or {}).get("diff") or []
            except Exception as exc:
                self.engine.log(f"东财ETF实时行情第 {start // 80 + 1} 批失败: {exc}")
                continue
            rows = []
            for item in items:
                symbol = str(item.get("f12", "")).zfill(6)
                ts_code = self._ts_code_from_symbol(symbol)
                rows.append({
                    "ts_code": ts_code, "close": item.get("f2", 0), "open": item.get("f17", 0),
                    "high": item.get("f15", 0), "low": item.get("f16", 0), "pre_close": item.get("f18", 0),
                    "vol": item.get("f5", 0), "amount": self._num(item.get("f6", 0)) / 1000,
                    "quote_at": datetime.fromtimestamp(
                        self._num(item.get("f124", 0))
                    ).strftime("%Y-%m-%d %H:%M:%S") if self._num(item.get("f124", 0)) > 0 else "",
                })
            if rows:
                frames.append(pd.DataFrame(rows))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def _now() -> datetime:
        return datetime.now()

    @staticmethod
    def _snapshot_mode(now: datetime, trade_dates: list[str]) -> tuple[str, str]:
        today = now.strftime("%Y%m%d")
        available = [item for item in trade_dates if item <= today]
        target = available[-1]
        if target != today:
            return "history", target
        if now.time() < time(9, 15):
            previous = [item for item in trade_dates if item < today]
            return "history", previous[-1]
        if time(11, 30) <= now.time() < time(13, 0):
            return "lunch", today
        if now.time() < time(15, 0):
            return "live", today
        return "close", today

    @staticmethod
    def _close_snapshot_at(trade_date: str) -> str:
        return f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 15:00:00"

    @staticmethod
    def _quote_time_series(data: pd.DataFrame) -> pd.Series:
        for col in ("trade_time", "datetime", "time"):
            if col in data.columns:
                values = data[col].fillna("").astype(str).str.strip()
                if values.ne("").any():
                    return values
        return pd.Series("", index=data.index, dtype=str)

    @staticmethod
    def _quotes_for_trade_date(
        quotes: pd.DataFrame, trade_date: str
    ) -> tuple[pd.DataFrame, str]:
        data = quotes.copy()
        if "quote_at" not in data.columns:
            return pd.DataFrame(), ""
        parsed = pd.to_datetime(data["quote_at"], errors="coerce")
        expected = datetime.strptime(trade_date, "%Y%m%d").date()
        valid = parsed.dt.date == expected
        data = data.loc[valid].copy()
        parsed = parsed.loc[valid]
        if data.empty:
            return data, ""
        quote_at = parsed.max().strftime("%Y-%m-%d %H:%M:%S")
        return data, quote_at

    @staticmethod
    def _eastmoney_secid(ts_code: str) -> str:
        symbol = str(ts_code).split(".")[0].zfill(6)
        market = "1" if str(ts_code).upper().endswith(".SH") or symbol.startswith(("5", "6", "9")) else "0"
        return f"{market}.{symbol}"

    @staticmethod
    def _ts_code_from_symbol(symbol: str) -> str:
        return f"{symbol}.{'SH' if symbol.startswith(('5', '6', '9')) else 'SZ'}"

    def _features(self, data: pd.DataFrame, latest_date: str) -> pd.DataFrame:
        latest = data[data["trade_date"] == latest_date].copy()
        if latest.empty:
            return latest
        for col in [f"rps{window}" for window in self.WINDOWS] + ["amount_ratio", "distance_ma20", "avg_amount20"]:
            latest[col] = pd.to_numeric(latest[col], errors="coerce")
        latest["recent_score"] = (
            latest["rps1"].fillna(0) * .30
            + latest["rps3"].fillna(0) * .30
            + latest["rps5"].fillna(0) * .25
            + latest["rps10"].fillna(0) * .10
            + latest["rps20"].fillna(0) * .05
        )
        latest["intraday_score"] = latest["pct_chg"].rank(pct=True) * 100
        trend_raw = (
            (latest["close"] > latest["ma5"]).astype(float) * 30
            + (latest["ma5"] >= latest["ma10"]).astype(float) * 25
            + (latest["ma10"] >= latest["ma20"]).astype(float) * 25
            + (latest["close"] / latest["high20"].replace(0, pd.NA) * 100).clip(0, 100).fillna(0) * .20
        )
        latest["trend_score"] = trend_raw.clip(0, 100)
        amount_rank = latest["avg_amount20"].rank(pct=True) * 100
        volume_rank = latest["amount_ratio"].clip(0, 2).fillna(0) / 2 * 100
        latest["liquidity_score"] = amount_rank.fillna(0) * .75 + volume_rank * .25
        latest["quality_score"] = 50.0  # 基础费率与跟踪误差数据若未披露，保持中性而不伪造数值。
        latest["overheat_penalty"] = (
            (latest["ret5"].fillna(0) * 100 - float(self.config.etf_overheat_ret5)).clip(lower=0) * .75
            + (latest["distance_ma20"].fillna(0) - float(self.config.etf_overheat_ma20)).clip(lower=0) * .50
        ).clip(0, 25)
        weights = [
            max(float(self.config.etf_weight_recent), 0), max(float(self.config.etf_weight_intraday), 0),
            max(float(self.config.etf_weight_trend), 0), max(float(self.config.etf_weight_liquidity), 0),
            max(float(self.config.etf_weight_quality), 0),
        ]
        total_weight = sum(weights) or 100.0
        latest["opportunity_score"] = (
            latest["recent_score"] * weights[0]
            + latest["intraday_score"] * weights[1]
            + latest["trend_score"] * weights[2]
            + latest["liquidity_score"] * weights[3]
            + latest["quality_score"] * weights[4]
        ) / total_weight - latest["overheat_penalty"]
        latest["opportunity_score"] = latest["opportunity_score"].clip(0, 100)
        latest = latest.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
        latest["rank"] = latest.index + 1
        history_ok = latest["history_days"] >= int(self.config.etf_min_history_days)
        latest["eligibility"] = history_ok
        latest["history_ok"] = history_ok
        return latest

    def _rows(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for _, item in data.iterrows():
            status, reason, risk = self._state(item)
            rows.append({
                "rank": int(item["rank"]),
                "code": str(item.get("code", "")),
                "ts_code": str(item.get("ts_code", "")),
                "name": str(item.get("display_name", item.get("name", ""))),
                "full_name": str(item.get("name", "")),
                "fund_type": str(item.get("fund_type", "未分类")),
                "benchmark": str(item.get("benchmark", "")),
                "management": str(item.get("management", "")),
                "trading_cycle": self._trading_cycle(item)[0],
                "trading_cycle_basis": self._trading_cycle(item)[1],
                "status": status,
                "opportunity_score": self._round(item.get("opportunity_score")),
                "close": self._round(item.get("close")),
                "pct_chg": self._round(item.get("pct_chg")),
                "rps1": self._round(item.get("rps1")), "rps3": self._round(item.get("rps3")),
                "rps5": self._round(item.get("rps5")), "rps10": self._round(item.get("rps10")), "rps20": self._round(item.get("rps20")),
                "ma5": self._round(item.get("ma5")), "ma10": self._round(item.get("ma10")), "ma20": self._round(item.get("ma20")),
                "distance_ma20": self._round(item.get("distance_ma20")),
                "ret1": self._round(item.get("ret1", 0) * 100), "ret3": self._round(item.get("ret3", 0) * 100),
                "ret5": self._round(item.get("ret5", 0) * 100), "ret10": self._round(item.get("ret10", 0) * 100),
                "ret20": self._round(item.get("ret20", 0) * 100),
                "amount_yi": self._round(item.get("amount_yi")), "adv20_yi": self._round(item.get("avg_amount20")),
                "amount_ratio": self._round(item.get("amount_ratio")),
                "recent_score": self._round(item.get("recent_score")), "intraday_score": self._round(item.get("intraday_score")),
                "trend_score": self._round(item.get("trend_score")), "liquidity_score": self._round(item.get("liquidity_score")),
                "quality_score": self._round(item.get("quality_score")), "overheat_penalty": self._round(item.get("overheat_penalty")),
                "history_days": int(item.get("history_days", 0)),
                "eligibility": "合格" if bool(item.get("eligibility")) else "不合格",
                "reason": reason, "risk": risk,
            })
        return rows

    @staticmethod
    def _trading_cycle(item: pd.Series) -> tuple[str, str]:
        """Classify secondary-market same-day trading conservatively from product metadata."""
        fund_type = str(item.get("fund_type", "") or "")
        text = " ".join(str(item.get(col, "") or "") for col in ("name", "display_name", "benchmark", "invest_type", "type", "fund_type"))
        lower = text.lower()
        if "货币" in text:
            return "T+0", "货币ETF（规则匹配）"
        if "黄金" in text:
            return "T+0", "黄金ETF（规则匹配）"
        if "债" in fund_type or "债券" in text:
            return "T+0", "债券ETF（规则匹配）"
        cross_border_markers = ("qdii", "跨境", "境外", "海外", "港股", "恒生", "香港", "纳指", "纳斯达克", "标普", "日经", "德国", "法国", "越南", "印度", "韩国", "美股", "全球")
        if any(marker in lower for marker in cross_border_markers):
            return "T+0", "跨境ETF（规则匹配；以券商交易端为准）"
        if "商品期货" in text or "商品型" in fund_type:
            return "T+0", "商品期货ETF（规则匹配）"
        return "T+1", "未匹配T+0类别，按国内股票ETF处理"

    def _state(self, item: pd.Series) -> tuple[str, str, str]:
        score = self._num(item.get("opportunity_score"))
        rank = int(self._num(item.get("rank")))
        if not bool(item.get("history_ok")):
            return "数据不足", "上市历史不足，无法稳定计算RPS60与20日流动性基线", "历史不足"
        if self._num(item.get("overheat_penalty")) >= 12:
            return "偏热等待", "近期强度高，但涨幅或均线偏离过大", "过热"
        trend_ok = self._num(item.get("close")) > self._num(item.get("ma10")) > 0
        if score >= float(self.config.etf_score_threshold) and rank <= int(self.config.etf_top_n) and trend_ok:
            return "可买", "全市场排名靠前；近期强度、趋势与流动性同时合格", "盘中需手动复核最新价格"
        if score >= float(self.config.etf_score_threshold) * .9:
            return "等待", "综合强度靠前，但尚未达到默认买入门槛", "等待强度或趋势确认"
        return "观察", "当前全市场机会分未进入可买区间", "不追弱势"

    def _probe(self, rows: list[dict[str, Any]], query: str) -> dict[str, Any] | None:
        key = query.upper()
        for row in rows:
            if key in row["code"].upper() or key in row["ts_code"].upper() or key in row["name"].upper() or key in row.get("full_name", "").upper():
                return row
        return {"status": "未找到", "reason": "未在当前ETF基础信息中匹配到该代码或名称", "name": query}

    @staticmethod
    def _summary(rows: list[dict[str, Any]], snapshot_state: str) -> dict[str, Any]:
        return {
            "total": len(rows),
            "buy": sum(row["status"] == "可买" for row in rows),
            "wait": sum(row["status"] == "等待" for row in rows),
            "watch": sum(row["status"] == "观察" for row in rows),
            "snapshot_state": snapshot_state,
        }

    @staticmethod
    def _num(value: Any) -> float:
        try:
            result = float(value)
            return result if math.isfinite(result) else 0.0
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _round(cls, value: Any) -> float | str:
        number = cls._num(value)
        return round(number, 2) if math.isfinite(number) else ""

    @classmethod
    def _pct(cls, value: Any) -> str:
        number = cls._num(value)
        return "" if number <= 0 else f"{number * 100:.2f}%"
