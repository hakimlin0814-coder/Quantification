from __future__ import annotations

import math
from datetime import datetime, time
from typing import Any

import pandas as pd

from quant_review.strategies.concept_radar import ConceptRadarStrategy


class CrossSectionalMomentumStrategy:
    """横截面动量策略。

    先在股票范围内计算个股动量排名，再叠加可选的涨停/回撤/开盘条件，
    可选地限制股票必须属于概念动量 Top N，最后给通过硬条件的股票评分推荐。

    该策略使用当前项目的 Tushare 数据层和缓存，不改变其他策略的配置语义。
    实时行情只有在配置明确开启且用户点击运行时才会被请求。
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.config = engine.config
        self._limit_cache: dict[str, set[str] | None] = {}
        self._limit_status: dict[str, str] = {}
        self._daily_basic_metric_cache: dict[tuple[str, str], pd.Series] = {}

    def log(self, message: str) -> None:
        self.engine.log(message)

    def run(self) -> dict[str, Any]:
        self._condition_diagnostics: list[dict[str, Any]] = []
        self._sync_dynamic_config()
        requested_date = str(getattr(self.config, "end_date", "") or "").strip()
        old_lookback = int(getattr(self.config, "lookback_days", 70) or 70)
        offsets = [
            int(getattr(self.config, "cross_momentum_limit_up_offset", -3)),
            int(getattr(self.config, "cross_momentum_limit_not_up_offset", -2)),
            int(getattr(self.config, "cross_momentum_low_offset", -1)),
            int(getattr(self.config, "cross_momentum_open_offset", 0)),
        ]
        for condition in self._conditions():
            offsets.extend(self._condition_offsets(condition))
        stock_days = max(1, int(getattr(self.config, "cross_momentum_days", 10)))
        concept_days = max(1, int(getattr(self.config, "cross_momentum_concept_days", 8)))
        for condition in self._conditions():
            days = self._condition_days(condition)
            if self._condition_object(condition) == "stock":
                stock_days = max(stock_days, days)
            if self._condition_object(condition) == "concept":
                concept_days = max(concept_days, days)
        required_history = max(
            30,
            stock_days + 5,
            concept_days + 5,
            max(abs(offset) for offset in offsets) + 5,
        )
        self.config.lookback_days = max(old_lookback, required_history)

        try:
            self.log("读取交易日历，计算 T 和相对交易日...")
            trade_dates = self.engine._trade_dates()
            trade_date = trade_dates[-1]
            self.config.end_date = trade_date
            date_note = self._date_note(requested_date, trade_date)

            self.log("读取股票基础信息并应用股票范围过滤...")
            basic = self.engine._stock_basic()

            self.log("读取个股日线行情...")
            daily = self.engine._daily_window(trade_dates)

            realtime_enabled = bool(getattr(self.config, "cross_momentum_realtime_enabled", False))
            snapshot_mode = self._snapshot_mode(trade_date)
            realtime_applied = False
            if realtime_enabled and snapshot_mode in {"live", "lunch"}:
                self.log("已启用实时数据，手动运行时读取当前时刻行情...")
                daily = self.engine._apply_intraday_snapshot(daily, basic, trade_date, snapshot_mode)
                realtime_applied = bool(self.engine._last_realtime_snapshot.get("count", 0))
            elif realtime_enabled:
                self.log("当前不在交易时段或 T 不是当天，使用已完成交易日数据。")
            else:
                self.engine._last_realtime_snapshot = {
                    "count": 0,
                    "quote_at": self._close_snapshot_at(trade_date),
                    "source": "日线收盘",
                }

            daily_basic = self.engine._daily_basic(trade_date)
            t1_date = trade_dates[-2] if len(trade_dates) >= 2 else ""
            t1_daily_basic = self.engine._daily_basic(t1_date) if t1_date else pd.DataFrame()
            if daily_basic.empty and not t1_daily_basic.empty:
                daily_basic = t1_daily_basic

            features = self._stock_features(daily, basic, daily_basic, trade_dates, t1_daily_basic)
            features = features[self._market_mask(features)].copy()
            if features.empty:
                return self._payload(trade_date, requested_date, date_note, snapshot_mode, [], [], [], realtime_applied, 0)

            self.log("计算个股横截面动量排名...")
            features = self._attach_stock_rank(features, stock_days)

            concepts: list[dict[str, Any]] = []
            if bool(getattr(self.config, "cross_momentum_concept_enabled", False)):
                self.log("读取概念行情和概念成员，计算概念 Top N...")
                features, concepts = self._attach_concept_top(features, trade_dates, concept_days)
            else:
                features["concept_name"] = ""
                features["concept_code"] = ""
                features["concept_rank"] = pd.NA
                features["concept_return"] = pd.NA
                features["concept_count"] = 0
                features["concept_pass"] = True

            self.log("执行动态条件构建器中的筛选条件...")
            features, hard_mask = self._apply_conditions(features, daily, trade_dates)
            result = features[hard_mask].copy()
            result = self._add_recommendation(result, stock_days, concept_days, realtime_applied)
            result = result.sort_values(
                ["recommended", "recommend_score", "stock_rank", "ts_code"],
                ascending=[False, False, True, True],
                na_position="last",
            ).reset_index(drop=True)

            recommend_count = max(1, int(getattr(self.config, "cross_momentum_recommend_count", 10)))
            result["recommend_rank"] = ""
            if not result.empty:
                take = min(recommend_count, len(result))
                result.loc[: take - 1, "recommend_rank"] = range(1, take + 1)
                result["recommended"] = result.index < take

            rows = self._rows(result, trade_date, realtime_applied)
            recommended = [row for row in rows if row.get("recommended")]
            self.log(f"横截面动量筛选完成：有效股票 {len(features)} 只，通过条件 {len(rows)} 只，推荐 {len(recommended)} 只。")
            return self._payload(
                trade_date,
                requested_date,
                date_note,
                snapshot_mode,
                rows,
                recommended,
                concepts,
                realtime_applied,
                len(features),
            )
        finally:
            self.config.lookback_days = old_lookback

    def run_backtest(self) -> dict[str, Any]:
        """按当前条件回放最近 N 个已完成交易日。

        每个历史 T 都独立运行一次当前筛选规则，保证动态条件、市场范围和
        排除项与页面完全一致。回测只读取正式日线，不使用实时快照。
        """
        requested_days = max(1, min(250, self._as_int(getattr(self.config, "cross_momentum_backtest_days", 60), 60)))
        old_end_date = str(getattr(self.config, "end_date", ""))
        old_lookback = int(getattr(self.config, "lookback_days", 70) or 70)
        old_realtime = bool(getattr(self.config, "cross_momentum_realtime_enabled", False))
        old_auto_refresh = bool(getattr(self.config, "cross_momentum_realtime_auto_refresh", False))
        old_logs = list(getattr(self.engine, "logs", []))
        try:
            # 先扩大日历窗口；逐日运行时数据层会复用同一份本地缓存。
            self.config.lookback_days = max(old_lookback, requested_days + 90)
            self.config.cross_momentum_realtime_enabled = False
            self.config.cross_momentum_realtime_auto_refresh = False
            all_dates = self.engine._trade_dates()
            completed_dates = [date for date in all_dates if self._is_completed_trade_date(date)]
            targets = completed_dates[-requested_days:]
            if not targets:
                raise RuntimeError("没有可用于回测的已完成交易日。")

            self.log(f"开始横截面动量历史回测：{len(targets)} 个交易日，使用当前筛选条件。")
            event_rows: list[dict[str, Any]] = []
            for trade_date in targets:
                self.config.end_date = trade_date
                screen = self.run()
                event_rows.extend(self._backtest_rows_for_date(screen.get("rows", []), trade_date))

            unique_codes = {str(row.get("ts_code", "")) for row in event_rows if row.get("ts_code")}
            return {
                "ok": True,
                "strategy": "cross_sectional_momentum_backtest",
                "strategy_name": "横截面动量策略历史回测",
                "requested_days": requested_days,
                "actual_days": len(targets),
                "start_date": targets[0],
                "end_date": targets[-1],
                "formula": self._selection_formula(),
                "event_count": len(event_rows),
                "stock_count": len(unique_codes),
                "rows": event_rows,
                "note": "历史回测仅使用已完成日线和当日已披露的 daily_basic；不使用盘中实时快照。",
            }
        finally:
            self.config.end_date = old_end_date
            self.config.lookback_days = old_lookback
            self.config.cross_momentum_realtime_enabled = old_realtime
            self.config.cross_momentum_realtime_auto_refresh = old_auto_refresh
            self.engine.logs = old_logs

    @staticmethod
    def _is_completed_trade_date(trade_date: str, now: datetime | None = None) -> bool:
        now = now or datetime.now()
        return trade_date != now.strftime("%Y%m%d") or now.time() >= time(15, 0)

    def _backtest_rows_for_date(self, screen_rows: list[dict[str, Any]], trade_date: str) -> list[dict[str, Any]]:
        """补齐某个历史 T 的 T-5 至 T-1 日线和 daily_basic 快照。"""
        if not screen_rows:
            return []
        trade_dates = self.engine._trade_dates()
        if len(trade_dates) < 6:
            return []
        dates = trade_dates[-6:]
        daily = self.engine._daily_window(trade_dates)
        daily = daily.copy()
        daily["ts_code"] = daily["ts_code"].astype(str).str.upper()
        daily["trade_date"] = daily["trade_date"].astype(str)
        daily_indexed = daily.set_index(["ts_code", "trade_date"])
        basic_by_date = {date: self._daily_basic_by_code(self.engine._daily_basic(date)) for date in dates[:-1]}

        rows: list[dict[str, Any]] = []
        for source in screen_rows:
            row = dict(source)
            code = str(row.get("ts_code", "")).upper()
            row["t_date"] = trade_date
            for offset, date in zip(range(5, 0, -1), dates[:-1]):
                prefix = f"tminus{offset}"
                daily_row = self._history_daily_row(daily_indexed, code, date)
                basic_row = basic_by_date[date].get(code, {})
                row[f"{prefix}_date"] = date
                row[f"{prefix}_pct_chg"] = self._round(daily_row.get("pct_chg"))
                row[f"{prefix}_turnover_rate"] = self._round(basic_row.get("turnover_rate"))
                row[f"{prefix}_turnover_rate_f"] = self._round(basic_row.get("turnover_rate_f"))
                row[f"{prefix}_market_cap_yi"] = self._round(self._as_float(basic_row.get("total_mv"), 0) / 10000)
                row[f"{prefix}_circ_mv_yi"] = self._round(self._as_float(basic_row.get("circ_mv"), 0) / 10000)
            rows.append(row)
        return rows

    @staticmethod
    def _history_daily_row(daily: pd.DataFrame, ts_code: str, trade_date: str) -> dict[str, Any]:
        try:
            value = daily.loc[(ts_code, trade_date)]
        except KeyError:
            return {}
        if isinstance(value, pd.DataFrame):
            value = value.iloc[-1]
        return value.to_dict() if isinstance(value, pd.Series) else {}

    @staticmethod
    def _daily_basic_by_code(data: pd.DataFrame) -> dict[str, dict[str, Any]]:
        if data is None or data.empty or "ts_code" not in data.columns:
            return {}
        result: dict[str, dict[str, Any]] = {}
        for _, row in data.iterrows():
            result[str(row.get("ts_code", "")).upper()] = row.to_dict()
        return result

    def _conditions(self) -> list[dict[str, Any]]:
        raw = getattr(self.config, "cross_momentum_conditions", None)
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict) and item.get("enabled", True)]

    def _sync_dynamic_config(self) -> None:
        """把动态条件的首个同类条件同步到旧字段，兼容推荐和结果说明。"""
        conditions = self._conditions()
        if not conditions:
            return
        stock_rank = next(
            (c for c in conditions if self._condition_object(c) == "stock" and self._condition_indicator(c) in {"stock_strength", "momentum"}),
            None,
        )
        if stock_rank:
            self.config.cross_momentum_days = self._condition_days(stock_rank)
            self.config.cross_momentum_rank_mode = str(stock_rank.get("rank_mode", "count"))
            self.config.cross_momentum_top_count = max(1, self._as_int(stock_rank.get("rank_value", stock_rank.get("value", 100)), 100))
            self.config.cross_momentum_top_percent = max(0.1, self._as_float(stock_rank.get("rank_value", stock_rank.get("value", 10)), 10))
            self.config.cross_momentum_stock_rank_enabled = True
        concept_rank = next(
            (c for c in conditions if self._condition_object(c) == "concept" and self._condition_indicator(c) in {"concept_strength", "momentum", "pct_chg"}),
            None,
        )
        if concept_rank:
            self.config.cross_momentum_concept_enabled = True
            self.config.cross_momentum_concept_days = self._condition_days(concept_rank)
            self.config.cross_momentum_concept_rank_mode = str(concept_rank.get("rank_mode", "count"))
            self.config.cross_momentum_concept_top_count = max(1, self._as_int(concept_rank.get("rank_value", concept_rank.get("value", 10)), 10))
            self.config.cross_momentum_concept_top_percent = max(0.1, self._as_float(concept_rank.get("rank_value", concept_rank.get("value", 10)), 10))

    @staticmethod
    def _condition_object(condition: dict[str, Any]) -> str:
        return str(condition.get("object", "stock") or "stock").strip().lower()

    @staticmethod
    def _condition_indicator(condition: dict[str, Any]) -> str:
        return str(condition.get("indicator", "") or "").strip().lower()

    @classmethod
    def _condition_offset(cls, condition: dict[str, Any]) -> int:
        return min(0, cls._as_int(condition.get("offset", 0), 0))

    @classmethod
    def _condition_days(cls, condition: dict[str, Any]) -> int:
        return max(1, cls._as_int(condition.get("window_days", condition.get("days", 10)), 10))

    @classmethod
    def _condition_uses_offset(cls, condition: dict[str, Any]) -> bool:
        return str(condition.get("time_type", "offset") or "offset").lower() == "offset"

    @classmethod
    def _condition_calculation(cls, condition: dict[str, Any]) -> str:
        value = str(condition.get("calculation", "value") or "value").strip().lower()
        return value if value in {"value", "sum", "ratio"} else "value"

    @classmethod
    def _condition_offsets(cls, condition: dict[str, Any]) -> list[int]:
        """返回一条条件需要的全部相对交易日，兼容旧版单日条件。"""
        calculation = cls._condition_calculation(condition)
        if calculation == "sum":
            start = min(0, cls._as_int(condition.get("range_start_offset", condition.get("offset", 0)), 0))
            end = min(0, cls._as_int(condition.get("range_end_offset", condition.get("offset", 0)), 0))
            low, high = sorted((start, end))
            return list(range(low, high + 1))
        if calculation == "ratio":
            return [
                min(0, cls._as_int(condition.get("offset", 0), 0)),
                min(0, cls._as_int(condition.get("compare_offset", -1), -1)),
            ]
        return [cls._condition_offset(condition)] if cls._condition_uses_offset(condition) else []

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _date_note(requested_date: str, trade_date: str) -> str:
        if requested_date and requested_date != trade_date:
            return f"输入日期 {requested_date} 非交易日或尚无最终数据，已使用 {trade_date}。"
        return "已使用输入日期对应的交易日。"

    @staticmethod
    def _snapshot_mode(trade_date: str, now: datetime | None = None) -> str:
        now = now or datetime.now()
        if trade_date != now.strftime("%Y%m%d"):
            return "history"
        current = now.time()
        if time(9, 30) <= current < time(11, 30):
            return "live"
        if time(11, 30) <= current < time(13, 0):
            return "lunch"
        if time(13, 0) <= current < time(15, 0):
            return "live"
        return "close"

    @staticmethod
    def _close_snapshot_at(trade_date: str) -> str:
        return f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 15:00:00"

    def _stock_features(
        self,
        daily: pd.DataFrame,
        basic: pd.DataFrame,
        daily_basic: pd.DataFrame,
        trade_dates: list[str],
        t1_daily_basic: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        data = daily.copy()
        data["ts_code"] = data["ts_code"].astype(str).str.upper()
        data["trade_date"] = data["trade_date"].astype(str)
        for col in ["open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]:
            if col not in data.columns:
                data[col] = 0
            data[col] = pd.to_numeric(data[col], errors="coerce")

        pivots = {
            col: data.pivot_table(index="ts_code", columns="trade_date", values=col, aggfunc="last")
            for col in ["open", "high", "low", "close", "pre_close", "pct_chg", "amount", "vol"]
        }
        pivots = {key: value.reindex(columns=trade_dates) for key, value in pivots.items()}
        codes = pivots["close"].index
        latest = trade_dates[-1]
        features = pd.DataFrame(index=codes)
        features["ts_code"] = features.index.astype(str)
        for target, source in [("open", "open"), ("high", "high"), ("low", "low"), ("close", "close"), ("pre_close", "pre_close"), ("pct_chg", "pct_chg"), ("amount", "amount"), ("vol", "vol")]:
            series = pivots[source][latest] if latest in pivots[source].columns else pd.Series(index=codes, dtype=float)
            features[target] = pd.to_numeric(series, errors="coerce")
        features["latest_price"] = features["close"]
        features["amount_yi"] = features["amount"] / 100000
        t1_date = trade_dates[-2] if len(trade_dates) >= 2 else ""
        t1_close = pivots["close"][t1_date] if t1_date in pivots["close"].columns else pd.Series(index=codes, dtype=float)
        t1_low = pivots["low"][t1_date] if t1_date in pivots["low"].columns else pd.Series(index=codes, dtype=float)
        t1_pre_close = pivots["pre_close"][t1_date] if t1_date in pivots["pre_close"].columns else pd.Series(index=codes, dtype=float)
        t1_amount = pivots["amount"][t1_date] if t1_date in pivots["amount"].columns else pd.Series(index=codes, dtype=float)
        features["t1_date"] = t1_date
        features["t1_close_price"] = pd.to_numeric(t1_close, errors="coerce")
        features["t1_low_pct"] = pd.to_numeric(t1_low, errors="coerce") / pd.to_numeric(t1_pre_close, errors="coerce").replace(0, pd.NA) * 100 - 100
        features["t1_amount_yi"] = pd.to_numeric(t1_amount, errors="coerce") / 100000

        dynamic_windows = {
            self._condition_days(condition)
            for condition in self._conditions()
            if self._condition_object(condition) == "stock"
            and self._condition_indicator(condition) in {"momentum", "stock_strength", "pct_chg"}
        }
        for window in sorted({2, 5, 10, 20, 30, 60, max(1, int(getattr(self.config, "cross_momentum_days", 10))), *dynamic_windows}):
            close = pivots["close"]
            base = close.shift(window, axis=1)[latest] if latest in close.columns else pd.Series(index=codes, dtype=float)
            features[f"ret{window}"] = features["close"] / pd.to_numeric(base, errors="coerce") * 100 - 100
            features[f"rps{window}"] = features[f"ret{window}"].rank(pct=True) * 100

        amount = pivots["amount"]
        features["avg_amount5"] = amount.iloc[:, -min(5, len(amount.columns)):].mean(axis=1)
        features["avg_amount20"] = amount.iloc[:, -min(20, len(amount.columns)):].mean(axis=1)
        features["amount_ratio"] = features["avg_amount5"] / features["avg_amount20"].replace(0, pd.NA)

        basic_cols = [col for col in ["ts_code", "symbol", "name", "industry", "market", "stock_type", "list_date"] if col in basic.columns]
        basic_use = basic[basic_cols].copy()
        basic_use["ts_code"] = basic_use["ts_code"].astype(str).str.upper()
        features = features.reset_index(drop=True).merge(basic_use, on="ts_code", how="left")
        features["symbol"] = features.get("symbol", features["ts_code"].str[:6]).fillna(features["ts_code"].str[:6]).astype(str).str.zfill(6)
        features["name"] = features.get("name", "").fillna("").astype(str)
        features["industry"] = features.get("industry", "").fillna("未分类").replace("", "未分类")
        features["market"] = features.get("market", "").fillna("").astype(str)
        features["stock_type"] = features.get("stock_type", "").fillna("").astype(str)

        db = daily_basic.copy() if daily_basic is not None else pd.DataFrame()
        if "ts_code" not in db.columns:
            db["ts_code"] = pd.Series(dtype=str)
        db["ts_code"] = db["ts_code"].astype(str).str.upper()
        for col in ["turnover_rate", "turnover_rate_f", "volume_ratio", "total_mv", "circ_mv"]:
            if col not in db.columns:
                db[col] = 0
            db[col] = pd.to_numeric(db[col], errors="coerce").fillna(0)
        features = features.merge(
            db[["ts_code", "turnover_rate", "turnover_rate_f", "volume_ratio", "total_mv", "circ_mv"]],
            on="ts_code",
            how="left",
        )
        features["market_cap_yi"] = pd.to_numeric(features["total_mv"], errors="coerce").fillna(0) / 10000
        features["circ_mv_yi"] = pd.to_numeric(features["circ_mv"], errors="coerce").fillna(0) / 10000
        t1_db = t1_daily_basic.copy() if t1_daily_basic is not None else pd.DataFrame()
        if "ts_code" not in t1_db.columns:
            t1_db["ts_code"] = pd.Series(dtype=str)
        t1_db["ts_code"] = t1_db["ts_code"].astype(str).str.upper()
        for col in ["turnover_rate", "turnover_rate_f", "total_mv", "circ_mv"]:
            if col not in t1_db.columns:
                t1_db[col] = 0
            t1_db[col] = pd.to_numeric(t1_db[col], errors="coerce").fillna(0)
        t1_db = t1_db[["ts_code", "turnover_rate", "turnover_rate_f", "total_mv", "circ_mv"]].rename(
            columns={
                "turnover_rate": "t1_turnover_rate",
                "turnover_rate_f": "t1_turnover_rate_f",
                "total_mv": "t1_total_mv",
                "circ_mv": "t1_circ_mv",
            }
        )
        features = features.merge(t1_db, on="ts_code", how="left")
        features["t1_market_cap_yi"] = pd.to_numeric(features["t1_total_mv"], errors="coerce").fillna(0) / 10000
        features["t1_circ_mv_yi"] = pd.to_numeric(features["t1_circ_mv"], errors="coerce").fillna(0) / 10000
        return features

    def _market_mask(self, data: pd.DataFrame) -> pd.Series:
        scope = getattr(self.config, "cross_momentum_market_scope", ["main_board"])
        if isinstance(scope, str):
            scope = [item.strip() for item in scope.split(",") if item.strip()]
        scope = set(scope or ["main_board"])
        symbol = data.get("symbol", data["ts_code"].astype(str).str[:6]).astype(str).str.zfill(6)
        ts_code = data["ts_code"].astype(str).str.upper()
        stock_type = data.get("stock_type", pd.Series("", index=data.index)).astype(str)
        market = data.get("market", pd.Series("", index=data.index)).astype(str)

        if "all" in scope:
            mask = pd.Series(True, index=data.index)
        else:
            mask = pd.Series(False, index=data.index)
            if "main_board" in scope:
                mask |= stock_type.str.contains("主板", na=False) | symbol.str.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))
            if "chinext" in scope:
                mask |= symbol.str.startswith(("300", "301"))
            if "star" in scope:
                mask |= symbol.str.startswith(("688", "689"))
            if "bj" in scope:
                mask |= ts_code.str.endswith(".BJ") | symbol.str.startswith(("4", "8", "92")) | market.str.contains("北", na=False)

        if bool(getattr(self.config, "cross_momentum_exclude_st", True)):
            mask &= ~data["name"].astype(str).str.upper().str.contains("ST", regex=False, na=False)
        if bool(getattr(self.config, "cross_momentum_exclude_star", True)):
            mask &= ~symbol.str.startswith(("688", "689"))
        if bool(getattr(self.config, "cross_momentum_exclude_chinext", True)):
            mask &= ~symbol.str.startswith(("300", "301"))
        if bool(getattr(self.config, "cross_momentum_exclude_bj", True)):
            mask &= ~(ts_code.str.endswith(".BJ") | symbol.str.startswith(("4", "8", "92")) | market.str.contains("北", na=False))
        return mask

    def _attach_stock_rank(self, features: pd.DataFrame, days: int) -> pd.DataFrame:
        result = features.copy()
        return_field = f"ret{days}"
        result["stock_momentum"] = pd.to_numeric(result.get(return_field, pd.Series(index=result.index)), errors="coerce")
        universe = result[(result["close"] > 0) & result["stock_momentum"].notna()].copy()
        universe = universe.sort_values(["stock_momentum", "ts_code"], ascending=[False, True]).reset_index()
        universe["stock_rank"] = range(1, len(universe) + 1)
        if len(universe) == 1:
            universe["stock_rank_pct"] = 100.0
        else:
            universe["stock_rank_pct"] = (len(universe) - universe["stock_rank"] + 1) / len(universe) * 100
        rank_map = universe.set_index("ts_code")[["stock_rank", "stock_rank_pct"]].to_dict("index")
        result["stock_rank"] = result["ts_code"].map(lambda code: rank_map.get(str(code), {}).get("stock_rank", math.nan))
        result["stock_rank_pct"] = result["ts_code"].map(lambda code: rank_map.get(str(code), {}).get("stock_rank_pct", math.nan))
        result["stock_rank"] = pd.to_numeric(result["stock_rank"], errors="coerce")
        result["stock_rank_pct"] = pd.to_numeric(result["stock_rank_pct"], errors="coerce")
        mode = str(getattr(self.config, "cross_momentum_rank_mode", "count") or "count").strip().lower()
        if mode == "percentile":
            pct = max(0.1, min(100.0, float(getattr(self.config, "cross_momentum_top_percent", 10))))
            cutoff = max(1, math.ceil(len(universe) * pct / 100))
            result["stock_rank_limit"] = cutoff
        else:
            result["stock_rank_limit"] = max(1, int(getattr(self.config, "cross_momentum_top_count", 100)))
        result["stock_rank_pass"] = result["stock_rank"] <= result["stock_rank_limit"]
        return result

    def _attach_concept_top(
        self,
        features: pd.DataFrame,
        trade_dates: list[str],
        days: int,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        radar = ConceptRadarStrategy(self.engine)
        concepts = radar._ths_concepts()
        concept_daily = radar._ths_daily_window(trade_dates, set(concepts["ts_code"].astype(str)))
        if concept_daily.empty:
            raise RuntimeError("概念强度已启用，但同花顺概念日线为空。")
        concept_daily["ts_code"] = concept_daily["ts_code"].astype(str)
        concept_daily["trade_date"] = concept_daily["trade_date"].astype(str)
        concept_daily["close"] = pd.to_numeric(concept_daily["close"], errors="coerce")
        close = concept_daily.pivot_table(index="ts_code", columns="trade_date", values="close", aggfunc="last").sort_index(axis=1)
        close = close.dropna(axis=1, how="all")
        if close.shape[1] <= days:
            raise RuntimeError(f"概念行情历史不足，无法计算近 {days} 个交易日强度。")
        latest = close.columns[-1]
        base = close.iloc[:, -days - 1]
        latest_close = close[latest]
        concept_data = concepts.rename(columns={"ts_code": "concept_code", "name": "concept_name"}).copy()
        concept_data["concept_code"] = concept_data["concept_code"].astype(str)
        # ``latest_close`` 以概念代码为索引，而概念列表通常保留数字行索引。
        # 直接赋值会按索引对齐，造成收益全部变为 NaN，进而错误提示“概念 Top N 结果为空”。
        concept_return = ((latest_close / base - 1) * 100).rename("concept_return")
        concept_return.index = concept_return.index.astype(str)
        concept_data["concept_return"] = concept_data["concept_code"].map(concept_return)
        concept_data = concept_data.dropna(subset=["concept_return"]).sort_values(["concept_return", "concept_code"], ascending=[False, True]).reset_index(drop=True)
        concept_data["concept_rank"] = range(1, len(concept_data) + 1)
        mode = str(getattr(self.config, "cross_momentum_concept_rank_mode", "count") or "count").strip().lower()
        if mode == "percentile":
            pct = max(0.1, min(100.0, float(getattr(self.config, "cross_momentum_concept_top_percent", 10))))
            cutoff = max(1, math.ceil(len(concept_data) * pct / 100))
        else:
            cutoff = max(1, int(getattr(self.config, "cross_momentum_concept_top_count", 10)))
        top = concept_data[concept_data["concept_rank"] <= cutoff].copy()
        if top.empty:
            raise RuntimeError("概念 Top N 结果为空。")

        members = radar._ths_members(top[["concept_code", "concept_name"]])
        if members.empty:
            raise RuntimeError("概念强度已启用，但未读取到 Top N 概念成员。")
        members["ts_code"] = members["ts_code"].astype(str).str.upper()
        member_data = members.merge(
            top[["concept_code", "concept_name", "concept_return", "concept_rank"]],
            on=["concept_code", "concept_name"],
            how="inner",
        )
        member_data = member_data[member_data["ts_code"].isin(set(features["ts_code"].astype(str)))].copy()
        if member_data.empty:
            raise RuntimeError("Top N 概念成员与当前股票池没有交集。")
        agg = member_data.groupby("ts_code", as_index=False).agg(
            concept_name=("concept_name", lambda s: "、".join(dict.fromkeys(str(item) for item in s))),
            concept_code=("concept_code", lambda s: "、".join(dict.fromkeys(str(item) for item in s))),
            concept_rank=("concept_rank", "min"),
            concept_return=("concept_return", "max"),
            concept_count=("concept_code", "nunique"),
        )
        result = features.merge(agg, on="ts_code", how="left", suffixes=("", "_concept"))
        result["concept_name"] = result["concept_name"].fillna("")
        result["concept_code"] = result["concept_code"].fillna("")
        result["concept_rank"] = pd.to_numeric(result["concept_rank"], errors="coerce")
        result["concept_return"] = pd.to_numeric(result["concept_return"], errors="coerce")
        result["concept_count"] = pd.to_numeric(result["concept_count"], errors="coerce").fillna(0).astype(int)
        result["concept_pass"] = result["concept_count"] > 0
        top_rows = []
        for _, row in top.iterrows():
            top_rows.append(
                {
                    "concept_code": str(row.get("concept_code", "")),
                    "concept_name": str(row.get("concept_name", "")),
                    "concept_rank": int(row.get("concept_rank", 0) or 0),
                    "concept_return": self._round(row.get("concept_return")),
                    "data_date": str(latest),
                }
            )
        return result, top_rows

    def _apply_conditions(
        self,
        features: pd.DataFrame,
        daily: pd.DataFrame,
        trade_dates: list[str],
    ) -> tuple[pd.DataFrame, pd.Series]:
        if self._conditions():
            return self._apply_dynamic_conditions(features, daily, trade_dates)
        result = features.copy()
        mask = result["close"].fillna(0) > 0
        if bool(getattr(self.config, "cross_momentum_stock_rank_enabled", True)):
            mask &= result["stock_rank_pass"].fillna(False)
        if bool(getattr(self.config, "cross_momentum_concept_enabled", False)):
            mask &= result["concept_pass"].fillna(False)

        limit_up_enabled = bool(getattr(self.config, "cross_momentum_limit_up_enabled", True))
        limit_up_offset = int(getattr(self.config, "cross_momentum_limit_up_offset", -3))
        if limit_up_enabled:
            codes = self._limit_up_codes(self._offset_date(trade_dates, limit_up_offset))
            if codes is None:
                raise RuntimeError(f"无法读取 T{limit_up_offset:+d} 涨停数据，不能安全执行涨停条件。")
            result["limit_up_pass"] = result["ts_code"].astype(str).isin(codes)
            mask &= result["limit_up_pass"]
        else:
            result["limit_up_pass"] = True

        not_up_enabled = bool(getattr(self.config, "cross_momentum_limit_not_up_enabled", True))
        not_up_offset = int(getattr(self.config, "cross_momentum_limit_not_up_offset", -2))
        if not_up_enabled:
            codes = self._limit_up_codes(self._offset_date(trade_dates, not_up_offset))
            if codes is None:
                raise RuntimeError(f"无法读取 T{not_up_offset:+d} 涨停数据，不能安全执行未涨停条件。")
            result["limit_not_up_pass"] = ~result["ts_code"].astype(str).isin(codes)
            mask &= result["limit_not_up_pass"]
        else:
            result["limit_not_up_pass"] = True

        low_enabled = bool(getattr(self.config, "cross_momentum_low_enabled", True))
        low_offset = int(getattr(self.config, "cross_momentum_low_offset", -1))
        low_date = self._offset_date(trade_dates, low_offset)
        result["low_pct"] = self._daily_metric(daily, low_date, "low_pct").reindex(result["ts_code"].astype(str)).to_numpy()
        if low_enabled:
            threshold = float(getattr(self.config, "cross_momentum_low_threshold", -6.0))
            result["low_pass"] = pd.to_numeric(result["low_pct"], errors="coerce") < threshold
            mask &= result["low_pass"].fillna(False)
        else:
            result["low_pass"] = True

        open_enabled = bool(getattr(self.config, "cross_momentum_open_enabled", False))
        open_offset = int(getattr(self.config, "cross_momentum_open_offset", 0))
        open_date = self._offset_date(trade_dates, open_offset)
        result["open_pct"] = self._daily_metric(daily, open_date, "open_pct").reindex(result["ts_code"].astype(str)).to_numpy()
        if open_enabled:
            threshold = float(getattr(self.config, "cross_momentum_open_threshold", 0.0))
            operator = str(getattr(self.config, "cross_momentum_open_operator", "gt") or "gt").lower()
            if operator in {"ge", "gte", ">="}:
                result["open_pass"] = pd.to_numeric(result["open_pct"], errors="coerce") >= threshold
            else:
                result["open_pass"] = pd.to_numeric(result["open_pct"], errors="coerce") > threshold
            mask &= result["open_pass"].fillna(False)
        else:
            result["open_pass"] = True

        result["limit_up_offset_label"] = self._offset_label(limit_up_offset)
        result["limit_not_up_offset_label"] = self._offset_label(not_up_offset)
        result["low_offset_label"] = self._offset_label(low_offset)
        result["open_offset_label"] = self._offset_label(open_offset)
        result["condition_pass"] = mask
        return result, mask

    def _apply_dynamic_conditions(
        self,
        features: pd.DataFrame,
        daily: pd.DataFrame,
        trade_dates: list[str],
    ) -> tuple[pd.DataFrame, pd.Series]:
        """执行条件构建器提交的条件行；所有启用条件默认使用 AND。"""
        result = features.copy()
        mask = result["close"].fillna(0) > 0
        for index, condition in enumerate(self._conditions(), start=1):
            obj = self._condition_object(condition)
            indicator = self._condition_indicator(condition)
            operator = str(condition.get("operator", "eq") or "eq").strip().lower()
            passed = pd.Series(False, index=result.index)
            label = str(condition.get("label", "") or "").strip()

            calculation = self._condition_calculation(condition)
            if obj == "stock" and indicator == "limit_up" and calculation == "value":
                date = self._offset_date(trade_dates, self._condition_offset(condition))
                codes = self._limit_up_codes(date)
                if codes is None:
                    raise RuntimeError(f"无法读取 {self._offset_label(self._condition_offset(condition))} 涨停数据。")
                value = str(condition.get("value", "limit_up") or "limit_up").lower()
                passed = result["ts_code"].astype(str).isin(codes)
                if value in {"not_limit_up", "not_up", "未涨停"} or operator in {"ne", "!=", "not"}:
                    passed = ~passed
            elif obj == "stock" and indicator in {"low_pct", "open_pct", "pct_chg", "turnover_rate", "turnover_rate_f", "amount", "total_mv", "circ_mv"}:
                values = self._condition_stock_values(result, daily, trade_dates, condition)
                result[f"condition_{index}_value"] = values
                if calculation == "value" and indicator in {"low_pct", "open_pct"}:
                    date = self._offset_date(trade_dates, self._condition_offset(condition))
                    result[indicator] = values
                    result[f"condition_{indicator}_date"] = date
                threshold = self._as_float(condition.get("threshold", condition.get("value", 0)), 0)
                passed = self._compare(values, operator, threshold)
            elif obj == "stock" and indicator in {"momentum", "stock_strength"}:
                days = self._condition_days(condition)
                values = pd.to_numeric(result.get(f"ret{days}", pd.Series(index=result.index)), errors="coerce")
                passed = self._rank_compare(values, condition)
                result[f"condition_{index}_value"] = values
            elif obj == "concept" and indicator in {"momentum", "concept_strength", "pct_chg"}:
                values = pd.to_numeric(result.get("concept_return", pd.Series(index=result.index)), errors="coerce")
                if indicator == "concept_strength":
                    passed = result.get("concept_pass", pd.Series(False, index=result.index)).fillna(False)
                else:
                    threshold = self._as_float(condition.get("threshold", condition.get("value", 0)), 0)
                    passed = self._compare(values, operator, threshold)
                result[f"condition_{index}_value"] = values
            else:
                self.log(f"暂不支持的动态条件：{obj}/{indicator}，该条件按不通过处理。")

            result[f"condition_{index}_pass"] = passed.fillna(False)
            mask &= result[f"condition_{index}_pass"]
            actual = pd.to_numeric(result.get(f"condition_{index}_value", pd.Series(index=result.index)), errors="coerce")
            actual_valid = actual.dropna()
            summary = "无可用数值"
            if not actual_valid.empty:
                unit = self._condition_value_unit(condition)
                summary = f"范围 {actual_valid.min():.2f}{unit} ~ {actual_valid.max():.2f}{unit}"
            self._condition_diagnostics.append(
                {
                    "index": index,
                    "condition": self._condition_formula(condition),
                    "pass_count": int(result[f"condition_{index}_pass"].sum()),
                    "total_count": int(len(result)),
                    "missing_count": int(len(result) - len(actual_valid)) if indicator != "limit_up" else 0,
                    "actual_summary": summary,
                }
            )
            if label:
                result[f"condition_{index}_label"] = label
        result["condition_pass"] = mask
        return result, mask

    @staticmethod
    def _compare(values: Any, operator: str, threshold: float) -> pd.Series:
        series = pd.Series(values)
        if operator in {"gt", ">"}:
            return series > threshold
        if operator in {"ge", "gte", ">="}:
            return series >= threshold
        if operator in {"lt", "<"}:
            return series < threshold
        if operator in {"le", "lte", "<="}:
            return series <= threshold
        if operator in {"ne", "!="}:
            return series != threshold
        return series == threshold

    @classmethod
    def _rank_compare(cls, values: pd.Series, condition: dict[str, Any]) -> pd.Series:
        rank = values.rank(method="first", ascending=False, na_option="keep")
        mode = str(condition.get("rank_mode", "count") or "count").lower()
        raw_value = condition.get("rank_value", condition.get("value", 20))
        if mode in {"percentile", "percent", "top_percent"}:
            try:
                cutoff = max(0.1, min(100.0, float(raw_value)))
            except (TypeError, ValueError):
                cutoff = 10.0
            limit = max(1, math.ceil(rank.notna().sum() * cutoff / 100))
        else:
            limit = max(1, cls._as_int(raw_value, 20))
        return rank <= limit

    def _limit_up_codes(self, trade_date: str) -> set[str] | None:
        if trade_date in self._limit_cache:
            return self._limit_cache[trade_date]
        frame = self.engine._limit_list(trade_date)
        if frame is None or frame.empty or "ts_code" not in frame.columns:
            # limit_list_d 空表也可能表示该日没有返回任何涨停事件；
            # 让 T-k 涨停条件自然得到 0 个命中，未涨停条件则全部通过，
            # 同时把状态写入日志，避免静默伪造结果。
            self._limit_cache[trade_date] = set()
            self._limit_status[trade_date] = "空"
            self.log(f"{trade_date} 涨跌停列表为空，按无涨停事件处理。")
            return set()
        data = frame.copy()
        data["ts_code"] = data["ts_code"].astype(str).str.upper()
        up = pd.Series(False, index=data.index)
        has_marker = False
        if "limit_type" in data.columns:
            has_marker = True
            up |= data["limit_type"].astype(str).str.upper().str.contains("U|涨停", regex=True, na=False)
        if "limit" in data.columns:
            has_marker = True
            up |= data["limit"].astype(str).str.contains("涨|U|UP", case=False, regex=True, na=False)
        if not has_marker:
            up = pd.Series(True, index=data.index)
        codes = set(data.loc[up, "ts_code"].tolist())
        self._limit_cache[trade_date] = codes
        self._limit_status[trade_date] = "正常"
        return codes

    @staticmethod
    def _offset_date(trade_dates: list[str], offset: int) -> str:
        position = len(trade_dates) - 1 + int(offset)
        if position < 0 or position >= len(trade_dates):
            raise RuntimeError(f"T{offset:+d} 超出可用交易日范围。")
        return trade_dates[position]

    @staticmethod
    def _offset_label(offset: int) -> str:
        if offset == 0:
            return "T"
        return f"T{offset:+d}"

    @staticmethod
    def _daily_metric(daily: pd.DataFrame, trade_date: str, metric: str) -> pd.Series:
        frame = daily[daily["trade_date"].astype(str) == str(trade_date)].copy()
        if frame.empty:
            return pd.Series(dtype=float)
        frame["ts_code"] = frame["ts_code"].astype(str).str.upper()
        for col in ["open", "low", "close", "pct_chg", "pre_close"]:
            if col not in frame.columns:
                frame[col] = pd.NA
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if metric == "pct_chg":
            values = frame["pct_chg"]
        elif metric == "open_pct":
            values = frame["open"] / frame["pre_close"].replace(0, pd.NA) * 100 - 100
        else:
            values = frame["low"] / frame["pre_close"].replace(0, pd.NA) * 100 - 100
        return pd.Series(values.to_numpy(), index=frame["ts_code"].tolist(), dtype=float)

    def _daily_basic_metric(self, trade_date: str, metric: str) -> pd.Series:
        """按交易日读取 daily_basic 指标，并做实例级缓存。

        ``turnover_rate_f`` 是自由流通股本口径的实换手率。这里不把它静默
        降级为普通换手率，缺失数据会自然导致条件不通过。
        """
        key = (str(trade_date), metric)
        if key in self._daily_basic_metric_cache:
            return self._daily_basic_metric_cache[key]
        frame = self.engine._daily_basic(str(trade_date))
        if frame is None or frame.empty or "ts_code" not in frame.columns or metric not in frame.columns:
            values = pd.Series(dtype=float)
        else:
            data = frame[["ts_code", metric]].copy()
            data["ts_code"] = data["ts_code"].astype(str).str.upper()
            values = pd.Series(pd.to_numeric(data[metric], errors="coerce").to_numpy(), index=data["ts_code"].tolist(), dtype=float)
        self._daily_basic_metric_cache[key] = values
        return values

    def _single_condition_metric(
        self,
        result: pd.DataFrame,
        daily: pd.DataFrame,
        trade_dates: list[str],
        indicator: str,
        offset: int,
    ) -> pd.Series:
        date = self._offset_date(trade_dates, offset)
        if indicator in {"low_pct", "open_pct", "pct_chg"}:
            metric = self._daily_metric(daily, date, indicator)
        else:
            metric = self._daily_basic_metric(date, indicator)
        return pd.Series(
            pd.to_numeric(metric.reindex(result["ts_code"].astype(str)).to_numpy(), errors="coerce"),
            index=result.index,
            dtype=float,
        )

    def _condition_stock_values(
        self,
        result: pd.DataFrame,
        daily: pd.DataFrame,
        trade_dates: list[str],
        condition: dict[str, Any],
    ) -> pd.Series:
        """计算单日原值、连续交易日累计值或两日比值。"""
        indicator = self._condition_indicator(condition)
        calculation = self._condition_calculation(condition)
        if calculation == "sum":
            if indicator != "pct_chg":
                self.log(f"区间累计目前只支持涨跌幅，{indicator} 按无效条件处理。")
                return pd.Series(pd.NA, index=result.index, dtype=float)
            offsets = self._condition_offsets(condition)
            values = pd.Series(0.0, index=result.index)
            valid = pd.Series(True, index=result.index)
            for offset in offsets:
                item = self._single_condition_metric(result, daily, trade_dates, indicator, offset)
                values = values.add(item.fillna(0), fill_value=0)
                valid &= item.notna()
            return values.where(valid)
        if calculation == "ratio":
            numerator = self._single_condition_metric(result, daily, trade_dates, indicator, self._condition_offset(condition))
            denominator = self._single_condition_metric(
                result,
                daily,
                trade_dates,
                indicator,
                min(0, self._as_int(condition.get("compare_offset", -1), -1)),
            )
            return numerator / denominator.replace(0, pd.NA)
        return self._single_condition_metric(result, daily, trade_dates, indicator, self._condition_offset(condition))

    @classmethod
    def _condition_value_unit(cls, condition: dict[str, Any]) -> str:
        calculation = cls._condition_calculation(condition)
        indicator = cls._condition_indicator(condition)
        if calculation == "ratio":
            return " 倍"
        if indicator in {"low_pct", "open_pct", "pct_chg", "turnover_rate", "turnover_rate_f"}:
            return "%"
        return ""

    def _add_recommendation(
        self,
        data: pd.DataFrame,
        stock_days: int,
        concept_days: int,
        realtime_applied: bool,
    ) -> pd.DataFrame:
        result = data.copy()
        if result.empty:
            result["recommend_score"] = pd.Series(dtype=float)
            result["recommended"] = pd.Series(dtype=bool)
            result["reason"] = pd.Series(dtype=str)
            return result
        result["momentum_score"] = pd.to_numeric(result["stock_rank_pct"], errors="coerce").fillna(0).clip(0, 100)
        if bool(getattr(self.config, "cross_momentum_concept_enabled", False)):
            concept_rank = pd.to_numeric(result["concept_rank"], errors="coerce")
            concept_score = (100 - (concept_rank - 1) / max(1, float(getattr(self.config, "cross_momentum_concept_top_count", 10))) * 100).clip(0, 100).fillna(0)
            result["concept_score"] = concept_score
            weights = (0.55, 0.20, 0.15, 0.10)
            result["recommend_score"] = (
                result["momentum_score"] * weights[0]
                + result["concept_score"] * weights[1]
                + result["amount_ratio"].fillna(0).clip(0, 2.5) / 2.5 * 100 * weights[2]
                + result["pct_chg"].fillna(0).clip(-5, 5).add(5) * 10 * weights[3]
            )
        else:
            result["concept_score"] = pd.NA
            result["recommend_score"] = (
                result["momentum_score"] * 0.65
                + result["amount_ratio"].fillna(0).clip(0, 2.5) / 2.5 * 100 * 0.20
                + result["pct_chg"].fillna(0).clip(-5, 5).add(5) * 10 * 0.15
            )
        result["recommend_score"] = result["recommend_score"].clip(0, 100).round(2)
        result["recommended"] = False
        result["risk_flags"] = result.apply(self._risk_flags, axis=1)
        result["reason"] = result.apply(
            lambda row: self._reason(row, stock_days, concept_days, realtime_applied),
            axis=1,
        )
        return result

    @staticmethod
    def _risk_flags(row: pd.Series) -> str:
        flags: list[str] = []
        if float(row.get("amount_yi", 0) or 0) <= 0:
            flags.append("成交额缺失")
        if float(row.get("amount_ratio", 0) or 0) > 3:
            flags.append("量能过猛")
        if float(row.get("ret10", 0) or 0) > 35:
            flags.append("10日过热")
        return "；".join(flags) if flags else "正常"

    def _reason(self, row: pd.Series, stock_days: int, concept_days: int, realtime_applied: bool) -> str:
        parts = [
            f"个股近{stock_days}日涨幅{self._fmt(row.get('stock_momentum'))}%、排名第{self._int(row.get('stock_rank'))}",
            f"{row.get('limit_up_offset_label', 'T-3')}涨停",
            f"{row.get('limit_not_up_offset_label', 'T-2')}未涨停",
            f"{row.get('low_offset_label', 'T-1')}盘中最低{self._fmt(row.get('low_pct'))}%",
        ]
        if bool(getattr(self.config, "cross_momentum_open_enabled", False)):
            parts.append(f"{row.get('open_offset_label', 'T')}开盘涨幅{self._fmt(row.get('open_pct'))}%")
        if bool(getattr(self.config, "cross_momentum_concept_enabled", False)):
            parts.append(f"概念近{concept_days}日排名第{self._int(row.get('concept_rank'))}（{row.get('concept_name', '')}）")
        if realtime_applied:
            parts.append("使用实时快照")
        risk = str(row.get("risk_flags", "") or "")
        if risk and risk != "正常":
            parts.append(f"风险：{risk}")
        return "；".join(parts)

    def _rows(self, data: pd.DataFrame, trade_date: str, realtime_applied: bool) -> list[dict[str, Any]]:
        if data.empty:
            return []
        rows: list[dict[str, Any]] = []
        for _, row in data.iterrows():
            code = str(row.get("symbol", str(row.get("ts_code", ""))[:6])).zfill(6)
            pre_close = self._as_float(row.get("pre_close"), 0)
            open_pct = (self._as_float(row.get("open"), 0) / pre_close - 1) * 100 if pre_close else ""
            high_pct = (self._as_float(row.get("high"), 0) / pre_close - 1) * 100 if pre_close else ""
            low_pct = (self._as_float(row.get("low"), 0) / pre_close - 1) * 100 if pre_close else ""
            rows.append(
                {
                    "ts_code": str(row.get("ts_code", "")),
                    "code": code,
                    "name": str(row.get("name", "")),
                    "stock_type": str(row.get("stock_type", "")),
                    "industry": str(row.get("industry", "")),
                    "open_price": self._round(row.get("open")),
                    "high_price": self._round(row.get("high")),
                    "low_price": self._round(row.get("low")),
                    "close_price": self._round(row.get("close")) if not realtime_applied else "",
                    "latest_price": self._round(row.get("latest_price")),
                    "pre_close": self._round(row.get("pre_close")),
                    "pct_chg": self._round(row.get("pct_chg")),
                    "t_pct_chg": self._round(row.get("pct_chg")),
                    "t_open_pct": self._round(open_pct),
                    "t_high_pct": self._round(high_pct),
                    "t_low_pct": self._round(low_pct),
                    "stock_momentum": self._round(row.get("stock_momentum")),
                    "stock_rank": self._int_or_blank(row.get("stock_rank")),
                    "stock_rank_pct": self._round(row.get("stock_rank_pct")),
                    "concept_name": str(row.get("concept_name", "")),
                    "concept_rank": self._int_or_blank(row.get("concept_rank")),
                    "concept_return": self._round(row.get("concept_return")),
                    "concept_count": self._int(row.get("concept_count")),
                    "low_pct": self._round(row.get("low_pct")),
                    "open_pct": self._round(row.get("open_pct")),
                    "condition_low_date": str(row.get("condition_low_pct_date", "")),
                    "condition_open_date": str(row.get("condition_open_pct_date", "")),
                    "t1_trade_date": str(row.get("t1_date", "")),
                    "t1_low_pct": self._round(row.get("t1_low_pct")),
                    "t1_close_price": self._round(row.get("t1_close_price")),
                    "t1_amount_yi": self._round(row.get("t1_amount_yi")),
                    "t1_turnover_rate": self._round(row.get("t1_turnover_rate")),
                    "t1_turnover_rate_f": self._round(row.get("t1_turnover_rate_f")),
                    "t1_market_cap_yi": self._round(row.get("t1_market_cap_yi")),
                    "t1_circ_mv_yi": self._round(row.get("t1_circ_mv_yi")),
                    "amount_yi": self._round(row.get("amount_yi")),
                    "amount_ratio": self._round(row.get("amount_ratio")),
                    "turnover_rate": self._round(row.get("turnover_rate")),
                    "turnover_rate_f": self._round(row.get("turnover_rate_f")),
                    "volume_ratio": self._round(row.get("volume_ratio")),
                    "market_cap_yi": self._round(row.get("market_cap_yi")),
                    "limit_up_pass": "是" if bool(row.get("limit_up_pass")) else "否",
                    "limit_not_up_pass": "是" if bool(row.get("limit_not_up_pass")) else "否",
                    "low_pass": "是" if bool(row.get("low_pass")) else "否",
                    "open_pass": "是" if bool(row.get("open_pass")) else "否",
                    "recommend_rank": self._int_or_blank(row.get("recommend_rank")),
                    "recommended": bool(row.get("recommended", False)),
                    "recommend_score": self._round(row.get("recommend_score")),
                    "momentum_score": self._round(row.get("momentum_score")),
                    "concept_score": self._round(row.get("concept_score")),
                    "risk_flags": str(row.get("risk_flags", "")),
                    "reason": str(row.get("reason", "")),
                    "ths_url": f"https://stockpage.10jqka.com.cn/{code}/",
                    "trade_date": trade_date,
                    "price_label": "最新价" if realtime_applied else "收盘价",
                    "data_state": "盘中实时快照" if realtime_applied else "日线收盘",
                }
            )
        return rows

    def _payload(
        self,
        trade_date: str,
        requested_date: str,
        date_note: str,
        snapshot_mode: str,
        rows: list[dict[str, Any]],
        recommended: list[dict[str, Any]],
        concepts: list[dict[str, Any]],
        realtime_applied: bool,
        stock_universe_count: int,
    ) -> dict[str, Any]:
        snapshot = getattr(self.engine, "_last_realtime_snapshot", {}) or {}
        return {
            "ok": True,
            "strategy": "cross_sectional_momentum",
            "strategy_name": "横截面动量策略",
            "trade_date": trade_date,
            "requested_date": requested_date,
            "date_note": date_note,
            "rows": rows,
            "recommended_rows": recommended,
            "concepts": concepts,
            "condition_diagnostics": getattr(self, "_condition_diagnostics", []),
            "market": {
                "trade_date": trade_date,
                "requested_date": requested_date,
                "date_note": date_note,
                "stock_universe_count": stock_universe_count,
                "result_count": len(rows),
                "recommend_count": len(recommended),
                "snapshot_mode": snapshot_mode,
                "snapshot_at": str(snapshot.get("quote_at", "")),
                "realtime_enabled": bool(getattr(self.config, "cross_momentum_realtime_enabled", False)),
                "realtime_applied": realtime_applied,
                "realtime_source": str(snapshot.get("source", "日线收盘")),
                "stock_days": int(getattr(self.config, "cross_momentum_days", 10)),
                "stock_rank_mode": str(getattr(self.config, "cross_momentum_rank_mode", "count")),
                "stock_top_count": int(getattr(self.config, "cross_momentum_top_count", 100)),
                "stock_top_percent": float(getattr(self.config, "cross_momentum_top_percent", 10)),
                "concept_enabled": bool(getattr(self.config, "cross_momentum_concept_enabled", False)),
                "concept_days": int(getattr(self.config, "cross_momentum_concept_days", 8)),
                "concept_top_count": int(getattr(self.config, "cross_momentum_concept_top_count", 10)),
                "conditions": self._conditions(),
                "selection_formula": self._selection_formula(),
            },
            "logs": self.engine.logs,
        }

    def _selection_formula(self) -> str:
        conditions = self._conditions()
        if conditions:
            return "；".join(self._condition_formula(condition) for condition in conditions)
        parts: list[str] = []
        if bool(getattr(self.config, "cross_momentum_stock_rank_enabled", True)):
            mode = str(getattr(self.config, "cross_momentum_rank_mode", "count"))
            parts.append(
                f"个股近{int(getattr(self.config, 'cross_momentum_days', 10))}日涨幅前{int(getattr(self.config, 'cross_momentum_top_count', 100))}名"
                if mode == "count"
                else f"个股近{int(getattr(self.config, 'cross_momentum_days', 10))}日涨幅前{float(getattr(self.config, 'cross_momentum_top_percent', 10)):g}%"
            )
        if bool(getattr(self.config, "cross_momentum_limit_up_enabled", True)):
            parts.append(f"T{int(getattr(self.config, 'cross_momentum_limit_up_offset', -3)):+d}涨停")
        if bool(getattr(self.config, "cross_momentum_limit_not_up_enabled", True)):
            parts.append(f"T{int(getattr(self.config, 'cross_momentum_limit_not_up_offset', -2)):+d}未涨停")
        if bool(getattr(self.config, "cross_momentum_low_enabled", True)):
            parts.append(f"T{int(getattr(self.config, 'cross_momentum_low_offset', -1)):+d}盘中最低<{float(getattr(self.config, 'cross_momentum_low_threshold', -6)):g}%")
        if bool(getattr(self.config, "cross_momentum_open_enabled", False)):
            parts.append(f"T{int(getattr(self.config, 'cross_momentum_open_offset', 0)):+d}开盘涨幅{float(getattr(self.config, 'cross_momentum_open_threshold', 0)):g}%")
        if bool(getattr(self.config, "cross_momentum_concept_enabled", False)):
            parts.append(f"概念近{int(getattr(self.config, 'cross_momentum_concept_days', 8))}日Top{int(getattr(self.config, 'cross_momentum_concept_top_count', 10))}")
        return "；".join(parts)

    def _condition_formula(self, condition: dict[str, Any]) -> str:
        obj = self._condition_object(condition)
        indicator = self._condition_indicator(condition)
        calculation = self._condition_calculation(condition)
        obj_label = {"stock": "个股", "concept": "概念", "market": "市场"}.get(obj, obj)
        if calculation == "sum":
            start, end = self._condition_offsets(condition)[0], self._condition_offsets(condition)[-1]
            time_label = f"{self._offset_label(start)}至{self._offset_label(end)}"
        elif calculation == "ratio":
            time_label = f"{self._offset_label(self._condition_offset(condition))}÷{self._offset_label(min(0, self._as_int(condition.get('compare_offset', -1), -1)))}"
        elif self._condition_uses_offset(condition):
            time_label = self._offset_label(self._condition_offset(condition))
        else:
            time_label = f"近{self._condition_days(condition)}个交易日"
        indicator_label = {
            "limit_up": "涨停状态",
            "low_pct": "盘中最低跌幅",
            "open_pct": "开盘涨幅",
            "pct_chg": "涨跌幅",
            "turnover_rate": "换手率",
            "turnover_rate_f": "实换手率",
            "amount": "成交额",
            "total_mv": "总市值",
            "circ_mv": "流通市值",
            "momentum": "累计涨幅",
            "stock_strength": "个股强度",
            "concept_strength": "概念强度",
        }.get(indicator, indicator or "未命名指标")
        if indicator == "limit_up":
            value = str(condition.get("value", "limit_up") or "limit_up")
            value = {"limit_up": "涨停", "not_limit_up": "未涨停", "not_up": "未涨停"}.get(value, value)
            return f"{obj_label} {time_label} {indicator_label}={value}"
        if indicator in {"momentum", "stock_strength", "concept_strength"}:
            mode = str(condition.get("rank_mode", "count") or "count").lower()
            value = condition.get("rank_value", condition.get("value", 20))
            suffix = f"前{self._as_int(value, 20)}名" if mode == "count" else f"前{self._as_float(value, 10):g}%"
            scope = str(condition.get("rank_scope_label", condition.get("rank_scope", "当前有效股票池")) or "当前有效股票池")
            return f"{obj_label} {time_label} {indicator_label}（{scope}，{suffix}）"
        operator = {"gt": ">", "ge": ">=", "lt": "<", "le": "<=", "eq": "="}.get(str(condition.get("operator", "lt")), str(condition.get("operator", "lt")))
        threshold = condition.get("threshold", condition.get("value", ""))
        if calculation == "sum":
            return f"{obj_label} {time_label}{indicator_label}累计{operator}{threshold}%"
        if calculation == "ratio":
            return f"{obj_label} {time_label}{indicator_label}比值{operator}{threshold}倍"
        unit = "%" if indicator in {"low_pct", "open_pct", "pct_chg", "turnover_rate", "turnover_rate_f"} else ""
        return f"{obj_label} {time_label} {indicator_label}{operator}{threshold}{unit}"

    @staticmethod
    def _round(value: Any) -> float | str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        return round(number, 2) if math.isfinite(number) else ""

    @classmethod
    def _int(cls, value: Any) -> int:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0
        return int(number) if math.isfinite(number) else 0

    @classmethod
    def _int_or_blank(cls, value: Any) -> int | str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        return int(number) if math.isfinite(number) else ""

    @classmethod
    def _fmt(cls, value: Any) -> str:
        number = cls._round(value)
        return f"{number:g}" if isinstance(number, float) else "--"
