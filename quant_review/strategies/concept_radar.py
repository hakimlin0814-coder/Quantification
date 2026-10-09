from __future__ import annotations

import math
from datetime import datetime, time
from typing import Any

import pandas as pd

from quant_review.strategies.canslim_tao import CanslimTaoStrategy


class ConceptRadarStrategy:
    """THS concept ranking, constituent drilldown, basket-friendly review data."""

    STAR_PREFIXES = ("688", "689")
    CHINEXT_PREFIXES = ("300", "301")
    BJ_PREFIXES = ("4", "8", "920")

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.config = engine.config

    def log(self, message: str) -> None:
        self.engine.log(message)

    def run(self) -> dict[str, Any]:
        requested_date = str(self.config.end_date or "")
        old_lookback = int(self.config.lookback_days)
        self.config.lookback_days = max(old_lookback, 92)
        try:
            self.log("读取交易日历，概念雷达需要覆盖5/10/20/30/60日窗口...")
            trade_dates = self.engine._trade_dates()
        finally:
            self.config.lookback_days = old_lookback

        trade_date = trade_dates[-1]
        self.config.end_date = trade_date
        date_note = self._date_note(requested_date, trade_date)
        snapshot_mode, _ = self._snapshot_mode(trade_dates)
        self.log(f"使用交易日: {trade_date}。{date_note}")
        self.log(f"行情模式: {self._snapshot_state(snapshot_mode, 0)}")

        self.log("读取同花顺概念列表...")
        concepts = self._ths_concepts()
        self.log(f"同花顺概念数量: {len(concepts)}")

        self.log("读取同花顺概念日线，用于计算5/10/20/30/60日排行...")
        ths_daily = self._ths_daily_window(trade_dates, set(concepts["ts_code"].astype(str)))
        if ths_daily.empty:
            raise RuntimeError("同花顺概念日线为空，请检查 TuShare 权限或日期。")

        self.log("读取同花顺概念资金流和领涨股...")
        moneyflow = self._ths_moneyflow(trade_date)

        boards = self._concept_rows(concepts, ths_daily, moneyflow, trade_dates)
        boards = self._apply_keyword(boards)
        if boards.empty:
            return self._empty_payload(trade_date, requested_date, date_note)

        boards = self._sort_boards(boards)
        top_count = max(1, int(getattr(self.config, "concept_top_count", 80)))
        boards = boards.head(top_count).copy()
        board_codes = boards["concept_code"].astype(str).tolist()

        self.log(f"读取排行前 {len(board_codes)} 个概念成分股...")
        members = self._ths_members(boards)

        self.log("读取个股日线和每日指标，计算概念内部个股表现...")
        daily = self.engine._daily_window(trade_dates)
        basic = self.engine._stock_basic()
        # 只有开盘后的实时阶段和午间快照阶段才请求实时行情。
        # 收盘后、开盘前及休市日均使用对应交易日的收盘日线。
        realtime_mode = snapshot_mode in {"live", "lunch"}
        if realtime_mode:
            label = "11:30 午间快照" if snapshot_mode == "lunch" else "盘中实时行情"
            self.log(f"刷新时读取 {label}，合成临时日K。")
            daily = self.engine._apply_intraday_snapshot(daily, basic, trade_date, snapshot_mode)

        if snapshot_mode in {"history", "close"}:
            self.engine._last_realtime_snapshot = {
                "count": 0,
                "quote_at": self._close_snapshot_at(trade_date),
                "source": "日线收盘",
            }

        daily_basic = self.engine._daily_basic(trade_date)
        if realtime_mode and self.engine._daily_basic_is_empty(daily_basic) and len(trade_dates) >= 2:
            prev_date = trade_dates[-2]
            self.log(f"盘中模式：当日 daily_basic 未更新，回退到 {prev_date}。")
            daily_basic = self.engine._daily_basic(prev_date)

        if realtime_mode and self.engine._last_realtime_snapshot.get("count", 0):
            boards = self._refresh_board_realtime(boards, members, daily, ths_daily, trade_dates, trade_date)
        stock_features = self._stock_features(daily, basic, daily_basic, trade_dates)
        stocks = self._stock_rows(members, stock_features, boards)
        if self._canslim_enabled():
            stocks = self._attach_canslim(stocks)
        boards = self._attach_breadth(boards, stocks)
        boards = self._score_boards(boards)
        boards = self._sort_boards(boards)
        stocks = self._refresh_stock_concept_state(stocks, boards)

        recommend_boards = self._recommended_boards(boards)
        recommend_stocks = self._recommended_stocks(stocks, boards)
        market = self._market_payload(
            trade_date=trade_date,
            requested_date=requested_date,
            date_note=date_note,
            boards=boards,
            stocks=stocks,
            recommend_boards=recommend_boards,
            recommend_stocks=recommend_stocks,
            snapshot_mode=snapshot_mode,
            realtime_snapshot=self.engine._last_realtime_snapshot,
        )
        self.log(f"概念雷达完成：概念 {len(boards)} 个，成分股明细 {len(stocks)} 行。")
        return {
            "ok": True,
            "trade_date": trade_date,
            "requested_date": requested_date,
            "date_note": date_note,
            "market": market,
            "boards": boards.fillna("").to_dict("records"),
            "stocks": stocks.fillna("").to_dict("records"),
            "recommend_boards": recommend_boards,
            "recommend_stocks": recommend_stocks,
            "logs": self.engine.logs,
        }

    def _empty_payload(self, trade_date: str, requested_date: str, date_note: str) -> dict[str, Any]:
        return {
            "ok": True,
            "trade_date": trade_date,
            "requested_date": requested_date,
            "date_note": date_note,
            "market": {
                "trade_date": trade_date,
                "requested_date": requested_date,
                "date_note": date_note,
                "note": "没有匹配到概念，请检查关键词或数据权限。",
            },
            "boards": [],
            "stocks": [],
            "recommend_boards": [],
            "recommend_stocks": [],
            "logs": self.engine.logs,
        }

    @staticmethod
    def _date_note(requested_date: str, trade_date: str) -> str:
        if requested_date and requested_date != trade_date:
            return f"输入日期 {requested_date} 非最终可用交易日，已自动使用上一开市日 {trade_date}。"
        return "已使用输入日期对应的交易日数据。"

    @staticmethod
    def _snapshot_mode(
        trade_dates: list[str],
        now: datetime | None = None,
    ) -> tuple[str, str]:
        now = now or datetime.now()
        today = now.strftime("%Y%m%d")
        available = [item for item in trade_dates if item <= today]
        if not available or available[-1] != today:
            return "history", available[-1] if available else trade_dates[-1]
        current = now.time()
        if current < time(9, 30):
            return "history", available[-2] if len(available) >= 2 else available[-1]
        if time(11, 30) <= current < time(13, 0):
            return "lunch", today
        if current < time(15, 0):
            return "live", today
        return "close", today

    @staticmethod
    def _close_snapshot_at(trade_date: str) -> str:
        return f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 15:00:00"

    @staticmethod
    def _snapshot_state(mode: str, realtime_count: int, source: str = "") -> str:
        if mode == "live":
            return "盘中实时行情" if realtime_count else "盘中实时行情缺失，使用最近收盘行情"
        if mode == "lunch":
            return "11:30午间快照" if realtime_count else "11:30快照缺失，使用最近收盘行情"
        if mode == "close":
            return "当日15:00收盘行情" if source == "日线收盘" or not realtime_count else "当日15:00最终行情"
        return "最近交易日收盘行情"

    def _refresh_board_realtime(
        self,
        boards: pd.DataFrame,
        members: pd.DataFrame,
        daily: pd.DataFrame,
        ths_daily: pd.DataFrame,
        trade_dates: list[str],
        trade_date: str,
    ) -> pd.DataFrame:
        """用成分股实时涨跌幅刷新概念当前快照和各周期涨幅。"""
        if boards.empty or members.empty:
            return boards
        current = daily[daily["trade_date"].astype(str) == str(trade_date)].copy()
        if current.empty:
            return boards
        for col in ["ts_code", "close", "pct_chg"]:
            if col not in current.columns:
                return boards
        current["ts_code"] = current["ts_code"].astype(str)
        current["close"] = pd.to_numeric(current["close"], errors="coerce")
        current["pct_chg"] = pd.to_numeric(current["pct_chg"], errors="coerce").fillna(0)
        live = members[["concept_code", "ts_code"]].copy()
        live["ts_code"] = live["ts_code"].astype(str)
        live = live.merge(current[["ts_code", "close", "pct_chg"]], on="ts_code", how="inner")
        live = live[live["close"] > 0]
        if live.empty:
            return boards
        pct_by_board = live.groupby("concept_code")["pct_chg"].mean()
        count_by_board = live.groupby("concept_code")["ts_code"].count()
        history = ths_daily.pivot_table(index="ts_code", columns="trade_date", values="close", aggfunc="last")
        result = boards.copy()
        windows = [2, 5, 10, 20, 30, 60]
        for idx, row in result.iterrows():
            code = str(row.get("concept_code", ""))
            if code not in pct_by_board.index:
                continue
            live_pct = float(pct_by_board.loc[code])
            result.at[idx, "pct_chg"] = live_pct
            result.at[idx, "realtime_member_count"] = int(count_by_board.loc[code])
            series = history.loc[code].reindex(trade_dates) if code in history.index else pd.Series(index=trade_dates, dtype=float)
            series = pd.to_numeric(series, errors="coerce").astype(float)
            previous = series.iloc[-2] if len(series) >= 2 else float("nan")
            live_close = previous * (1 + live_pct / 100) if pd.notna(previous) else float(row.get("close", 0) or 0)
            if live_close <= 0:
                continue
            series.iloc[-1] = live_close
            result.at[idx, "close"] = live_close
            for window in windows:
                base_pos = len(trade_dates) - 1 - window
                base = series.iloc[base_pos] if base_pos >= 0 else float("nan")
                result.at[idx, f"ret{window}"] = (live_close / base - 1) * 100 if pd.notna(base) and base else float("nan")
            for window in [5, 10, 20]:
                ma = series.iloc[-min(window, len(series)):].mean()
                result.at[idx, f"distance_ma{window}"] = (live_close / ma - 1) * 100 if pd.notna(ma) and ma else float("nan")
        for window in windows:
            result[f"rps{window}"] = pd.to_numeric(result[f"ret{window}"], errors="coerce").rank(pct=True) * 100
        return result

    def _ths_concepts(self) -> pd.DataFrame:
        try:
            data = self.engine._cached(
                "ths_index_concept_A",
                lambda: self.engine.pro.ths_index(exchange="A", type="N"),
            )
        except Exception as exc:
            self.log(f"ths_index(exchange=A,type=N) 失败，尝试只按 type=N 读取: {exc}")
            data = self.engine._cached(
                "ths_index_concept_N",
                lambda: self.engine.pro.ths_index(type="N"),
            )
        if data.empty or "ts_code" not in data.columns:
            raise RuntimeError("同花顺概念列表为空，请检查 ths_index 权限。")
        result = data.copy()
        result["ts_code"] = result["ts_code"].astype(str)
        if "name" not in result.columns:
            result["name"] = result["ts_code"]
        result["name"] = result["name"].fillna("").astype(str)
        if "count" not in result.columns:
            result["count"] = 0
        result["count"] = pd.to_numeric(result["count"], errors="coerce").fillna(0).astype(int)
        for col in ["exchange", "list_date", "type"]:
            if col not in result.columns:
                result[col] = ""
        return result[["ts_code", "name", "count", "exchange", "list_date", "type"]].drop_duplicates("ts_code")

    def _ths_daily_window(self, trade_dates: list[str], concept_codes: set[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for td in trade_dates:
            try:
                frame = self.engine._cached(
                    f"ths_daily_{td}",
                    lambda td=td: self.engine.pro.ths_daily(trade_date=td),
                )
            except Exception as exc:
                self.log(f"{td} 同花顺概念日线读取失败，跳过: {exc}")
                continue
            if frame.empty or "ts_code" not in frame.columns:
                continue
            data = frame.copy()
            data["ts_code"] = data["ts_code"].astype(str)
            data = data[data["ts_code"].isin(concept_codes)].copy()
            if data.empty:
                continue
            if "trade_date" not in data.columns:
                data["trade_date"] = td
            data["trade_date"] = data["trade_date"].astype(str)
            for col in ["open", "high", "low", "close", "pre_close", "pct_change", "pct_chg", "vol", "turnover_rate"]:
                if col not in data.columns:
                    data[col] = 0
                data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
            frames.append(data)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(["ts_code", "trade_date"])

    def _ths_moneyflow(self, trade_date: str) -> pd.DataFrame:
        try:
            data = self.engine._cached(
                f"moneyflow_cnt_ths_{trade_date}",
                lambda: self.engine.pro.moneyflow_cnt_ths(trade_date=trade_date),
            )
        except Exception as exc:
            self.log(f"同花顺概念资金流读取失败，资金字段置空: {exc}")
            return pd.DataFrame()
        if data.empty:
            return data
        result = data.copy()
        if "ts_code" in result.columns:
            result["ts_code"] = result["ts_code"].astype(str)
        if "name" in result.columns:
            result["name"] = result["name"].astype(str)
        return result

    def _concept_rows(
        self,
        concepts: pd.DataFrame,
        daily: pd.DataFrame,
        moneyflow: pd.DataFrame,
        trade_dates: list[str],
    ) -> pd.DataFrame:
        close = daily.pivot_table(index="ts_code", columns="trade_date", values="close", aggfunc="last").reindex(columns=trade_dates)
        pct_col = "pct_change" if "pct_change" in daily.columns else "pct_chg"
        pct = daily.pivot_table(index="ts_code", columns="trade_date", values=pct_col, aggfunc="last").reindex(columns=trade_dates)
        turnover = daily.pivot_table(index="ts_code", columns="trade_date", values="turnover_rate", aggfunc="last").reindex(columns=trade_dates)
        latest = trade_dates[-1]
        rows = pd.DataFrame(index=close.index)
        rows["concept_code"] = rows.index.astype(str)
        rows["close"] = pd.to_numeric(close[latest], errors="coerce")
        rows["pct_chg"] = pd.to_numeric(pct[latest], errors="coerce").fillna(0)
        rows["turnover_rate"] = pd.to_numeric(turnover[latest], errors="coerce").fillna(0)
        for window in [2, 5, 10, 20, 30, 60]:
            base = close.shift(window, axis=1)[latest] if latest in close.columns else pd.Series(index=close.index)
            rows[f"ret{window}"] = (rows["close"] / pd.to_numeric(base, errors="coerce") - 1) * 100
            rows[f"rps{window}"] = rows[f"ret{window}"].rank(pct=True) * 100
        for window in [5, 10, 20]:
            take = min(window, len(trade_dates))
            ma = close.iloc[:, -take:].mean(axis=1)
            rows[f"distance_ma{window}"] = (rows["close"] / ma - 1) * 100
        rows = rows.reset_index(drop=True)
        meta = concepts.rename(
            columns={
                "ts_code": "concept_code",
                "name": "concept_name",
                "count": "declared_member_count",
                "list_date": "concept_list_date",
            }
        )
        rows = rows.merge(
            meta[["concept_code", "concept_name", "declared_member_count", "exchange", "concept_list_date"]],
            on="concept_code",
            how="left",
        )
        flow = self._normalize_moneyflow(moneyflow)
        if not flow.empty:
            rows = rows.merge(flow, on="concept_code", how="left")
        for col in ["lead_stock", "lead_stock_pct", "net_amount_yi"]:
            if col not in rows.columns:
                rows[col] = "" if col == "lead_stock" else 0
        rows["lead_stock"] = rows["lead_stock"].fillna("").astype(str)
        rows["net_amount_yi"] = pd.to_numeric(rows["net_amount_yi"], errors="coerce").fillna(0)
        rows["member_count"] = pd.to_numeric(rows["declared_member_count"], errors="coerce").fillna(0).astype(int)
        return rows

    def _normalize_moneyflow(self, moneyflow: pd.DataFrame) -> pd.DataFrame:
        if moneyflow.empty:
            return pd.DataFrame()
        data = moneyflow.copy()
        concept_code = self._first_existing(data, ["ts_code", "industry_index", "index_code"])
        if not concept_code:
            return pd.DataFrame()
        data["concept_code"] = data[concept_code].astype(str)
        lead_col = self._first_existing(data, ["lead_stock", "leader", "leading_stock"])
        lead_pct_col = self._first_existing(data, ["lead_stock_pct", "leader_pct", "pct_change_stock"])
        net_col = self._first_existing(data, ["net_amount", "net_buy_amount", "net_mf_amount"])
        data["lead_stock"] = data[lead_col].fillna("").astype(str) if lead_col else ""
        data["lead_stock_pct"] = pd.to_numeric(data[lead_pct_col], errors="coerce").fillna(0) if lead_pct_col else 0
        if net_col:
            net = pd.to_numeric(data[net_col], errors="coerce").fillna(0)
            data["net_amount_yi"] = net / 10000
        else:
            data["net_amount_yi"] = 0.0
        return data[["concept_code", "lead_stock", "lead_stock_pct", "net_amount_yi"]].drop_duplicates("concept_code")

    @staticmethod
    def _first_existing(data: pd.DataFrame, columns: list[str]) -> str:
        for col in columns:
            if col in data.columns:
                return col
        return ""

    def _apply_keyword(self, boards: pd.DataFrame) -> pd.DataFrame:
        keyword = str(getattr(self.config, "concept_keyword", "") or "").strip()
        if not keyword:
            return boards
        words = [item.strip() for item in keyword.replace("，", ",").split(",") if item.strip()]
        if not words:
            return boards
        pattern = "|".join(words)
        return boards[boards["concept_name"].astype(str).str.contains(pattern, case=False, regex=True, na=False)].copy()

    def _sort_boards(self, boards: pd.DataFrame) -> pd.DataFrame:
        field = str(getattr(self.config, "concept_sort_field", "") or "").strip()
        if not field:
            period = int(getattr(self.config, "concept_rank_period", 10))
            field = f"ret{period}"
        if field not in boards.columns:
            field = "ret10"
        result = boards.copy()
        sort_key = f"__sort_{field}"
        result[sort_key] = pd.to_numeric(result[field], errors="coerce")
        if result[sort_key].notna().sum() == 0 and field != "ret10" and "ret10" in result.columns:
            sort_key = "__sort_ret10"
            result[sort_key] = pd.to_numeric(result["ret10"], errors="coerce")
        result = (
            result.sort_values([sort_key, "ret20", "net_amount_yi"], ascending=False, na_position="last")
            .drop(columns=[col for col in result.columns if col.startswith("__sort_")])
            .reset_index(drop=True)
        )
        result["rank"] = range(1, len(result) + 1)
        return result

    def _ths_members(self, boards: pd.DataFrame) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for _, board in boards.iterrows():
            code = str(board.get("concept_code", "")).strip()
            name = str(board.get("concept_name", "")).strip()
            if not code:
                continue
            try:
                raw = self.engine._cached(
                    f"ths_member_{code}",
                    lambda code=code: self.engine.pro.ths_member(ts_code=code),
                )
            except Exception as exc:
                self.log(f"{name} 成分股读取失败，跳过: {exc}")
                continue
            if raw.empty:
                continue
            data = raw.copy()
            code_col = self._first_existing(data, ["con_code", "code", "ts_code"])
            if not code_col:
                continue
            stock_name_col = self._first_existing(data, ["con_name", "stock_name", "name"])
            data["ts_code"] = data[code_col].astype(str)
            data = data[data["ts_code"] != code].copy()
            data["concept_code"] = code
            data["concept_name"] = name
            data["member_name"] = data[stock_name_col].fillna("").astype(str) if stock_name_col else ""
            frames.append(data[["concept_code", "concept_name", "ts_code", "member_name"]])
        if not frames:
            return pd.DataFrame(columns=["concept_code", "concept_name", "ts_code", "member_name"])
        return pd.concat(frames, ignore_index=True).drop_duplicates(["concept_code", "ts_code"])

    def _stock_features(
        self,
        daily: pd.DataFrame,
        basic: pd.DataFrame,
        daily_basic: pd.DataFrame,
        trade_dates: list[str],
    ) -> pd.DataFrame:
        data = daily.copy()
        data["ts_code"] = data["ts_code"].astype(str)
        data["trade_date"] = data["trade_date"].astype(str)
        for col in ["open", "high", "low", "close", "pct_chg", "amount"]:
            if col not in data.columns:
                data[col] = 0
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
        close = data.pivot_table(index="ts_code", columns="trade_date", values="close", aggfunc="last").reindex(columns=trade_dates)
        high = data.pivot_table(index="ts_code", columns="trade_date", values="high", aggfunc="last").reindex(columns=trade_dates)
        amount = data.pivot_table(index="ts_code", columns="trade_date", values="amount", aggfunc="last").reindex(columns=trade_dates)
        pct = data.pivot_table(index="ts_code", columns="trade_date", values="pct_chg", aggfunc="last").reindex(columns=trade_dates)
        latest = trade_dates[-1]
        features = pd.DataFrame(index=close.index)
        features["ts_code"] = features.index.astype(str)
        features["close"] = pd.to_numeric(close[latest], errors="coerce")
        features["pct_chg"] = pd.to_numeric(pct[latest], errors="coerce").fillna(0)
        features["amount_yi"] = pd.to_numeric(amount[latest], errors="coerce").fillna(0) / 100000
        for window in [2, 5, 10, 20, 30, 60]:
            base = close.shift(window, axis=1)[latest]
            features[f"ret{window}"] = (features["close"] / pd.to_numeric(base, errors="coerce") - 1) * 100
            features[f"rps{window}"] = features[f"ret{window}"].rank(pct=True) * 100
            take = min(window, len(trade_dates))
            ma = close.iloc[:, -take:].mean(axis=1)
            features[f"ma{window}"] = ma
        for window in [5, 10, 20]:
            features[f"distance_ma{window}"] = (features["close"] / features[f"ma{window}"] - 1) * 100
        features["avg_amount5"] = amount.iloc[:, -min(5, len(trade_dates)) :].mean(axis=1)
        features["avg_amount20"] = amount.iloc[:, -min(20, len(trade_dates)) :].mean(axis=1)
        features["amount_ratio"] = features["avg_amount5"] / features["avg_amount20"].replace(0, pd.NA)
        features["high20"] = high.iloc[:, -min(20, len(trade_dates)) :].max(axis=1)
        features["high60"] = high.iloc[:, -min(60, len(trade_dates)) :].max(axis=1)
        features["breakout20"] = features["close"] >= features["high20"] * 0.995
        features["high_position_60"] = features["close"] / features["high60"].replace(0, pd.NA)

        basic_use = basic[[col for col in ["ts_code", "symbol", "name", "industry", "market", "stock_type", "list_date"] if col in basic.columns]].copy()
        basic_use["ts_code"] = basic_use["ts_code"].astype(str)
        features = features.reset_index(drop=True).merge(basic_use, on="ts_code", how="left")
        features["symbol"] = features["symbol"].fillna(features["ts_code"].str.slice(0, 6)).astype(str).str.zfill(6)
        features["name"] = features["name"].fillna("")
        features["industry"] = features["industry"].fillna("未分类").replace("", "未分类")
        features["stock_type"] = features["stock_type"].fillna("")

        db = daily_basic.copy()
        if "ts_code" not in db.columns:
            db["ts_code"] = pd.Series(dtype=str)
        db["ts_code"] = db["ts_code"].astype(str)
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
        return features

    def _stock_rows(self, members: pd.DataFrame, features: pd.DataFrame, boards: pd.DataFrame) -> pd.DataFrame:
        if members.empty or features.empty:
            return pd.DataFrame()
        data = members.merge(features, on="ts_code", how="left")
        data = data[data["close"].fillna(0) > 0].copy()
        if bool(getattr(self.config, "concept_exclude_st", False)):
            data = data[~data["name"].astype(str).str.upper().str.contains("ST", regex=False)].copy()
        if bool(getattr(self.config, "concept_exclude_star", False)):
            data = data[~data["symbol"].astype(str).str.startswith(self.STAR_PREFIXES)].copy()
        if bool(getattr(self.config, "concept_exclude_chinext", False)):
            data = data[~data["symbol"].astype(str).str.startswith(self.CHINEXT_PREFIXES)].copy()
        if bool(getattr(self.config, "concept_exclude_bj", False)):
            symbol = data["symbol"].astype(str)
            ts_code = data["ts_code"].astype(str)
            market = data.get("market", pd.Series("", index=data.index)).astype(str)
            bj_mask = ts_code.str.endswith(".BJ") | symbol.str.startswith(self.BJ_PREFIXES) | market.str.contains("北", na=False)
            data = data[~bj_mask].copy()
        if data.empty:
            return data
        board_state = boards[["concept_code", "state"]] if "state" in boards.columns else pd.DataFrame(columns=["concept_code", "state"])
        data = data.merge(board_state.rename(columns={"state": "concept_state"}), on="concept_code", how="left")
        data["concept_stock_rank"] = data.groupby("concept_code")["rps10"].rank(method="min", ascending=False)
        data["role"] = data.apply(self._stock_role, axis=1)
        data["risk_flags"] = data.apply(self._stock_risks, axis=1)
        data["stock_observe_score"] = data.apply(self._stock_observe_score, axis=1)
        data["source_concepts"] = data["concept_name"]
        limit = max(10, int(getattr(self.config, "concept_stock_limit_per_board", 120)))
        data = data.sort_values(["concept_code", "stock_observe_score", "rps10"], ascending=[True, False, False])
        data = data.groupby("concept_code", group_keys=False).head(limit).copy()
        return pd.DataFrame([self._stock_dict(row) for _, row in data.iterrows()])

    def _stock_role(self, row: pd.Series) -> str:
        rank = self._num(row.get("concept_stock_rank"), 999)
        rps10 = self._num(row.get("rps10"))
        amount = self._num(row.get("amount_yi"))
        cap = self._num(row.get("market_cap_yi"))
        if rank <= 3 and rps10 >= 85:
            return "龙头/前排"
        if cap >= 300 and amount >= 5 and rps10 >= 70:
            return "中军确认"
        if self._num(row.get("rps10")) - self._num(row.get("rps20")) >= 12:
            return "弹性补涨"
        if rps10 < 55:
            return "跟风偏弱"
        return "跟随观察"

    def _stock_risks(self, row: pd.Series) -> str:
        flags: list[str] = []
        if "ST" in str(row.get("name", "")).upper():
            flags.append("ST")
        if str(row.get("symbol", "")).startswith(self.STAR_PREFIXES):
            flags.append("科创板")
        if str(row.get("symbol", "")).startswith(self.CHINEXT_PREFIXES):
            flags.append("创业板")
        if self._num(row.get("distance_ma20")) > 15:
            flags.append("偏离MA20")
        if self._num(row.get("ret10")) > 30:
            flags.append("10日过热")
        if self._num(row.get("turnover_rate_f")) > 18:
            flags.append("换手偏高")
        if self._num(row.get("amount_ratio")) > 3:
            flags.append("量能过猛")
        return "；".join(flags) if flags else "正常"

    def _stock_observe_score(self, row: pd.Series) -> float:
        score = 0.0
        score += self._clip((self._num(row.get("rps10")) - 55) / 40) * 30
        score += self._clip((self._num(row.get("rps20")) - 50) / 45) * 20
        score += self._clip((self._num(row.get("rps5")) - 50) / 45) * 15
        score += self._clip((self._num(row.get("amount_ratio")) - 0.8) / 1.2) * 10
        score += (1 if bool(row.get("breakout20")) else 0) * 10
        score += self._clip((15 - abs(self._num(row.get("distance_ma20")))) / 15) * 10
        score += self._clip((6 - self._num(row.get("concept_stock_rank"), 99)) / 5) * 5
        penalty = max(self._num(row.get("distance_ma20")) - 18, 0) * 0.6
        penalty += max(self._num(row.get("ret10")) - 35, 0) * 0.35
        return max(0.0, min(100.0, score - penalty))

    def _stock_dict(self, row: pd.Series) -> dict[str, Any]:
        return {
            "concept_code": str(row.get("concept_code", "")),
            "concept_name": str(row.get("concept_name", "")),
            "concept_state": str(row.get("concept_state", "")),
            "ts_code": str(row.get("ts_code", "")),
            "code": str(row.get("symbol", str(row.get("ts_code", ""))[:6])).zfill(6),
            "name": str(row.get("name", "")),
            "industry": str(row.get("industry", "")),
            "stock_type": str(row.get("stock_type", "")),
            "source_concepts": str(row.get("source_concepts", "")),
            "role": str(row.get("role", "")),
            "stock_observe_score": round(self._num(row.get("stock_observe_score")), 2),
            "concept_stock_rank": int(self._num(row.get("concept_stock_rank"), 0)),
            "close": round(self._num(row.get("close")), 2),
            "pct_chg": round(self._num(row.get("pct_chg")), 2),
            "ret2": round(self._num(row.get("ret2")), 2),
            "ret5": round(self._num(row.get("ret5")), 2),
            "ret10": round(self._num(row.get("ret10")), 2),
            "ret20": round(self._num(row.get("ret20")), 2),
            "ret30": round(self._num(row.get("ret30")), 2),
            "ret60": round(self._num(row.get("ret60")), 2),
            "rps5": round(self._num(row.get("rps5")), 2),
            "rps10": round(self._num(row.get("rps10")), 2),
            "rps20": round(self._num(row.get("rps20")), 2),
            "rps30": round(self._num(row.get("rps30")), 2),
            "rps60": round(self._num(row.get("rps60")), 2),
            "amount_yi": round(self._num(row.get("amount_yi")), 2),
            "amount_ratio": round(self._num(row.get("amount_ratio")), 2),
            "turnover_rate_f": round(self._num(row.get("turnover_rate_f")), 2),
            "volume_ratio": round(self._num(row.get("volume_ratio")), 2),
            "market_cap_yi": round(self._num(row.get("market_cap_yi")), 2),
            "circ_mv_yi": round(self._num(row.get("circ_mv_yi")), 2),
            "distance_ma5": round(self._num(row.get("distance_ma5")), 2),
            "distance_ma10": round(self._num(row.get("distance_ma10")), 2),
            "distance_ma20": round(self._num(row.get("distance_ma20")), 2),
            "breakout20": "是" if bool(row.get("breakout20", False)) else "否",
            "high_position_60": round(self._num(row.get("high_position_60")), 3),
            "risk_flags": str(row.get("risk_flags", "")),
            "reason": self._stock_reason(row),
        }

    def _stock_reason(self, row: pd.Series) -> str:
        return (
            f"{row.get('role', '')}；概念内排名{int(self._num(row.get('concept_stock_rank'), 0))}；"
            f"RPS5/10/20={self._num(row.get('rps5')):.0f}/{self._num(row.get('rps10')):.0f}/{self._num(row.get('rps20')):.0f}；"
            f"距MA20 {self._num(row.get('distance_ma20')):.1f}%；风险：{row.get('risk_flags', '')}"
        )

    def _attach_breadth(self, boards: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
        result = boards.copy()
        if stocks.empty:
            for col in ["actual_member_count", "up_count", "down_count", "strong_stock_count", "above_ma20_ratio", "top_stock", "top_stock_pct", "avg_canslim_score", "canslim_pass_count"]:
                result[col] = 0 if col not in {"top_stock"} else ""
            result["state"] = result.apply(self._concept_state, axis=1)
            result["risk_flags"] = result.apply(self._concept_risks, axis=1)
            return result
        agg: dict[str, Any] = {
            "actual_member_count": ("ts_code", "nunique"),
            "up_count": ("pct_chg", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            "down_count": ("pct_chg", lambda s: int((pd.to_numeric(s, errors="coerce") < 0).sum())),
            "strong_stock_count": ("rps10", lambda s: int((pd.to_numeric(s, errors="coerce") >= 80).sum())),
            "above_ma20_ratio": ("distance_ma20", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean() * 100)),
        }
        if "canslim_score" in stocks.columns:
            agg["avg_canslim_score"] = (
                "canslim_score",
                lambda s: pd.to_numeric(s, errors="coerce").mean(),
            )
            agg["canslim_pass_count"] = ("canslim_pass", lambda s: int((s.astype(str) == "是").sum()))
        grouped = stocks.groupby("concept_code", as_index=False).agg(**agg)
        if "avg_canslim_score" not in grouped.columns:
            grouped["avg_canslim_score"] = 0
        if "canslim_pass_count" not in grouped.columns:
            grouped["canslim_pass_count"] = 0
        top = stocks.sort_values(["concept_code", "pct_chg"], ascending=[True, False]).groupby("concept_code", as_index=False).head(1)
        top = top[["concept_code", "name", "pct_chg"]].rename(columns={"name": "top_stock", "pct_chg": "top_stock_pct"})
        result = result.merge(grouped, on="concept_code", how="left").merge(top, on="concept_code", how="left")
        result["actual_member_count"] = pd.to_numeric(result["actual_member_count"], errors="coerce").fillna(0).astype(int)
        result["member_count"] = result["actual_member_count"].where(result["actual_member_count"] > 0, result["member_count"])
        for col in ["up_count", "down_count", "strong_stock_count", "above_ma20_ratio", "avg_canslim_score", "canslim_pass_count", "top_stock_pct"]:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
        result["top_stock"] = result["top_stock"].fillna("")
        result["state"] = result.apply(self._concept_state, axis=1)
        result["risk_flags"] = result.apply(self._concept_risks, axis=1)
        return result

    def _score_boards(self, boards: pd.DataFrame) -> pd.DataFrame:
        result = boards.copy()
        weights = {
            "trend": float(getattr(self.config, "concept_weight_trend", 0)),
            "stability": float(getattr(self.config, "concept_weight_stability", 0)),
            "breadth": float(getattr(self.config, "concept_weight_breadth", 0)),
            "money": float(getattr(self.config, "concept_weight_money", 0)),
            "leader": float(getattr(self.config, "concept_weight_leader", 0)),
            "persistence": float(getattr(self.config, "concept_weight_persistence", 0)),
            "canslim": float(getattr(self.config, "concept_weight_canslim", 0)),
            "risk": float(getattr(self.config, "concept_weight_risk", 0)),
        }
        result["trend_score"] = (
            result["rps5"].fillna(0) * 0.20
            + result["rps10"].fillna(0) * 0.30
            + result["rps20"].fillna(0) * 0.25
            + result["rps30"].fillna(0) * 0.15
            + result["rps60"].fillna(0) * 0.10
        )
        result["stability_score"] = (
            self._series_clip((result["rps10"].fillna(0) - 55) / 40) * 35
            + self._series_clip((result["rps20"].fillna(0) - 50) / 45) * 30
            + self._series_clip((result["distance_ma20"].fillna(0) + 4) / 14) * 20
            + self._series_clip((result["distance_ma10"].fillna(0) + 3) / 10) * 15
        )
        result["breadth_score"] = (
            self._series_clip(result.get("above_ma20_ratio", 0).fillna(0) / 80) * 55
            + self._series_clip(result.get("strong_stock_count", 0).fillna(0) / 8) * 35
            + self._series_clip(result.get("up_count", 0).fillna(0) / result.get("member_count", 1).replace(0, 1)) * 10
        )
        result["money_score"] = result["net_amount_yi"].rank(pct=True).fillna(0) * 100
        result["leader_score"] = (
            self._series_clip(result.get("top_stock_pct", 0).fillna(0) / 10) * 45
            + self._series_clip(result.get("strong_stock_count", 0).fillna(0) / 6) * 35
            + self._series_clip(result.get("lead_stock_pct", 0).fillna(0) / 10) * 20
        )
        result["persistence_score"] = (
            result["rps10"].fillna(0) * 0.35
            + result["rps20"].fillna(0) * 0.35
            + result["rps30"].fillna(0) * 0.20
            + result["rps60"].fillna(0) * 0.10
        )
        result["canslim_score"] = pd.to_numeric(result.get("avg_canslim_score", 0), errors="coerce").fillna(0)
        result["risk_penalty"] = result.apply(self._concept_risk_penalty, axis=1)
        total_weight = sum(max(v, 0.0) for k, v in weights.items() if k != "risk")
        score_enabled = bool(getattr(self.config, "concept_score_enabled", False))
        if score_enabled and total_weight > 0:
            score = (
                result["trend_score"] * max(weights["trend"], 0)
                + result["stability_score"] * max(weights["stability"], 0)
                + result["breadth_score"] * max(weights["breadth"], 0)
                + result["money_score"] * max(weights["money"], 0)
                + result["leader_score"] * max(weights["leader"], 0)
                + result["persistence_score"] * max(weights["persistence"], 0)
                + result["canslim_score"] * max(weights["canslim"], 0)
            ) / total_weight
            score -= result["risk_penalty"] * max(weights["risk"], 0) / max(total_weight, 1)
            result["concept_score"] = score.clip(0, 100).round(2)
        else:
            result["concept_score"] = ""
        result["observe_score"] = (
            result["trend_score"] * 0.32
            + result["stability_score"] * 0.22
            + result["breadth_score"] * 0.22
            + result["money_score"] * 0.08
            + result["leader_score"] * 0.10
            + result["persistence_score"] * 0.06
            - result["risk_penalty"]
        ).clip(0, 100)
        result["recommend_reason"] = result.apply(self._concept_reason, axis=1)
        return result

    def _concept_state(self, row: pd.Series) -> str:
        if self._num(row.get("rps5")) < 45 and self._num(row.get("rps10")) < 50 and self._num(row.get("distance_ma20")) < -2:
            return "退潮规避"
        if self._num(row.get("ret5")) >= float(getattr(self.config, "concept_overheat_ret5", 15)) or self._num(row.get("distance_ma20")) >= float(getattr(self.config, "concept_overheat_distance_ma20", 15)):
            return "加速过热"
        if self._num(row.get("rps10")) >= 80 and self._num(row.get("rps20")) >= 70 and self._num(row.get("above_ma20_ratio")) >= 55:
            return "短期站稳"
        if self._num(row.get("rps5")) >= 85 and self._num(row.get("rps20")) < 60:
            return "突破待确认"
        if self._num(row.get("rps20")) >= 75 and self._num(row.get("rps5")) < 55:
            return "回踩观察"
        if self._num(row.get("strong_stock_count")) <= 1 and self._num(row.get("rps5")) >= 75:
            return "龙头独舞"
        return "普通观察"

    def _concept_risks(self, row: pd.Series) -> str:
        flags: list[str] = []
        if self._num(row.get("member_count")) < int(getattr(self.config, "concept_member_warn_threshold", 8)):
            flags.append("成分少")
        if self._num(row.get("ret5")) >= float(getattr(self.config, "concept_overheat_ret5", 15)):
            flags.append("5日过热")
        if self._num(row.get("ret10")) >= float(getattr(self.config, "concept_overheat_ret10", 28)):
            flags.append("10日过热")
        if self._num(row.get("distance_ma20")) >= float(getattr(self.config, "concept_overheat_distance_ma20", 15)):
            flags.append("偏离MA20")
        if self._num(row.get("above_ma20_ratio")) and self._num(row.get("above_ma20_ratio")) < float(getattr(self.config, "concept_low_breadth_ma20", 45)):
            flags.append("广度不足")
        if self._num(row.get("net_amount_yi")) < 0:
            flags.append("资金流出")
        return "；".join(flags) if flags else "正常"

    def _concept_risk_penalty(self, row: pd.Series) -> float:
        penalty = 0.0
        penalty += max(self._num(row.get("ret5")) - float(getattr(self.config, "concept_overheat_ret5", 15)), 0) * 0.8
        penalty += max(self._num(row.get("ret10")) - float(getattr(self.config, "concept_overheat_ret10", 28)), 0) * 0.5
        penalty += max(self._num(row.get("distance_ma20")) - float(getattr(self.config, "concept_overheat_distance_ma20", 15)), 0) * 0.7
        if self._num(row.get("net_amount_yi")) < 0:
            penalty += min(abs(self._num(row.get("net_amount_yi"))) * 0.4, 8)
        if self._num(row.get("above_ma20_ratio")) < 35 and self._num(row.get("above_ma20_ratio")) > 0:
            penalty += 8
        return min(35.0, penalty)

    def _concept_reason(self, row: pd.Series) -> str:
        pieces = [
            f"状态{row.get('state', '')}",
            f"5/10/20日涨幅={self._num(row.get('ret5')):.1f}%/{self._num(row.get('ret10')):.1f}%/{self._num(row.get('ret20')):.1f}%",
            f"强势股{int(self._num(row.get('strong_stock_count')))}只",
            f"站上MA20 {self._num(row.get('above_ma20_ratio')):.0f}%",
            f"资金{self._num(row.get('net_amount_yi')):.2f}亿",
        ]
        risk = str(row.get("risk_flags", ""))
        if risk and risk != "正常":
            pieces.append(f"风险：{risk}")
        return "；".join(pieces)

    def _refresh_stock_concept_state(self, stocks: pd.DataFrame, boards: pd.DataFrame) -> pd.DataFrame:
        if stocks.empty:
            return stocks
        states = boards[["concept_code", "state", "concept_score", "observe_score"]].rename(
            columns={"state": "concept_state", "concept_score": "concept_score", "observe_score": "concept_observe_score"}
        )
        keep = [col for col in stocks.columns if col not in {"concept_state", "concept_score", "concept_observe_score"}]
        result = stocks[keep].merge(states, on="concept_code", how="left")
        result["reason"] = result.apply(
            lambda row: f"{row.get('role', '')}；来源概念{row.get('concept_name', '')}({row.get('concept_state', '')})；"
            f"RPS10/20={self._num(row.get('rps10')):.0f}/{self._num(row.get('rps20')):.0f}；"
            f"距MA20 {self._num(row.get('distance_ma20')):.1f}%；风险：{row.get('risk_flags', '')}",
            axis=1,
        )
        return result

    def _recommended_boards(self, boards: pd.DataFrame) -> list[dict[str, Any]]:
        if boards.empty:
            return []
        count = max(1, int(getattr(self.config, "concept_recommend_count", 8)))
        data = boards.copy()
        data = data[~data["state"].astype(str).isin(["退潮规避"])].copy()
        if data.empty:
            data = boards.copy()
        data = data.sort_values(["observe_score", "rps10", "rps20"], ascending=False).head(count)
        rows = []
        for _, row in data.iterrows():
            rows.append(
                {
                    "concept_code": str(row.get("concept_code", "")),
                    "concept_name": str(row.get("concept_name", "")),
                    "state": str(row.get("state", "")),
                    "observe_score": round(self._num(row.get("observe_score")), 2),
                    "risk_flags": str(row.get("risk_flags", "")),
                    "reason": str(row.get("recommend_reason", "")),
                }
            )
        return rows

    def _recommended_stocks(self, stocks: pd.DataFrame, boards: pd.DataFrame) -> list[dict[str, Any]]:
        if stocks.empty:
            return []
        good_boards = set(
            boards[~boards["state"].astype(str).isin(["退潮规避", "加速过热"])]
            .sort_values("observe_score", ascending=False)
            .head(max(6, int(getattr(self.config, "concept_recommend_count", 8))))["concept_code"]
            .astype(str)
        )
        data = stocks[stocks["concept_code"].astype(str).isin(good_boards)].copy()
        if data.empty:
            data = stocks.copy()
        if "canslim_score" in data.columns and str(getattr(self.config, "concept_canslim_mode", "off") or "off").strip().lower() == "bonus":
            data["recommend_score"] = data["stock_observe_score"].fillna(0) * 0.85 + pd.to_numeric(data["canslim_score"], errors="coerce").fillna(0) * 0.15
        else:
            data["recommend_score"] = data["stock_observe_score"].fillna(0)
        data = data.sort_values(["recommend_score", "rps10", "rps20"], ascending=False).drop_duplicates("ts_code").head(12)
        return [
            {
                "code": str(row.get("code", "")),
                "ts_code": str(row.get("ts_code", "")),
                "name": str(row.get("name", "")),
                "concept_name": str(row.get("concept_name", "")),
                "role": str(row.get("role", "")),
                "recommend_score": round(self._num(row.get("recommend_score")), 2),
                "risk_flags": str(row.get("risk_flags", "")),
                "reason": str(row.get("reason", "")),
            }
            for _, row in data.iterrows()
        ]

    def _market_payload(
        self,
        trade_date: str,
        requested_date: str,
        date_note: str,
        boards: pd.DataFrame,
        stocks: pd.DataFrame,
        recommend_boards: list[dict[str, Any]],
        recommend_stocks: list[dict[str, Any]],
        snapshot_mode: str,
        realtime_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        realtime_count = int(realtime_snapshot.get("count", 0) or 0)
        return {
            "trade_date": trade_date,
            "requested_date": requested_date,
            "date_note": date_note,
            "board_count": len(boards),
            "stock_row_count": len(stocks),
            "recommend_board_count": len(recommend_boards),
            "recommend_stock_count": len(recommend_stocks),
            "rank_period": int(getattr(self.config, "concept_rank_period", 10)),
            "sort_field": str(getattr(self.config, "concept_sort_field", "ret10")),
            "score_enabled": bool(getattr(self.config, "concept_score_enabled", False)),
            "canslim_mode": str(getattr(self.config, "concept_canslim_mode", "off")),
            "snapshot_mode": snapshot_mode,
            "snapshot_state": self._snapshot_state(
                snapshot_mode,
                realtime_count,
                str(realtime_snapshot.get("source", "")),
            ),
            "snapshot_at": str(realtime_snapshot.get("quote_at", "")),
            "realtime_count": realtime_count,
            "realtime_source": str(realtime_snapshot.get("source", "")),
            "note": "完整排行不做默认强过滤；推荐观察只用于复盘优先级，不代表买入建议。",
        }

    def _canslim_enabled(self) -> bool:
        return str(getattr(self.config, "concept_canslim_mode", "off") or "off").strip().lower() != "off"

    def _attach_canslim(self, stocks: pd.DataFrame) -> pd.DataFrame:
        if stocks.empty:
            return stocks
        mode = str(getattr(self.config, "concept_canslim_mode", "off") or "off").strip().lower()
        self.log("财务质量叠加已开启，读取CANSLIM财务成长指标...")
        result = stocks.copy()
        if not bool(getattr(self.config, "canslim_include_fundamental", True)):
            result["canslim_pass"] = "未读取"
            result["canslim_score"] = pd.NA
            result["canslim_channel"] = "财务质量"
            result["canslim_finance_status"] = "已关闭"
            result["canslim_reason"] = "财务读取已关闭"
            return result
        try:
            finance_strategy = CanslimTaoStrategy(self.engine)
            overlay = finance_strategy._fundamental_scores(str(self.config.end_date))
        except Exception as exc:
            self.log(f"财务质量读取失败，相关列置空: {exc}")
            result["canslim_pass"] = "未知"
            result["canslim_score"] = pd.NA
            result["canslim_channel"] = "财务质量"
            result["canslim_finance_status"] = "读取失败"
            result["canslim_reason"] = "CANSLIM读取失败"
            return result

        if overlay.empty or "ts_code" not in overlay.columns:
            result["canslim_pass"] = "未知"
            result["canslim_score"] = pd.NA
            result["canslim_channel"] = "财务质量"
            result["canslim_finance_status"] = "无财务数据"
            result["canslim_reason"] = "未读取到可用财务指标"
            return result

        overlay["ts_code"] = overlay["ts_code"].astype(str)
        use_cols = [
            col
            for col in [
                "ts_code",
                "end_date",
                "eps_yoy",
                "basic_eps_yoy",
                "dt_eps_yoy",
                "profit_yoy",
                "sales_yoy",
                "annual_eps_yoy",
                "annual_roe",
                "finance_status",
                "finance_note",
                "c_score",
                "a_score",
            ]
            if col in overlay.columns
        ]
        result = result.merge(overlay[use_cols].drop_duplicates("ts_code"), on="ts_code", how="left", suffixes=("", "_canslim"))
        result["canslim_score"] = (
            pd.to_numeric(result.get("c_score", pd.NA), errors="coerce") * 0.6
            + pd.to_numeric(result.get("a_score", pd.NA), errors="coerce") * 0.4
        )
        result["canslim_pass"] = result.apply(self._finance_quality_pass_label, axis=1)
        result["canslim_channel"] = "财务质量"
        result["canslim_finance_status"] = result.get("finance_status", "").fillna("")
        result["canslim_reason"] = result.apply(self._finance_quality_reason, axis=1)
        drop_cols = [col for col in ["finance_status", "finance_note", "c_score", "a_score"] if col in result.columns]
        result = result.drop(columns=drop_cols)

        if mode == "filter":
            result = result[result["canslim_pass"] == "是"].copy()
        elif mode == "bonus":
            result["stock_observe_score"] = (
                pd.to_numeric(result["stock_observe_score"], errors="coerce").fillna(0)
                + pd.to_numeric(result["canslim_score"], errors="coerce").fillna(0) * 0.08
            ).clip(0, 100)
        return result

    def _finance_quality_pass_label(self, row: pd.Series) -> str:
        if pd.isna(row.get("canslim_score", pd.NA)):
            return "未知"
        return "是" if all(ok for _, _, _, ok in self._finance_quality_checks(row)) else "否"

    def _finance_quality_reason(self, row: pd.Series) -> str:
        if pd.isna(row.get("canslim_score", pd.NA)):
            return "无匹配财报或财务读取失败"
        checks = self._finance_quality_checks(row)
        failed = [f"{name}{self._fmt_metric(value)}<{threshold:g}%" for name, value, threshold, ok in checks if not ok]
        passed = [f"{name}{self._fmt_metric(value)}" for name, value, _, ok in checks if ok]
        status = str(row.get("canslim_finance_status", "") or "")
        if failed:
            return f"未通过：{'；'.join(failed[:4])}" + (f"；{status}" if status else "")
        return f"通过：{'；'.join(passed[:4])}" + (f"；{status}" if status else "")

    def _finance_quality_checks(self, row: pd.Series) -> list[tuple[str, float | None, float, bool]]:
        checks = [
            ("EPS同比", self._first_num(row, ["eps_yoy", "basic_eps_yoy"]), float(getattr(self.config, "canslim_min_eps_yoy", 25))),
            ("扣非EPS同比", self._num_or_none(row.get("dt_eps_yoy")), float(getattr(self.config, "canslim_min_dt_eps_yoy", 0))),
            ("净利同比", self._num_or_none(row.get("profit_yoy")), float(getattr(self.config, "canslim_min_profit_yoy", 20))),
            ("营收同比", self._num_or_none(row.get("sales_yoy")), float(getattr(self.config, "canslim_min_sales_yoy", 20))),
            ("年度EPS同比", self._num_or_none(row.get("annual_eps_yoy")), float(getattr(self.config, "canslim_min_annual_eps_yoy", 20))),
            ("ROE", self._num_or_none(row.get("annual_roe")), float(getattr(self.config, "canslim_min_roe", 5))),
        ]
        return [(name, value, threshold, value is not None and value >= threshold) for name, value, threshold in checks]

    def _first_num(self, row: pd.Series, keys: list[str]) -> float | None:
        for key in keys:
            value = self._num_or_none(row.get(key))
            if value is not None:
                return value
        return None

    def _num_or_none(self, value: Any) -> float | None:
        try:
            if pd.isna(value):
                return None
            result = float(value)
            if math.isfinite(result):
                return result
        except Exception:
            return None
        return None

    def _fmt_metric(self, value: float | None) -> str:
        return "缺失" if value is None else f"{value:.1f}%"

    @staticmethod
    def _series_clip(value: Any) -> pd.Series:
        if isinstance(value, pd.Series):
            return value.clip(lower=0, upper=1)
        return pd.Series(value).clip(lower=0, upper=1)

    @staticmethod
    def _clip(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _num(value: Any, default: float = 0.0) -> float:
        try:
            if pd.isna(value):
                return default
            result = float(value)
            if math.isfinite(result):
                return result
        except Exception:
            pass
        return default
