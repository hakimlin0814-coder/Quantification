from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from quant_review.strategies.first_limit import FirstLimitStrategy


class OneToTwoLiveStrategy:
    """1进2早盘实时判断。

    这不是自动交易信号，而是把候选票按竞价/早盘状态分层：
    可买、等承接、等回封、继续盯、观望、放弃。
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.config = engine.config

    def log(self, message: str) -> None:
        self.engine.log(message)

    def run(self) -> dict[str, Any]:
        self.log("读取交易日历...")
        trade_dates = self.engine._trade_dates()
        live_date = trade_dates[-1]
        if len(trade_dates) < 2:
            raise RuntimeError("交易日数据不足，无法找到前一交易日首板候选。")
        candidate_date = trade_dates[-2]
        self.log(f"实时判断日: {live_date}，首板候选日: {candidate_date}")

        self.log("读取股票基础信息...")
        basic = self.engine._stock_basic()

        self.log("生成前一交易日首板候选池...")
        old_range = self.config.first_limit_board_range
        self.config.first_limit_board_range = "1-1"
        try:
            candidate_basic = self.engine._daily_basic(candidate_date)
            candidate_daily = self.engine._daily_window([candidate_date])
            candidate_limit = self.engine._limit_list(candidate_date)
            candidates = FirstLimitStrategy(self.engine)._rows(
                basic,
                candidate_basic,
                candidate_daily,
                candidate_limit,
            )
        finally:
            self.config.first_limit_board_range = old_range

        if not candidates:
            self.log("首板候选池为空。")
            return {
                "ok": True,
                "live_date": live_date,
                "candidate_date": candidate_date,
                "stage": self._stage_label(self._decision_time()),
                "rows": [],
                "logs": self.engine.logs,
            }

        self.log("读取实时行情...")
        rt = self.engine._realtime_list()
        live_rows = self.engine._normalize_realtime(rt, basic, live_date) if not rt.empty else pd.DataFrame()
        if live_rows.empty:
            self.log("实时行情为空或字段无法识别。")

        result = self._score_candidates(
            candidates,
            candidate_daily,
            live_rows,
            basic,
        )
        self.log(f"1进2实时判断完成，共 {len(result)} 只。")
        return {
            "ok": True,
            "live_date": live_date,
            "candidate_date": candidate_date,
            "stage": self._stage_label(self._decision_time()),
            "rows": result,
            "logs": self.engine.logs,
        }

    def _score_candidates(
        self,
        candidates: list[dict[str, Any]],
        candidate_daily: pd.DataFrame,
        live_rows: pd.DataFrame,
        basic: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        hist = candidate_daily.copy()
        hist["ts_code"] = hist["ts_code"].astype(str)
        for col in ["open", "high", "low", "close", "amount", "vol"]:
            if col not in hist.columns:
                hist[col] = 0
            hist[col] = pd.to_numeric(hist[col], errors="coerce").fillna(0)
        hist_map = hist.set_index("ts_code").to_dict("index")

        live_map: dict[str, dict[str, Any]] = {}
        industry_live = pd.DataFrame()
        if not live_rows.empty:
            live = live_rows.copy()
            live["ts_code"] = live["ts_code"].astype(str)
            live_map = live.set_index("ts_code").to_dict("index")
            industry_live = self._industry_live_state(live, basic)

        now = self._decision_time()
        stage = self._stage_key(now)
        rows = []
        for item in candidates:
            ts_code = str(item.get("ts_code", ""))
            prev = hist_map.get(ts_code, {})
            live_item = live_map.get(ts_code)
            rows.append(self._score_one(item, prev, live_item, industry_live, stage))

        action_order = {"可买": 0, "等回封": 1, "等承接": 2, "继续盯": 3, "观望": 4, "放弃": 5, "无实时": 6}
        return sorted(
            rows,
            key=lambda row: (
                action_order.get(str(row.get("action", "")), 9),
                -float(row.get("score", 0) or 0),
                -float(row.get("current_pct", 0) or 0),
            ),
        )

    def _score_one(
        self,
        candidate: dict[str, Any],
        prev: dict[str, Any],
        live: dict[str, Any] | None,
        industry_live: pd.DataFrame,
        stage: str,
    ) -> dict[str, Any]:
        code = str(candidate.get("code", ""))
        ts_code = str(candidate.get("ts_code", ""))
        name = str(candidate.get("name", ""))
        industry = str(candidate.get("industry", ""))
        prev_close = self._num(prev.get("close", candidate.get("close", 0)))
        prev_high = self._num(prev.get("high", prev_close))
        prev_amount_yi = self._num(prev.get("amount", 0)) / 100000
        prev_vwap = self._vwap(prev)
        limit_price = self._limit_price(code, name, prev_close)

        if not live:
            return self._row(
                candidate,
                action="无实时",
                score=0,
                stage=stage,
                buy_point="等待实时行情",
                invalid_condition="无实时行情时不做早盘决策",
                reasons=["未匹配到实时行情"],
                prev_close=prev_close,
                prev_high=prev_high,
                prev_vwap=prev_vwap,
                limit_price=limit_price,
            )

        open_price = self._num(live.get("open", 0))
        current_price = self._num(live.get("close", 0))
        high_price = self._num(live.get("high", 0))
        low_price = self._num(live.get("low", 0))
        current_amount_yi = self._num(live.get("amount", 0)) / 100000
        if open_price <= 0:
            open_price = current_price
        if high_price <= 0:
            high_price = max(open_price, current_price)
        if low_price <= 0:
            low_price = min(open_price, current_price)

        open_pct = self._pct(open_price, prev_close)
        current_pct = self._pct(current_price, prev_close)
        amount_ratio = current_amount_yi / prev_amount_yi * 100 if prev_amount_yi > 0 else 0
        pullback_from_high = (high_price - current_price) / prev_close * 100 if prev_close > 0 else 0
        distance_to_limit = (limit_price - current_price) / prev_close * 100 if prev_close > 0 else 99
        industry_score, industry_note = self._industry_score(industry, industry_live)

        score, score_reasons = self._score_signal(
            open_pct,
            current_pct,
            amount_ratio,
            current_price,
            open_price,
            prev_close,
            prev_high,
            industry_score,
        )
        action, buy_point, invalid_condition, action_reasons = self._decision(
            stage,
            score,
            open_pct,
            current_pct,
            amount_ratio,
            current_price,
            open_price,
            prev_close,
            prev_high,
            high_price,
            limit_price,
            pullback_from_high,
            distance_to_limit,
        )
        reasons = action_reasons + score_reasons
        if industry_note:
            reasons.append(industry_note)

        return self._row(
            candidate,
            action=action,
            score=score,
            stage=stage,
            buy_point=buy_point,
            invalid_condition=invalid_condition,
            reasons=reasons[:5],
            prev_close=prev_close,
            prev_high=prev_high,
            prev_vwap=prev_vwap,
            limit_price=limit_price,
            open_price=open_price,
            current_price=current_price,
            high_price=high_price,
            low_price=low_price,
            open_pct=open_pct,
            current_pct=current_pct,
            current_amount_yi=current_amount_yi,
            amount_ratio=amount_ratio,
            distance_to_limit=distance_to_limit,
        )

    def _score_signal(
        self,
        open_pct: float,
        current_pct: float,
        amount_ratio: float,
        current_price: float,
        open_price: float,
        prev_close: float,
        prev_high: float,
        industry_score: float,
    ) -> tuple[float, list[str]]:
        score = 0.0
        reasons = []

        if 2 <= open_pct <= 5:
            score += 25
            reasons.append(f"开盘涨幅 {open_pct:.2f}% 处于1进2舒适区")
        elif 0 <= open_pct < 2 or 5 < open_pct <= 7:
            score += 15
            reasons.append(f"开盘涨幅 {open_pct:.2f}% 可观察但不是最佳区间")
        elif 7 < open_pct < 9.8:
            score += 8
            reasons.append(f"开盘涨幅 {open_pct:.2f}% 偏一致，追买性价比下降")
        else:
            score -= 20
            reasons.append(f"开盘涨幅 {open_pct:.2f}% 不适合直接接力")

        if current_price >= open_price:
            score += 20
            reasons.append("当前价未跌破开盘价，承接尚可")
        elif current_price >= open_price * 0.985:
            score += 12
            reasons.append("当前价贴近开盘价，仍需观察承接")
        else:
            score -= 15
            reasons.append("当前价明显跌破开盘价，有兑现压力")

        if 3 <= amount_ratio <= 10:
            score += 20
            reasons.append(f"当前成交额为昨日 {amount_ratio:.1f}%，量能较健康")
        elif 1 <= amount_ratio < 3:
            score += 8
            reasons.append(f"当前成交额为昨日 {amount_ratio:.1f}%，量能偏轻")
        elif 10 < amount_ratio <= 15:
            score += 10
            reasons.append(f"当前成交额为昨日 {amount_ratio:.1f}%，分歧偏大")
        elif amount_ratio > 15:
            score -= 10
            reasons.append(f"当前成交额为昨日 {amount_ratio:.1f}%，早盘爆量需防兑现")

        if current_price >= prev_high:
            score += 15
            reasons.append("当前价站上昨高，关键位较强")
        elif current_price >= prev_close:
            score += 5
            reasons.append("当前价仍在昨收上方，但未站稳昨高")
        else:
            score -= 20
            reasons.append("当前价跌破昨收，1进2强度不足")

        if current_pct >= open_pct:
            score += 10
            reasons.append("当前涨幅强于开盘涨幅，开盘后没有走弱")
        score += industry_score
        return max(0, min(100, round(score, 1))), reasons

    def _decision(
        self,
        stage: str,
        score: float,
        open_pct: float,
        current_pct: float,
        amount_ratio: float,
        current_price: float,
        open_price: float,
        prev_close: float,
        prev_high: float,
        high_price: float,
        limit_price: float,
        pullback_from_high: float,
        distance_to_limit: float,
    ) -> tuple[str, str, str, list[str]]:
        hard_abandon = (
            current_price < prev_close
            or (current_price < open_price * 0.98 and amount_ratio >= 4)
            or (amount_ratio > 15 and current_pct < open_pct - 2)
        )
        near_limit = current_price >= limit_price * 0.985 or current_pct >= 8.5 or distance_to_limit <= 1.5
        reasons = []

        if hard_abandon:
            return (
                "放弃",
                "不买",
                "重新站回开盘价和昨高前不再考虑",
                ["跌破关键位或放量走弱，疑似首板资金兑现"],
            )

        if stage == "竞价资格":
            if score >= 65:
                return (
                    "继续盯",
                    "9:30后看是否跌破开盘价和昨高",
                    "开盘后跌破昨高或高开低走则放弃",
                    ["竞价阶段只判定盯盘资格，不直接买"],
                )
            if score >= 45:
                return (
                    "观望",
                    "等开盘后重新站上昨高",
                    "开盘后不能转强则放弃",
                    ["竞价不够强，需要连续竞价确认"],
                )
            return ("放弃", "不买", "除非开盘快速转强并站回昨高", ["竞价强度不足"])

        if stage == "开盘确认":
            if score >= 75 and current_price >= open_price and current_price >= prev_high:
                return (
                    "继续盯",
                    "等9:35后突破早盘高点或回踩不破再买",
                    "跌破开盘价且3分钟不能收回",
                    ["开盘确认较强，但仍等第一波兑现完成"],
                )
            if current_price >= prev_high:
                return (
                    "等承接",
                    "回踩昨高或开盘价附近不破后再买",
                    "跌破昨高后反抽无力",
                    ["已站上昨高，但买点需要承接确认"],
                )
            return (
                "观望",
                "只看重新站上昨高后的机会",
                "跌破昨收或放量不涨则放弃",
                ["开盘后强度还不够"],
            )

        if stage == "买点确认":
            if near_limit:
                return (
                    "等回封",
                    f"接近涨停价 {limit_price:.2f}，等回封或封单稳定",
                    "炸板后跌破开盘价或昨高",
                    ["已接近涨停，半路追买盈亏比下降"],
                )
            if (
                score >= 80
                and current_price >= prev_high
                and current_price >= open_price
                and pullback_from_high <= 2.5
            ):
                return (
                    "可买",
                    f"站稳昨高 {prev_high:.2f} 和开盘价 {open_price:.2f} 后，放量突破早盘高点 {high_price:.2f}",
                    "跌破开盘价且3分钟不能收回，或跌破昨高",
                    ["9:35后仍站稳关键位，符合1进2弱转强"],
                )
            if score >= 65 and current_price >= prev_high:
                return (
                    "等承接",
                    f"等回踩昨高 {prev_high:.2f} / 开盘价 {open_price:.2f} 不破后再上攻",
                    "跌破昨高后反抽无力",
                    ["强度够观察，但还没有最佳买点"],
                )
            if score >= 50:
                return (
                    "观望",
                    "只看重新放量站上昨高",
                    "10点前不能转强则放弃",
                    ["信号一般，等待更明确的资金选择"],
                )
            return ("放弃", "不买", "除非重新站回昨高并放量转强", ["评分不足"])

        if stage == "延后确认":
            if near_limit:
                return (
                    "等回封",
                    f"只看涨停价 {limit_price:.2f} 附近回封确认",
                    "回落跌破昨高或开盘价",
                    ["9:45后新开仓只看封板确认"],
                )
            if score >= 82 and current_price >= prev_high and current_price >= open_price:
                return (
                    "可买",
                    "小仓，必须站稳昨高并继续上攻",
                    "不能快速冲板或跌破开盘价则退出观察",
                    ["延后确认仍强，但时间性价比下降"],
                )
            return (
                "放弃",
                "不买",
                "只保留涨停回封观察",
                ["9:45后仍不强，1进2确定性下降"],
            )

        if near_limit:
            return (
                "等回封",
                f"10点后只看涨停价 {limit_price:.2f} 回封",
                "回封失败或跌破昨高",
                ["10点后不做普通半路，只看封板确认"],
            )
        return ("放弃", "不买", "10点后没有封板动作，不再新开1进2", ["超过最佳1进2执行窗口"])

    def _row(
        self,
        candidate: dict[str, Any],
        action: str,
        score: float,
        stage: str,
        buy_point: str,
        invalid_condition: str,
        reasons: list[str],
        prev_close: float,
        prev_high: float,
        prev_vwap: float,
        limit_price: float,
        open_price: float = 0,
        current_price: float = 0,
        high_price: float = 0,
        low_price: float = 0,
        open_pct: float = 0,
        current_pct: float = 0,
        current_amount_yi: float = 0,
        amount_ratio: float = 0,
        distance_to_limit: float = 0,
    ) -> dict[str, Any]:
        return {
            "action": action,
            "score": round(score, 1),
            "stage": stage,
            "code": candidate.get("code", ""),
            "name": candidate.get("name", ""),
            "stock_type": candidate.get("stock_type", ""),
            "industry": candidate.get("industry", ""),
            "rank_industry": candidate.get("rank_industry", ""),
            "industry_mv_rank": candidate.get("industry_mv_rank", ""),
            "open_pct": round(open_pct, 2),
            "current_pct": round(current_pct, 2),
            "open_price": round(open_price, 2),
            "current_price": round(current_price, 2),
            "high_price": round(high_price, 2),
            "low_price": round(low_price, 2),
            "prev_close": round(prev_close, 2),
            "prev_high": round(prev_high, 2),
            "prev_vwap": round(prev_vwap, 2),
            "limit_price": round(limit_price, 2),
            "distance_to_limit": round(distance_to_limit, 2),
            "current_amount_yi": round(current_amount_yi, 2),
            "amount_ratio": round(amount_ratio, 1),
            "market_cap_yi": candidate.get("market_cap_yi", ""),
            "turnover_rate_f": candidate.get("turnover_rate_f", ""),
            "buy_point": buy_point,
            "invalid_condition": invalid_condition,
            "reasons": "；".join(reasons),
        }

    def _industry_live_state(self, live: pd.DataFrame, basic: pd.DataFrame) -> pd.DataFrame:
        cols = ["ts_code", "industry"]
        basic_use = basic[[col for col in cols if col in basic.columns]].copy()
        data = live.merge(basic_use, on="ts_code", how="left")
        data["industry"] = data["industry"].fillna("未分类").replace("", "未分类")
        data["pct_chg"] = pd.to_numeric(data["pct_chg"], errors="coerce").fillna(0)
        grouped = data.groupby("industry", dropna=False)["pct_chg"].agg(["mean", "count"])
        grouped["red_ratio"] = data.groupby("industry", dropna=False)["pct_chg"].apply(lambda s: float((s > 0).mean()))
        return grouped

    def _industry_score(self, industry: str, state: pd.DataFrame) -> tuple[float, str]:
        if state.empty or industry not in state.index:
            return 0, ""
        row = state.loc[industry]
        avg = float(row.get("mean", 0) or 0)
        red_ratio = float(row.get("red_ratio", 0) or 0)
        score = 0.0
        if avg >= 1.5:
            score += 6
        elif avg >= 0.5:
            score += 3
        if red_ratio >= 0.60:
            score += 4
        elif red_ratio >= 0.50:
            score += 2
        return score, f"基础行业实时均涨 {avg:.2f}%，红盘比例 {red_ratio:.0%}"

    def _decision_time(self) -> datetime:
        return datetime.now()

    def _stage_key(self, moment: datetime) -> str:
        value = moment.hour * 60 + moment.minute
        if value < 9 * 60 + 25:
            return "未到判断"
        if value < 9 * 60 + 30:
            return "竞价资格"
        if value < 9 * 60 + 35:
            return "开盘确认"
        if value < 9 * 60 + 45:
            return "买点确认"
        if value < 10 * 60:
            return "延后确认"
        return "过时降级"

    def _stage_label(self, moment: datetime) -> str:
        return self._stage_key(moment)

    def _limit_price(self, code: str, name: str, prev_close: float) -> float:
        if prev_close <= 0:
            return 0
        if "ST" in name.upper():
            rate = 0.05
        elif code.startswith(("300", "301", "688", "689")):
            rate = 0.20
        else:
            rate = 0.10
        return round(prev_close * (1 + rate), 2)

    def _vwap(self, row: dict[str, Any]) -> float:
        amount = self._num(row.get("amount", 0))
        vol = self._num(row.get("vol", 0))
        if vol <= 0:
            return 0
        return amount * 10 / vol

    @staticmethod
    def _pct(value: float, base: float) -> float:
        if base <= 0:
            return 0.0
        return (value / base - 1) * 100

    @staticmethod
    def _num(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
