from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from quant_review.strategies.first_wave import FirstWaveStrategy


STRATEGY_LABELS = {
    "mainline": "主线复盘策略",
    "first_limit": "涨停板策略",
    "first_wave": "第一波启动策略",
    "canslim_tao": "CANSLIM+RPS策略",
    "canslim_early": "CANSLIM早期防钝化策略",
    "value_fcf": "低估值现金流策略",
    "dividend_low_vol": "红利低波防守策略",
}


@dataclass
class BacktestOrder:
    signal_date: str
    buy_date: str
    ts_code: str
    code: str
    name: str
    stock_type: str
    industry: str
    signal_score: float
    signal_rank: int
    raw: dict[str, Any]


class BacktestEngine:
    """Portfolio-style strategy backtest.

    回测口径：
    - 信号日收盘后运行策略。
    - 下一交易日开盘买入。
    - 盘中止盈/止损优先；持有天数和均线跌破在当日收盘执行。
    - 同一天同时触发止盈和止损时，保守按止损先触发处理。
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.config = engine.config
        self.logs: list[str] = []
        self.skipped: dict[str, int] = defaultdict(int)

    def run(self) -> dict[str, Any]:
        strategy = str(getattr(self.config, "backtest_strategy", "canslim_tao") or "canslim_tao")
        if strategy not in STRATEGY_LABELS:
            raise RuntimeError(f"暂不支持回测策略: {strategy}")

        start_date = self._clean_date(getattr(self.config, "backtest_start_date", ""))
        end_date = self._clean_date(getattr(self.config, "backtest_end_date", self.config.end_date))
        if not start_date or not end_date:
            raise RuntimeError("回测开始和结束日期都必须是 YYYYMMDD")
        if start_date >= end_date:
            raise RuntimeError("回测开始日期必须早于结束日期")

        signal_dates = self._calendar(start_date, end_date, warmup_days=0)
        if len(signal_dates) < 3:
            raise RuntimeError("回测区间内交易日不足，至少需要 3 个交易日")
        if len(signal_dates) > int(getattr(self.config, "backtest_max_signal_days", 260)):
            raise RuntimeError("回测交易日过多，先把区间控制在约一年内，避免接口请求过慢。")

        price_dates = self._calendar(start_date, end_date, warmup_days=80)
        self._log(f"准备价格数据: {price_dates[0]} - {price_dates[-1]}，共 {len(price_dates)} 个交易日")
        daily = self._price_frame(price_dates)

        self._log(f"开始生成 {STRATEGY_LABELS[strategy]} 历史信号，共 {len(signal_dates) - 1} 个信号日")
        orders_by_buy_date = self._build_orders(strategy, signal_dates, daily=daily, price_dates=price_dates)

        self._log("开始组合回测撮合")
        result = self._simulate(strategy, signal_dates, daily, orders_by_buy_date)
        result["logs"] = self.logs
        result["skipped"] = dict(self.skipped)
        return result

    def _calendar(self, start_date: str, end_date: str, warmup_days: int = 0) -> list[str]:
        start_obj = datetime.strptime(start_date, "%Y%m%d").date()
        fetch_start = (start_obj - timedelta(days=max(0, warmup_days) * 2 + 20)).strftime("%Y%m%d")
        cal = self.engine._cached(
            f"trade_cal_{fetch_start}_{end_date}",
            lambda: self.engine.pro.trade_cal(exchange="", start_date=fetch_start, end_date=end_date, is_open="1"),
        )
        if cal.empty or "cal_date" not in cal.columns:
            raise RuntimeError("交易日历为空，无法回测")
        dates = sorted(cal["cal_date"].astype(str).tolist())
        return [item for item in dates if fetch_start <= item <= end_date and (warmup_days > 0 or item >= start_date)]

    def _price_frame(self, trade_dates: list[str]) -> pd.DataFrame:
        daily = self.engine._daily_window(trade_dates).copy()
        for col in ["open", "high", "low", "close", "pre_close", "pct_chg", "amount"]:
            if col not in daily.columns:
                daily[col] = 0
            daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0)
        daily["trade_date"] = daily["trade_date"].astype(str)
        daily["ts_code"] = daily["ts_code"].astype(str)
        daily = daily.sort_values(["ts_code", "trade_date"])
        for window in [5, 10, 20]:
            daily[f"ma{window}"] = (
                daily.groupby("ts_code")["close"]
                .transform(lambda series, window=window: series.rolling(window, min_periods=window).mean())
                .fillna(0)
            )
        return daily

    def _build_orders(
        self,
        strategy: str,
        signal_dates: list[str],
        daily: pd.DataFrame | None = None,
        price_dates: list[str] | None = None,
    ) -> dict[str, list[BacktestOrder]]:
        if strategy == "first_wave" and daily is not None and price_dates is not None:
            return self._build_first_wave_orders(signal_dates, daily, price_dates)

        orders_by_buy_date: dict[str, list[BacktestOrder]] = defaultdict(list)
        max_rows = int(getattr(self.config, "backtest_signal_top_n", 20))
        for index, signal_date in enumerate(signal_dates[:-1]):
            buy_date = signal_dates[index + 1]
            try:
                rows = self._strategy_rows(strategy, signal_date)
            except Exception as exc:
                self.skipped["signal_error"] += 1
                self._log(f"{signal_date} 信号生成失败: {exc}")
                continue
            if not rows:
                self.skipped["empty_signal"] += 1
                continue
            orders = []
            for rank, row in enumerate(rows[:max_rows], start=1):
                code = str(row.get("code", "")).strip().zfill(6)
                if not code or code == "000000":
                    continue
                orders.append(
                    BacktestOrder(
                        signal_date=signal_date,
                        buy_date=buy_date,
                        ts_code=str(row.get("ts_code", "") or self._symbol_to_ts_code(code)),
                        code=code,
                        name=str(row.get("name", "")),
                        stock_type=str(row.get("stock_type", "")),
                        industry=str(row.get("industry", row.get("board", ""))),
                        signal_score=self._signal_score(strategy, row),
                        signal_rank=rank,
                        raw=dict(row),
                    )
                )
            orders_by_buy_date[buy_date] = orders
            if (index + 1) % 10 == 0:
                self._log(f"已生成 {index + 1}/{len(signal_dates) - 1} 个信号日")
        return orders_by_buy_date

    def _build_first_wave_orders(
        self,
        signal_dates: list[str],
        daily: pd.DataFrame,
        price_dates: list[str],
    ) -> dict[str, list[BacktestOrder]]:
        """Build first-wave backtest signals in one vectorized pass."""
        orders_by_buy_date: dict[str, list[BacktestOrder]] = defaultdict(list)
        max_rows = int(getattr(self.config, "backtest_signal_top_n", 20))
        self._log("第一波回测：一次性计算全区间启动特征，避免逐日重复读取日线。")
        old_review_mode = self.config.review_mode
        self.config.review_mode = "after_close"
        try:
            basic = self.engine._stock_basic()
            empty_daily_basic = pd.DataFrame(
                columns=["ts_code", "turnover_rate", "turnover_rate_f", "volume_ratio", "total_mv", "circ_mv"]
            )
            strategy = FirstWaveStrategy(self.engine)
            features = strategy._features(daily, basic, empty_daily_basic, price_dates)
            features = strategy.mark_pullbacks(features, price_dates)
            mode = str(getattr(self.config, "first_wave_entry_mode", "both") or "both")
            if mode not in {"both", "launch", "pullback"}:
                mode = "both"
            date_to_price_index = {date: index for index, date in enumerate(price_dates)}
            for index, signal_date in enumerate(signal_dates[:-1]):
                buy_date = signal_dates[index + 1]
                if signal_date not in date_to_price_index:
                    self.skipped["empty_signal"] += 1
                    continue
                available_dates = price_dates[: date_to_price_index[signal_date] + 1]
                latest = features[features["trade_date"] == signal_date]
                rows: list[dict[str, Any]] = []
                if mode in {"both", "launch"} and not latest.empty:
                    for _, row in latest[latest["first_wave_launch"]].iterrows():
                        rows.append(strategy._row_dict(row, "启动日", "低缺口跟随"))
                if mode in {"both", "pullback"} and not latest.empty:
                    existing = {row["ts_code"] for row in rows}
                    pullback_rows = latest[latest["first_wave_pullback"]]
                    for _, row in pullback_rows.iterrows():
                        if str(row["ts_code"]) in existing:
                            continue
                        rows.append(strategy._row_dict(row, "回踩确认", "回踩确认"))
                rows = sorted(rows, key=lambda item: float(item.get("score", 0) or 0), reverse=True)[:max_rows]
                if not rows:
                    self.skipped["empty_signal"] += 1
                    continue
                orders_by_buy_date[buy_date] = [
                    BacktestOrder(
                        signal_date=signal_date,
                        buy_date=buy_date,
                        ts_code=str(row.get("ts_code", "") or self._symbol_to_ts_code(str(row.get("code", "")))),
                        code=str(row.get("code", "")).strip().zfill(6),
                        name=str(row.get("name", "")),
                        stock_type=str(row.get("stock_type", "")),
                        industry=str(row.get("industry", "")),
                        signal_score=self._signal_score("first_wave", row),
                        signal_rank=rank,
                        raw=dict(row),
                    )
                    for rank, row in enumerate(rows, start=1)
                ]
                if (index + 1) % 20 == 0:
                    self._log(f"第一波信号已切片 {index + 1}/{len(signal_dates) - 1} 个交易日")
        finally:
            self.config.review_mode = old_review_mode
        return orders_by_buy_date

    def _strategy_rows(self, strategy: str, signal_date: str) -> list[dict[str, Any]]:
        old_values = {
            "end_date": self.config.end_date,
            "review_mode": self.config.review_mode,
            "canslim_probe_query": getattr(self.config, "canslim_probe_query", ""),
            "canslim_probe_only": getattr(self.config, "canslim_probe_only", False),
            "value_probe_query": getattr(self.config, "value_probe_query", ""),
            "dividend_probe_query": getattr(self.config, "dividend_probe_query", ""),
        }
        self.config.end_date = signal_date
        self.config.review_mode = "after_close"
        self.config.canslim_probe_query = ""
        self.config.canslim_probe_only = False
        self.config.value_probe_query = ""
        self.config.dividend_probe_query = ""
        self.engine.logs = []
        try:
            if strategy == "canslim_tao":
                data = self.engine.run_canslim_tao()
                rows = data.get("rows", [])
            elif strategy == "canslim_early":
                data = self.engine.run_canslim_early()
                rows = data.get("rows", [])
            elif strategy == "value_fcf":
                data = self.engine.run_value_fcf()
                rows = data.get("rows", [])
            elif strategy == "dividend_low_vol":
                data = self.engine.run_dividend_low_vol()
                rows = data.get("rows", [])
            elif strategy == "first_limit":
                data = self.engine.find_first_limit_up()
                rows = data.get("rows", [])
            elif strategy == "first_wave":
                data = self.engine.run_first_wave()
                rows = data.get("rows", [])
            else:
                data = self.engine.run()
                rows = data.get("stocks", [])
            return list(rows or [])
        finally:
            for key, value in old_values.items():
                setattr(self.config, key, value)

    def _simulate(
        self,
        strategy: str,
        signal_dates: list[str],
        daily: pd.DataFrame,
        orders_by_buy_date: dict[str, list[BacktestOrder]],
    ) -> dict[str, Any]:
        initial_cash = float(getattr(self.config, "backtest_initial_cash", 1_000_000))
        cash = initial_cash
        positions: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        equity_curve: list[dict[str, Any]] = []
        date_to_index = {date: idx for idx, date in enumerate(signal_dates)}
        daily_by_code = {code: frame.set_index("trade_date") for code, frame in daily.groupby("ts_code")}

        for current_date in signal_dates[1:]:
            today_index = date_to_index[current_date]
            cash, positions = self._execute_buys(
                current_date,
                today_index,
                cash,
                positions,
                orders_by_buy_date.get(current_date, []),
                daily_by_code,
            )
            cash, positions = self._execute_exits(
                current_date,
                today_index,
                cash,
                positions,
                trades,
                daily_by_code,
                final_close=False,
            )
            equity = self._portfolio_value(cash, positions, current_date, daily_by_code)
            equity_curve.append({"date": current_date, "equity": round(equity, 2)})

        final_date = signal_dates[-1]
        cash, positions = self._execute_exits(
            final_date,
            date_to_index[final_date],
            cash,
            positions,
            trades,
            daily_by_code,
            final_close=True,
        )
        final_equity = self._portfolio_value(cash, positions, final_date, daily_by_code)
        if equity_curve:
            equity_curve[-1]["equity"] = round(final_equity, 2)

        equity_curve = self._decorate_equity_curve(equity_curve, initial_cash)
        summary = self._summary(initial_cash, final_equity, trades, equity_curve, signal_dates)
        return {
            "ok": True,
            "strategy": strategy,
            "strategy_label": STRATEGY_LABELS[strategy],
            "start_date": signal_dates[0],
            "end_date": signal_dates[-1],
            "summary": summary,
            "equity_curve": equity_curve,
            "trades": trades,
            "open_positions": positions,
            "settings": self._settings(),
        }

    def _execute_buys(
        self,
        current_date: str,
        today_index: int,
        cash: float,
        positions: list[dict[str, Any]],
        orders: list[BacktestOrder],
        daily_by_code: dict[str, pd.DataFrame],
    ) -> tuple[float, list[dict[str, Any]]]:
        max_positions = int(getattr(self.config, "backtest_max_positions", 5))
        max_entries = int(getattr(self.config, "backtest_max_entries_per_day", 2))
        fee_rate = float(getattr(self.config, "backtest_fee_rate", 0.0003))
        slippage = float(getattr(self.config, "backtest_slippage_bps", 5)) / 10000
        skip_limit_open = bool(getattr(self.config, "backtest_skip_limit_open", True))

        held_codes = {pos["ts_code"] for pos in positions}
        bought_today = 0
        sorted_orders = sorted(orders, key=lambda item: (-item.signal_score, item.signal_rank))
        for order in sorted_orders:
            if bought_today >= max_entries or len(positions) >= max_positions:
                break
            if order.ts_code in held_codes:
                self.skipped["already_holding"] += 1
                continue
            price_row = self._row_at(daily_by_code, order.ts_code, current_date)
            if price_row is None or self._num(price_row.get("open")) <= 0:
                self.skipped["missing_buy_price"] += 1
                continue
            strategy_skip = self._strategy_buy_skip_reason(order, price_row)
            if strategy_skip:
                self.skipped[strategy_skip] += 1
                continue
            if skip_limit_open and self._is_limit_open(order, price_row):
                self.skipped["limit_open"] += 1
                continue

            open_price = self._num(price_row.get("open"))
            entry_price = open_price * (1 + slippage)
            equity_for_size = self._portfolio_value(cash, positions, current_date, daily_by_code, price_field="open")
            target_notional = min(cash * 0.98, equity_for_size / max(max_positions, 1))
            if target_notional <= 1000:
                self.skipped["cash_low"] += 1
                continue
            shares = target_notional / entry_price
            amount = shares * entry_price
            entry_fee = amount * fee_rate
            if amount + entry_fee > cash:
                shares = cash * 0.98 / entry_price
                amount = shares * entry_price
                entry_fee = amount * fee_rate
            if shares <= 0 or amount + entry_fee > cash:
                self.skipped["cash_low"] += 1
                continue

            cash -= amount + entry_fee
            positions.append(
                {
                    "signal_date": order.signal_date,
                    "entry_date": current_date,
                    "entry_index": today_index,
                    "ts_code": order.ts_code,
                    "code": order.code,
                    "name": order.name,
                    "stock_type": order.stock_type,
                    "industry": order.industry,
                    "signal_score": round(order.signal_score, 2),
                    "signal_rank": order.signal_rank,
                    "shares": shares,
                    "entry_price": entry_price,
                    "entry_amount": amount,
                    "entry_fee": entry_fee,
                    "raw": order.raw,
                }
            )
            held_codes.add(order.ts_code)
            bought_today += 1
        return cash, positions

    def _execute_exits(
        self,
        current_date: str,
        today_index: int,
        cash: float,
        positions: list[dict[str, Any]],
        trades: list[dict[str, Any]],
        daily_by_code: dict[str, pd.DataFrame],
        final_close: bool,
    ) -> tuple[float, list[dict[str, Any]]]:
        remaining = []
        for pos in positions:
            price_row = self._row_at(daily_by_code, pos["ts_code"], current_date)
            if price_row is None:
                remaining.append(pos)
                continue
            exit_plan = self._exit_plan(pos, price_row, today_index, final_close)
            if exit_plan is None:
                remaining.append(pos)
                continue
            exit_price, reason = exit_plan
            cash, trade = self._close_position(pos, current_date, exit_price, reason, cash)
            trades.append(trade)
        return cash, remaining

    def _exit_plan(self, pos: dict[str, Any], price_row: pd.Series, today_index: int, final_close: bool) -> tuple[float, str] | None:
        entry_price = float(pos["entry_price"])
        open_price = self._num(price_row.get("open"))
        high = self._num(price_row.get("high"))
        low = self._num(price_row.get("low"))
        close = self._num(price_row.get("close"))
        hold_days = today_index - int(pos["entry_index"]) + 1

        if final_close:
            return close, "回测结束"

        stop_loss = float(getattr(self.config, "backtest_stop_loss_pct", 8)) / 100
        take_profit = float(getattr(self.config, "backtest_take_profit_pct", 15)) / 100
        if stop_loss > 0:
            stop_price = entry_price * (1 - stop_loss)
            if low <= stop_price:
                return (open_price if open_price <= stop_price else stop_price), "止损"
        if take_profit > 0:
            take_price = entry_price * (1 + take_profit)
            if high >= take_price:
                return (open_price if open_price >= take_price else take_price), "止盈"

        exit_ma = str(getattr(self.config, "backtest_exit_ma", "ma10") or "none")
        if exit_ma in {"ma5", "ma10", "ma20"}:
            ma_value = self._num(price_row.get(exit_ma))
            if ma_value > 0 and close < ma_value:
                return close, f"跌破{exit_ma.upper()}"

        max_hold_days = int(getattr(self.config, "backtest_max_hold_days", 10))
        if max_hold_days > 0 and hold_days >= max_hold_days:
            return close, "到期卖出"
        return None

    def _close_position(
        self,
        pos: dict[str, Any],
        exit_date: str,
        raw_exit_price: float,
        reason: str,
        cash: float,
    ) -> tuple[float, dict[str, Any]]:
        fee_rate = float(getattr(self.config, "backtest_fee_rate", 0.0003))
        sell_tax_rate = float(getattr(self.config, "backtest_sell_tax_rate", 0.0005))
        slippage = float(getattr(self.config, "backtest_slippage_bps", 5)) / 10000
        exit_price = max(0.01, raw_exit_price * (1 - slippage))
        exit_amount = pos["shares"] * exit_price
        exit_fee = exit_amount * (fee_rate + sell_tax_rate)
        cash += exit_amount - exit_fee

        cost = pos["entry_amount"] + pos["entry_fee"]
        pnl = exit_amount - exit_fee - cost
        ret = pnl / cost * 100 if cost > 0 else 0
        return cash, {
            "signal_date": pos["signal_date"],
            "entry_date": pos["entry_date"],
            "exit_date": exit_date,
            "code": pos["code"],
            "name": pos["name"],
            "stock_type": pos["stock_type"],
            "industry": pos["industry"],
            "signal_score": pos["signal_score"],
            "signal_rank": pos["signal_rank"],
            "entry_price": round(pos["entry_price"], 3),
            "exit_price": round(exit_price, 3),
            "entry_amount": round(pos["entry_amount"], 2),
            "exit_amount": round(exit_amount, 2),
            "fee": round(pos["entry_fee"] + exit_fee, 2),
            "pnl": round(pnl, 2),
            "return_pct": round(ret, 2),
            "hold_days": self._trade_day_span(pos["entry_date"], exit_date),
            "exit_reason": reason,
        }

    def _portfolio_value(
        self,
        cash: float,
        positions: list[dict[str, Any]],
        current_date: str,
        daily_by_code: dict[str, pd.DataFrame],
        price_field: str = "close",
    ) -> float:
        value = cash
        for pos in positions:
            row = self._row_at(daily_by_code, pos["ts_code"], current_date)
            price = self._num(row.get(price_field)) if row is not None else 0
            if price <= 0:
                price = float(pos["entry_price"])
            value += pos["shares"] * price
        return float(value)

    def _decorate_equity_curve(self, equity_curve: list[dict[str, Any]], initial_cash: float) -> list[dict[str, Any]]:
        peak = initial_cash
        prev = initial_cash
        decorated = []
        for item in equity_curve:
            equity = float(item["equity"])
            peak = max(peak, equity)
            daily_return = (equity / prev - 1) * 100 if prev > 0 else 0
            drawdown = (equity / peak - 1) * 100 if peak > 0 else 0
            decorated.append(
                {
                    "date": item["date"],
                    "equity": round(equity, 2),
                    "daily_return_pct": round(daily_return, 3),
                    "drawdown_pct": round(drawdown, 2),
                }
            )
            prev = equity
        return decorated

    def _summary(
        self,
        initial_cash: float,
        final_equity: float,
        trades: list[dict[str, Any]],
        equity_curve: list[dict[str, Any]],
        signal_dates: list[str],
    ) -> dict[str, Any]:
        total_return = (final_equity / initial_cash - 1) * 100 if initial_cash > 0 else 0
        days = max(1, (datetime.strptime(signal_dates[-1], "%Y%m%d") - datetime.strptime(signal_dates[0], "%Y%m%d")).days)
        annual_return = ((final_equity / initial_cash) ** (365 / days) - 1) * 100 if initial_cash > 0 and final_equity > 0 else 0
        returns = [float(item["return_pct"]) for item in trades]
        wins = [value for value in returns if value > 0]
        losses = [value for value in returns if value < 0]
        daily_returns = [float(item["daily_return_pct"]) / 100 for item in equity_curve[1:]]
        sharpe = 0.0
        if len(daily_returns) >= 2:
            mean = sum(daily_returns) / len(daily_returns)
            variance = sum((value - mean) ** 2 for value in daily_returns) / (len(daily_returns) - 1)
            std = math.sqrt(variance)
            sharpe = mean / std * math.sqrt(252) if std > 0 else 0.0
        max_drawdown = min((float(item["drawdown_pct"]) for item in equity_curve), default=0)
        profit_factor = sum(wins) / abs(sum(losses)) if losses else (999.0 if wins else 0.0)
        return {
            "initial_cash": round(initial_cash, 2),
            "final_equity": round(final_equity, 2),
            "total_return_pct": round(total_return, 2),
            "annual_return_pct": round(annual_return, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe": round(sharpe, 2),
            "trades": len(trades),
            "win_rate_pct": round(len(wins) / len(trades) * 100, 2) if trades else 0,
            "profit_factor": round(profit_factor, 2),
            "avg_return_pct": round(sum(returns) / len(returns), 2) if returns else 0,
            "avg_hold_days": round(sum(float(item["hold_days"]) for item in trades) / len(trades), 1) if trades else 0,
            "best_trade_pct": round(max(returns), 2) if returns else 0,
            "worst_trade_pct": round(min(returns), 2) if returns else 0,
            "signal_days": len(signal_dates) - 1,
        }

    def _settings(self) -> dict[str, Any]:
        return {
            "initial_cash": getattr(self.config, "backtest_initial_cash", 1_000_000),
            "max_positions": getattr(self.config, "backtest_max_positions", 5),
            "max_entries_per_day": getattr(self.config, "backtest_max_entries_per_day", 2),
            "max_hold_days": getattr(self.config, "backtest_max_hold_days", 10),
            "take_profit_pct": getattr(self.config, "backtest_take_profit_pct", 15),
            "stop_loss_pct": getattr(self.config, "backtest_stop_loss_pct", 8),
            "exit_ma": getattr(self.config, "backtest_exit_ma", "ma10"),
            "fee_rate": getattr(self.config, "backtest_fee_rate", 0.0003),
            "sell_tax_rate": getattr(self.config, "backtest_sell_tax_rate", 0.0005),
            "slippage_bps": getattr(self.config, "backtest_slippage_bps", 5),
        }

    def _is_limit_open(self, order: BacktestOrder, row: pd.Series) -> bool:
        pre_close = self._num(row.get("pre_close"))
        open_price = self._num(row.get("open"))
        if pre_close <= 0 or open_price <= 0:
            return False
        name = order.name.upper()
        code = order.code
        if "ST" in name:
            rate = 0.05
        elif code.startswith(("300", "301", "688", "689")):
            rate = 0.20
        elif code.startswith(("8", "4", "92")):
            rate = 0.30
        else:
            rate = 0.10
        return open_price >= pre_close * (1 + rate) * 0.995

    def _strategy_buy_skip_reason(self, order: BacktestOrder, row: pd.Series) -> str:
        """Apply strategy-specific next-open filters before buying."""
        if str(getattr(self.config, "backtest_strategy", "")) != "first_wave":
            return ""
        trigger_close = self._num(order.raw.get("trigger_close_raw") or order.raw.get("trigger_close"))
        open_price = self._num(row.get("open"))
        if trigger_close <= 0 or open_price <= 0:
            return ""
        gap_pct = (open_price / trigger_close - 1) * 100
        max_gap = self._num(order.raw.get("max_buy_gap_pct"))
        min_gap = self._num(order.raw.get("min_buy_gap_pct"))
        if max_gap > 0 and gap_pct > max_gap:
            return "first_wave_gap_high"
        if min_gap < 0 and gap_pct < min_gap:
            return "first_wave_gap_low"
        return ""

    @staticmethod
    def _row_at(daily_by_code: dict[str, pd.DataFrame], ts_code: str, trade_date: str) -> pd.Series | None:
        frame = daily_by_code.get(ts_code)
        if frame is None or trade_date not in frame.index:
            return None
        row = frame.loc[trade_date]
        if isinstance(row, pd.DataFrame):
            return row.iloc[-1]
        return row

    @staticmethod
    def _signal_score(strategy: str, row: dict[str, Any]) -> float:
        keys = {
            "mainline": "stock_score",
            "first_limit": "market_cap_yi",
            "first_wave": "score",
            "canslim_tao": "score",
            "canslim_early": "score",
            "value_fcf": "value_score",
            "dividend_low_vol": "dividend_score",
        }
        value = row.get(keys.get(strategy, "score"), 0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _symbol_to_ts_code(symbol: str) -> str:
        code = str(symbol).zfill(6)
        if code.startswith(("6", "9")):
            return f"{code}.SH"
        if code.startswith(("8", "4", "92")):
            return f"{code}.BJ"
        return f"{code}.SZ"

    @staticmethod
    def _clean_date(value: Any) -> str:
        text = str(value or "").strip()
        return text if len(text) == 8 and text.isdigit() else ""

    @staticmethod
    def _num(value: Any) -> float:
        try:
            if pd.isna(value):
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _trade_day_span(entry_date: str, exit_date: str) -> int:
        try:
            start = datetime.strptime(entry_date, "%Y%m%d")
            end = datetime.strptime(exit_date, "%Y%m%d")
        except ValueError:
            return 0
        return max(1, (end - start).days + 1)

    def _log(self, message: str) -> None:
        self.logs.append(f"{datetime.now():%H:%M:%S} {message}")
