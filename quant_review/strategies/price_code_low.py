from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

import pandas as pd


class PriceCodeLowStrategy:
    """Realtime scanner for coded-price lows at an N-day low."""

    VOLUME_BASELINE_DAYS = 5

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.config = engine.config

    def log(self, message: str) -> None:
        self.engine.log(message)

    def run(self) -> dict[str, Any]:
        low_days = self._bounded_int(
            getattr(self.config, "price_code_low_days", 20),
            default=20,
            minimum=2,
            maximum=250,
        )
        volume_days = self._bounded_int(
            getattr(self.config, "price_code_volume_days", self.VOLUME_BASELINE_DAYS),
            default=self.VOLUME_BASELINE_DAYS,
            minimum=1,
            maximum=60,
        )
        exclude_st = bool(getattr(self.config, "price_code_exclude_st", True))
        exclude_chinext = bool(
            getattr(self.config, "price_code_exclude_chinext", True)
        )
        exclude_star = bool(getattr(self.config, "price_code_exclude_star", True))
        volume_filter = self._volume_filter(
            getattr(self.config, "price_code_volume_filter", "all")
        )
        use_patterns = bool(getattr(self.config, "price_code_use_patterns", True))
        volume_min_pct, volume_max_pct = self._volume_range(
            getattr(self.config, "price_code_volume_min_pct", 10),
            getattr(self.config, "price_code_volume_max_pct", 30),
        )
        industry_mv_rank_top = self._bounded_int(
            getattr(self.config, "price_code_industry_mv_rank_top", 5),
            default=5,
            minimum=0,
            maximum=1000,
        )
        old_lookback = int(getattr(self.config, "lookback_days", 70) or 70)
        self.config.lookback_days = max(old_lookback, low_days + volume_days + 10)
        try:
            self.log(
                f"读取交易日历，新低窗口 {low_days} 日，"
                f"量能基准为前 {volume_days} 个完整交易日，"
                f"量能变化范围 {volume_min_pct:g}% 至 {volume_max_pct:g}%，"
                f"暗语条件{'启用' if use_patterns else '关闭'}..."
            )
            trade_dates = self.engine._trade_dates()
        finally:
            self.config.lookback_days = old_lookback

        trade_date = trade_dates[-1]
        self.config.end_date = trade_date
        self.log(f"使用交易日: {trade_date}")

        self.log("读取股票基础信息和历史日线...")
        full_basic = self.engine._stock_basic()
        basic, excluded_counts = self._filter_stock_universe(
            full_basic,
            exclude_st=exclude_st,
            exclude_chinext=exclude_chinext,
            exclude_star=exclude_star,
        )
        industry_rank_trade_date = ""
        if industry_mv_rank_top > 0:
            daily_basic, industry_rank_trade_date = (
                self._latest_daily_basic_for_rank(trade_dates)
            )
            industry_members = self._sw_l2_members()
            rank_snapshot = self._industry_market_cap_snapshot(
                full_basic,
                daily_basic,
                industry_members,
                trade_date=industry_rank_trade_date,
            )
            basic = self._merge_industry_market_cap_ranks(basic, rank_snapshot)
            before_rank_filter = len(basic)
            industry_ranks = basic["industry_mv_rank"].fillna(0).astype(int)
            basic = basic[
                (industry_ranks > 0)
                & (industry_ranks <= industry_mv_rank_top)
            ].copy()
            self.log(
                f"申万二级行业市值前 {industry_mv_rank_top} 过滤："
                f"{before_rank_filter} 只缩减为 {len(basic)} 只，"
                f"市值日期 {industry_rank_trade_date}。"
            )
        else:
            basic = self._merge_industry_market_cap_ranks(
                basic,
                pd.DataFrame(),
            )
        filters = []
        if exclude_st:
            filters.append(f"ST {excluded_counts['st']}只")
        if exclude_chinext:
            filters.append(f"创业板 {excluded_counts['chinext']}只")
        if exclude_star:
            filters.append(f"科创板 {excluded_counts['star']}只")
        self.log(
            "股票范围过滤："
            + ("，".join(filters) if filters else "未启用排除项")
            + f"；保留 {len(basic)} 只。"
        )
        daily = self.engine._daily_window(trade_dates)

        self.log("读取全市场实时行情，使用当日实时最低价、现价和成交量...")
        realtime = self._realtime_quotes(basic, trade_date)
        if realtime.empty:
            raise RuntimeError("实时行情字段无法识别，无法运行新低量价策略。")

        rows = self._build_rows(
            basic=basic,
            daily=daily,
            realtime=realtime,
            trade_date=trade_date,
            low_days=low_days,
            volume_days=volume_days,
            volume_filter=volume_filter,
            volume_min_pct=volume_min_pct,
            volume_max_pct=volume_max_pct,
            use_patterns=use_patterns,
        )
        snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log(
            f"新低量价实时筛选完成：实时行情 {len(realtime)} 只，"
            f"近 {low_days} 日最低价命中 {len(rows)} 只。"
        )
        return {
            "ok": True,
            "trade_date": trade_date,
            "snapshot_time": snapshot_time,
            "low_days": low_days,
            "volume_days": volume_days,
            "volume_filter": volume_filter,
            "volume_threshold_pct": self._round(volume_min_pct),
            "volume_min_pct": self._round(volume_min_pct),
            "volume_max_pct": self._round(volume_max_pct),
            "use_patterns": use_patterns,
            "industry_mv_rank_top": industry_mv_rank_top,
            "industry_rank_trade_date": industry_rank_trade_date,
            "filters": {
                "exclude_st": exclude_st,
                "exclude_chinext": exclude_chinext,
                "exclude_star": exclude_star,
                "excluded_counts": excluded_counts,
            },
            "live_count": len(realtime),
            "count": len(rows),
            "rows": rows,
            "logs": self.engine.logs,
        }

    def run_backtest(self) -> dict[str, Any]:
        start_date = str(
            getattr(self.config, "price_code_backtest_start_date", "") or ""
        ).strip()
        end_date = str(
            getattr(self.config, "price_code_backtest_end_date", "") or ""
        ).strip()
        self._validate_backtest_dates(start_date, end_date)

        low_days = self._bounded_int(
            getattr(self.config, "price_code_low_days", 20),
            default=20,
            minimum=2,
            maximum=250,
        )
        volume_days = self._bounded_int(
            getattr(self.config, "price_code_volume_days", self.VOLUME_BASELINE_DAYS),
            default=self.VOLUME_BASELINE_DAYS,
            minimum=1,
            maximum=60,
        )
        max_signal_days = self._bounded_int(
            getattr(self.config, "price_code_backtest_max_signal_days", 120),
            default=120,
            minimum=5,
            maximum=500,
        )
        exclude_st = bool(getattr(self.config, "price_code_exclude_st", True))
        exclude_chinext = bool(
            getattr(self.config, "price_code_exclude_chinext", True)
        )
        exclude_star = bool(getattr(self.config, "price_code_exclude_star", True))
        volume_filter = self._volume_filter(
            getattr(self.config, "price_code_volume_filter", "all")
        )
        use_patterns = bool(getattr(self.config, "price_code_use_patterns", True))
        volume_min_pct, volume_max_pct = self._volume_range(
            getattr(self.config, "price_code_volume_min_pct", 10),
            getattr(self.config, "price_code_volume_max_pct", 30),
        )
        industry_mv_rank_top = self._bounded_int(
            getattr(self.config, "price_code_industry_mv_rank_top", 5),
            default=5,
            minimum=0,
            maximum=1000,
        )
        selected_codes = self._selected_codes(
            getattr(self.config, "price_code_backtest_codes", [])
        )

        calendar_start = (
            datetime.strptime(start_date, "%Y%m%d").date()
            - timedelta(days=max(120, (low_days + volume_days) * 3))
        ).strftime("%Y%m%d")
        requested_end = datetime.strptime(end_date, "%Y%m%d").date()
        calendar_end = min(requested_end + timedelta(days=14), date.today()).strftime(
            "%Y%m%d"
        )
        self.log(
            f"读取回测交易日历：{start_date} 至 {end_date}，"
            f"新低窗口 {low_days} 日，"
            f"量能变化范围 {volume_min_pct:g}% 至 {volume_max_pct:g}%，"
            f"暗语条件{'启用' if use_patterns else '关闭'}。"
        )
        calendar = self.engine._cached(
            f"trade_cal_{calendar_start}_{calendar_end}",
            lambda: self.engine.pro.trade_cal(
                exchange="",
                start_date=calendar_start,
                end_date=calendar_end,
                is_open="1",
            ),
        )
        if calendar.empty or "cal_date" not in calendar.columns:
            raise RuntimeError("交易日历为空，无法运行新低量价回测。")
        all_dates = sorted(calendar["cal_date"].astype(str).unique().tolist())
        date_to_next = {
            current: all_dates[index + 1]
            for index, current in enumerate(all_dates[:-1])
        }
        signal_dates = [
            item
            for item in all_dates
            if start_date <= item <= end_date and item in date_to_next
        ]
        if not signal_dates:
            raise RuntimeError("回测区间内没有具备下一交易日的完整信号日。")
        if len(signal_dates) > max_signal_days:
            raise RuntimeError(
                f"回测信号日共 {len(signal_dates)} 天，超过上限 {max_signal_days} 天，"
                "请缩短日期区间或调高上限。"
            )

        first_index = all_dates.index(signal_dates[0])
        last_next_date = date_to_next[signal_dates[-1]]
        last_index = all_dates.index(last_next_date)
        warmup_days = max(low_days - 1, volume_days) + 5
        price_dates = all_dates[max(0, first_index - warmup_days) : last_index + 1]
        self.log(
            f"读取 {len(price_dates)} 个交易日历史日线，"
            f"其中信号日 {len(signal_dates)} 个。"
        )
        daily = self.engine._daily_window(price_dates)
        full_basic = self.engine._stock_basic()
        basic, excluded_counts = self._filter_stock_universe(
            full_basic,
            exclude_st=exclude_st,
            exclude_chinext=exclude_chinext,
            exclude_star=exclude_star,
        )
        industry_rank_history = pd.DataFrame()
        if industry_mv_rank_top > 0:
            industry_rank_history = self._industry_market_cap_history(
                full_basic,
                signal_dates,
            )
            self.log(
                f"回测启用申万二级行业市值前 {industry_mv_rank_top}，"
                "按每个信号日的总市值排名。"
            )
        if selected_codes:
            symbols = basic.get(
                "symbol",
                basic["ts_code"].astype(str).str[:6],
            ).astype(str).str.zfill(6)
            basic = basic.loc[symbols.isin(selected_codes)].copy()
            self.log(f"回测仅保留页面勾选的 {len(basic)} 只股票。")
        rows = self._build_backtest_rows(
            basic=basic,
            daily=daily,
            start_date=start_date,
            end_date=end_date,
            low_days=low_days,
            volume_days=volume_days,
            next_trade_dates=date_to_next,
            volume_filter=volume_filter,
            volume_min_pct=volume_min_pct,
            volume_max_pct=volume_max_pct,
            use_patterns=use_patterns,
            industry_mv_rank_top=industry_mv_rank_top,
            industry_rank_history=industry_rank_history,
        )
        summary = self._backtest_summary(rows)
        self.log(
            f"新低量价次日收盘回测完成：{summary['signal_count']} 个信号，"
            f"次日上涨 {summary['up_count']} 个，胜率 {summary['win_rate']}%。"
        )
        return {
            "ok": True,
            "start_date": start_date,
            "end_date": end_date,
            "low_days": low_days,
            "volume_days": volume_days,
            "volume_filter": volume_filter,
            "volume_threshold_pct": self._round(volume_min_pct),
            "volume_min_pct": self._round(volume_min_pct),
            "volume_max_pct": self._round(volume_max_pct),
            "use_patterns": use_patterns,
            "industry_mv_rank_top": industry_mv_rank_top,
            "selected_codes": sorted(selected_codes),
            "filters": {
                "exclude_st": exclude_st,
                "exclude_chinext": exclude_chinext,
                "exclude_star": exclude_star,
                "excluded_counts": excluded_counts,
            },
            "summary": summary,
            "rows": rows,
            "logs": self.engine.logs,
        }

    def run_parameter_scan(self) -> dict[str, Any]:
        start_date = str(
            getattr(self.config, "price_code_backtest_start_date", "") or ""
        ).strip()
        end_date = str(
            getattr(self.config, "price_code_backtest_end_date", "") or ""
        ).strip()
        self._validate_backtest_dates(start_date, end_date)

        direction = str(
            getattr(self.config, "price_code_scan_direction", "both") or "both"
        ).strip().lower()
        if direction not in {"both", "expand", "shrink"}:
            direction = "both"
        minimum = self._bounded_float(
            getattr(self.config, "price_code_scan_min_pct", 10),
            default=10,
            minimum=0,
            maximum=100,
        )
        maximum = self._bounded_float(
            getattr(self.config, "price_code_scan_max_pct", 100),
            default=100,
            minimum=0,
            maximum=100,
        )
        step = self._bounded_float(
            getattr(self.config, "price_code_scan_step_pct", 10),
            default=10,
            minimum=0.1,
            maximum=100,
        )
        min_signals = self._bounded_int(
            getattr(self.config, "price_code_scan_min_signals", 30),
            default=30,
            minimum=1,
            maximum=100000,
        )
        thresholds = self._scan_thresholds(minimum, maximum, step)
        low_min_days = self._bounded_int(
            getattr(self.config, "price_code_scan_low_min_days", 10),
            default=10,
            minimum=2,
            maximum=250,
        )
        low_max_days = self._bounded_int(
            getattr(self.config, "price_code_scan_low_max_days", 50),
            default=50,
            minimum=2,
            maximum=250,
        )
        low_step_days = self._bounded_int(
            getattr(self.config, "price_code_scan_low_step_days", 10),
            default=10,
            minimum=1,
            maximum=250,
        )
        low_days_values = self._scan_low_days(
            low_min_days,
            low_max_days,
            low_step_days,
        )
        industry_rank_min = self._bounded_int(
            getattr(self.config, "price_code_scan_industry_rank_min", 1),
            default=1,
            minimum=1,
            maximum=1000,
        )
        industry_rank_max = self._bounded_int(
            getattr(self.config, "price_code_scan_industry_rank_max", 5),
            default=5,
            minimum=1,
            maximum=1000,
        )
        industry_rank_step = self._bounded_int(
            getattr(self.config, "price_code_scan_industry_rank_step", 1),
            default=1,
            minimum=1,
            maximum=1000,
        )
        industry_rank_values = self._scan_industry_ranks(
            industry_rank_min,
            industry_rank_max,
            industry_rank_step,
        )
        volume_days = self._bounded_int(
            getattr(self.config, "price_code_volume_days", self.VOLUME_BASELINE_DAYS),
            default=self.VOLUME_BASELINE_DAYS,
            minimum=1,
            maximum=60,
        )
        max_signal_days = self._bounded_int(
            getattr(self.config, "price_code_backtest_max_signal_days", 120),
            default=120,
            minimum=5,
            maximum=500,
        )
        exclude_st = bool(getattr(self.config, "price_code_exclude_st", True))
        exclude_chinext = bool(
            getattr(self.config, "price_code_exclude_chinext", True)
        )
        exclude_star = bool(getattr(self.config, "price_code_exclude_star", True))
        selected_codes = self._selected_codes(
            getattr(self.config, "price_code_backtest_codes", [])
        )

        self.log(
            f"参数扫描准备：新低天数 {low_days_values[0]} 至 "
            f"{low_days_values[-1]}，量能阈值 {thresholds[0]:g}% 至 "
            f"{thresholds[-1]:g}%，二级行业市值排名前 "
            f"{industry_rank_values[0]} 至前 {industry_rank_values[-1]}，"
            f"步长 {industry_rank_step}；历史行情只读取一次。"
        )
        prepared = self._prepare_scan_backtest_data(
            start_date=start_date,
            end_date=end_date,
            max_low_days=max(low_days_values),
            volume_days=volume_days,
            max_signal_days=max_signal_days,
            exclude_st=exclude_st,
            exclude_chinext=exclude_chinext,
            exclude_star=exclude_star,
            selected_codes=selected_codes,
        )

        scan_rows: list[dict[str, Any]] = []
        base_signal_counts: dict[str, int] = {}
        for low_days in low_days_values:
            base_rows = self._build_backtest_rows(
                basic=prepared["basic"],
                daily=prepared["daily"],
                start_date=start_date,
                end_date=end_date,
                low_days=low_days,
                volume_days=volume_days,
                next_trade_dates=prepared["date_to_next"],
                volume_filter="all",
                volume_threshold_pct=0,
                industry_rank_history=prepared["industry_rank_history"],
            )
            base_signal_counts[str(low_days)] = len(base_rows)
            scan_rows.extend(
                self._parameter_scan_rows(
                    base_rows,
                    thresholds=thresholds,
                    direction=direction,
                    min_signals=min_signals,
                    low_days=low_days,
                    industry_mv_rank_tops=industry_rank_values,
                )
            )
        scan_rows.sort(
            key=lambda row: (
                bool(row["qualified"]),
                bool(row["positive_expectation"]),
                float(row["win_rate"]),
                float(row["average_return_pct"]),
                int(row["signal_count"]),
            ),
            reverse=True,
        )
        for index, row in enumerate(scan_rows, start=1):
            row["rank"] = index

        qualified = [row for row in scan_rows if row["qualified"]]
        positive_candidates = [
            row for row in qualified if row["positive_expectation"]
        ]
        best_parameter = positive_candidates[0] if positive_candidates else None
        self.log(
            f"参数扫描完成：{len(low_days_values)} 组新低天数，"
            f"扫描 {len(scan_rows)} 组参数，"
            f"满足最低样本数 {min_signals} 的组合 {len(qualified)} 组。"
        )
        return {
            "ok": True,
            "start_date": start_date,
            "end_date": end_date,
            "low_days_values": low_days_values,
            "volume_days": volume_days,
            "selected_codes": sorted(selected_codes),
            "filters": {
                "exclude_st": exclude_st,
                "exclude_chinext": exclude_chinext,
                "exclude_star": exclude_star,
                "excluded_counts": prepared["excluded_counts"],
            },
            "scan_direction": direction,
            "scan_min_pct": thresholds[0],
            "scan_max_pct": thresholds[-1],
            "scan_step_pct": step,
            "scan_low_min_days": low_days_values[0],
            "scan_low_max_days": low_days_values[-1],
            "scan_low_step_days": low_step_days,
            "scan_industry_rank_min": industry_rank_values[0],
            "scan_industry_rank_max": industry_rank_values[-1],
            "scan_industry_rank_step": industry_rank_step,
            "scan_industry_mv_rank_tops": industry_rank_values,
            "min_signals": min_signals,
            "base_signal_count": sum(base_signal_counts.values()),
            "base_signal_counts": base_signal_counts,
            "combination_count": len(scan_rows),
            "qualified_count": len(qualified),
            "positive_candidate_count": len(positive_candidates),
            "best_parameter": best_parameter,
            "rows": scan_rows,
            "logs": self.engine.logs,
        }

    def _prepare_scan_backtest_data(
        self,
        *,
        start_date: str,
        end_date: str,
        max_low_days: int,
        volume_days: int,
        max_signal_days: int,
        exclude_st: bool,
        exclude_chinext: bool,
        exclude_star: bool,
        selected_codes: set[str],
    ) -> dict[str, Any]:
        calendar_start = (
            datetime.strptime(start_date, "%Y%m%d").date()
            - timedelta(days=max(120, (max_low_days + volume_days) * 3))
        ).strftime("%Y%m%d")
        requested_end = datetime.strptime(end_date, "%Y%m%d").date()
        calendar_end = min(requested_end + timedelta(days=14), date.today()).strftime(
            "%Y%m%d"
        )
        calendar = self.engine._cached(
            f"trade_cal_{calendar_start}_{calendar_end}",
            lambda: self.engine.pro.trade_cal(
                exchange="",
                start_date=calendar_start,
                end_date=calendar_end,
                is_open="1",
            ),
        )
        if calendar.empty or "cal_date" not in calendar.columns:
            raise RuntimeError("交易日历为空，无法运行参数扫描。")
        all_dates = sorted(calendar["cal_date"].astype(str).unique().tolist())
        date_to_next = {
            current: all_dates[index + 1]
            for index, current in enumerate(all_dates[:-1])
        }
        signal_dates = [
            item
            for item in all_dates
            if start_date <= item <= end_date and item in date_to_next
        ]
        if not signal_dates:
            raise RuntimeError("扫描区间内没有具备下一交易日的完整信号日。")
        if len(signal_dates) > max_signal_days:
            raise RuntimeError(
                f"扫描信号日共 {len(signal_dates)} 天，超过上限 "
                f"{max_signal_days} 天，请缩短日期区间或调高上限。"
            )

        first_index = all_dates.index(signal_dates[0])
        last_next_date = date_to_next[signal_dates[-1]]
        last_index = all_dates.index(last_next_date)
        warmup_days = max(max_low_days - 1, volume_days) + 5
        price_dates = all_dates[max(0, first_index - warmup_days) : last_index + 1]
        self.log(
            f"参数扫描一次读取 {len(price_dates)} 个交易日历史日线，"
            f"覆盖最大 {max_low_days} 日新低窗口。"
        )
        daily = self.engine._daily_window(price_dates)
        full_basic = self.engine._stock_basic()
        industry_rank_history = self._industry_market_cap_history(
            full_basic,
            signal_dates,
        )
        basic = full_basic
        basic, excluded_counts = self._filter_stock_universe(
            basic,
            exclude_st=exclude_st,
            exclude_chinext=exclude_chinext,
            exclude_star=exclude_star,
        )
        if selected_codes:
            symbols = basic.get(
                "symbol",
                basic["ts_code"].astype(str).str[:6],
            ).astype(str).str.zfill(6)
            basic = basic.loc[symbols.isin(selected_codes)].copy()
            self.log(f"参数扫描仅保留页面勾选的 {len(basic)} 只股票。")
        return {
            "daily": daily,
            "basic": basic,
            "date_to_next": date_to_next,
            "industry_rank_history": industry_rank_history,
            "excluded_counts": excluded_counts,
        }

    @classmethod
    def _parameter_scan_rows(
        cls,
        rows: list[dict[str, Any]],
        *,
        thresholds: list[float],
        direction: str,
        min_signals: int,
        low_days: int = 20,
        industry_mv_rank_tops: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        directions = ["expand", "shrink"] if direction == "both" else [direction]
        rank_tops = industry_mv_rank_tops or [0]
        if rank_tops != [0]:
            combined: list[dict[str, Any]] = []
            for industry_mv_rank_top in rank_tops:
                rank_rows = [
                    row
                    for row in rows
                    if 0
                    < int(float(row.get("industry_mv_rank", 0) or 0))
                    <= industry_mv_rank_top
                ]
                current_rows = cls._parameter_scan_rows(
                    rank_rows,
                    thresholds=thresholds,
                    direction=direction,
                    min_signals=min_signals,
                    low_days=low_days,
                    industry_mv_rank_tops=[0],
                )
                for row in current_rows:
                    row["industry_mv_rank_top"] = industry_mv_rank_top
                combined.extend(current_rows)
            combined.sort(
                key=lambda row: (
                    bool(row["qualified"]),
                    bool(row["positive_expectation"]),
                    float(row["win_rate"]),
                    float(row["average_return_pct"]),
                    int(row["signal_count"]),
                ),
                reverse=True,
            )
            for index, row in enumerate(combined, start=1):
                row["rank"] = index
            return combined
        result: list[dict[str, Any]] = []
        for current_direction in directions:
            for threshold in thresholds:
                if current_direction == "expand":
                    selected = [
                        row
                        for row in rows
                        if float(row.get("signal_pct_chg", 0) or 0) > 0
                        and float(row.get("volume_change_pct", 0) or 0) >= threshold
                    ]
                else:
                    selected = [
                        row
                        for row in rows
                        if float(row.get("signal_pct_chg", 0) or 0) < 0
                        and float(row.get("volume_change_pct", 0) or 0) <= -threshold
                    ]
                summary = cls._backtest_summary(selected)
                best_category = summary.get("best_category") or {}
                result.append(
                    {
                        "direction": current_direction,
                        "direction_text": (
                            "放量" if current_direction == "expand" else "缩量"
                        ),
                        "low_days": low_days,
                        "threshold_pct": cls._round(threshold),
                        "industry_mv_rank_top": 0,
                        "signal_count": summary["signal_count"],
                        "stock_count": summary["stock_count"],
                        "up_count": summary["up_count"],
                        "down_count": summary["down_count"],
                        "win_rate": summary["win_rate"],
                        "average_return_pct": summary["average_return_pct"],
                        "median_return_pct": summary["median_return_pct"],
                        "best_return_pct": summary["best_return_pct"],
                        "worst_return_pct": summary["worst_return_pct"],
                        "best_category": str(best_category.get("category", "")),
                        "best_category_win_rate": float(
                            best_category.get("win_rate", 0) or 0
                        ),
                        "qualified": summary["signal_count"] >= min_signals,
                        "positive_expectation": summary["average_return_pct"] > 0,
                    }
                )
        result.sort(
            key=lambda row: (
                bool(row["qualified"]),
                bool(row["positive_expectation"]),
                float(row["win_rate"]),
                float(row["average_return_pct"]),
                int(row["signal_count"]),
            ),
            reverse=True,
        )
        for index, row in enumerate(result, start=1):
            row["rank"] = index
        return result

    @staticmethod
    def _scan_thresholds(
        minimum: float,
        maximum: float,
        step: float,
    ) -> list[float]:
        if minimum > maximum:
            raise RuntimeError("扫描最小阈值不能大于最大阈值。")
        values: list[float] = []
        current = minimum
        while current <= maximum + 0.000001:
            values.append(round(current, 6))
            if len(values) > 50:
                raise RuntimeError("扫描参数组合过多，请调大步长或缩小范围。")
            current += step
        return values

    @staticmethod
    def _scan_low_days(
        minimum: int,
        maximum: int,
        step: int,
    ) -> list[int]:
        if minimum > maximum:
            raise RuntimeError("新低最小天数不能大于最大天数。")
        values = list(range(minimum, maximum + 1, step))
        if len(values) > 50:
            raise RuntimeError("新低天数组合过多，请调大步长或缩小范围。")
        return values

    @staticmethod
    def _scan_industry_ranks(
        minimum: int,
        maximum: int,
        step: int,
    ) -> list[int]:
        if minimum > maximum:
            raise RuntimeError("行业市值排名最小值不能大于最大值。")
        values = list(range(minimum, maximum + 1, step))
        if len(values) > 50:
            raise RuntimeError("行业市值排名组合过多，请调大步长或缩小范围。")
        return values

    @staticmethod
    def _validate_backtest_dates(start_date: str, end_date: str) -> None:
        for label, value in [("开始日期", start_date), ("结束日期", end_date)]:
            if len(value) != 8 or not value.isdigit():
                raise RuntimeError(f"{label}格式应为 YYYYMMDD。")
            try:
                datetime.strptime(value, "%Y%m%d")
            except ValueError as exc:
                raise RuntimeError(f"{label}不是有效日期。") from exc
        if start_date >= end_date:
            raise RuntimeError("回测开始日期必须早于结束日期。")

    def _build_backtest_rows(
        self,
        *,
        basic: pd.DataFrame,
        daily: pd.DataFrame,
        start_date: str,
        end_date: str,
        low_days: int,
        volume_days: int,
        next_trade_dates: dict[str, str],
        volume_filter: str = "all",
        volume_threshold_pct: float = 20,
        volume_min_pct: float | None = None,
        volume_max_pct: float | None = None,
        use_patterns: bool = True,
        industry_mv_rank_top: int = 0,
        industry_rank_history: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        minimum_volume_pct, maximum_volume_pct = self._effective_volume_range(
            volume_threshold_pct,
            volume_min_pct,
            volume_max_pct,
        )
        basic_columns = [
            "ts_code",
            "symbol",
            "name",
            "industry",
            "stock_type",
            "rank_industry",
            "rank_industry_code",
            "market_cap_yi",
            "industry_mv_rank",
        ]
        metadata = basic[[col for col in basic_columns if col in basic.columns]].copy()
        metadata["ts_code"] = metadata["ts_code"].astype(str)
        metadata_by_code = metadata.set_index("ts_code").to_dict("index")
        rank_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        if (
            industry_rank_history is not None
            and not industry_rank_history.empty
            and {"trade_date", "ts_code"}.issubset(
                industry_rank_history.columns
            )
        ):
            history = industry_rank_history.copy()
            history["trade_date"] = history["trade_date"].astype(str)
            history["ts_code"] = history["ts_code"].astype(str)
            rank_lookup = {
                (str(row["trade_date"]), str(row["ts_code"])): row
                for row in history.to_dict("records")
            }

        data = daily.copy()
        data["ts_code"] = data["ts_code"].astype(str)
        data["trade_date"] = data["trade_date"].astype(str)
        data = data[data["ts_code"].isin(set(metadata["ts_code"]))].copy()
        for column in ["low", "close", "pre_close", "pct_chg", "vol"]:
            if column not in data.columns:
                data[column] = 0
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.sort_values(["ts_code", "trade_date"])

        rows: list[dict[str, Any]] = []
        for ts_code, frame in data.groupby("ts_code", sort=False):
            frame = frame.sort_values("trade_date").reset_index(drop=True)
            lows = pd.to_numeric(frame["low"], errors="coerce")
            volumes = pd.to_numeric(frame["vol"], errors="coerce")
            previous_low = lows.shift(1).rolling(
                low_days - 1,
                min_periods=low_days - 1,
            ).min()
            previous_volume = volumes.shift(1).rolling(
                volume_days,
                min_periods=volume_days,
            ).mean()
            info = metadata_by_code.get(ts_code, {})

            for index, current in frame.iterrows():
                signal_date = str(current.get("trade_date", ""))
                if not (start_date <= signal_date <= end_date):
                    continue
                expected_next_date = next_trade_dates.get(signal_date)
                if not expected_next_date or index + 1 >= len(frame):
                    continue
                next_row = frame.iloc[index + 1]
                if str(next_row.get("trade_date", "")) != expected_next_date:
                    continue
                rank_info = rank_lookup.get((signal_date, ts_code), {})
                industry_mv_rank = int(
                    self._num(rank_info.get("industry_mv_rank"))
                )
                if industry_mv_rank_top > 0 and not (
                    0 < industry_mv_rank <= industry_mv_rank_top
                ):
                    continue

                signal_low = self._num(current.get("low"))
                prior_low = self._num(previous_low.iloc[index])
                signal_close = self._num(current.get("close"))
                next_close = self._num(next_row.get("close"))
                average_volume = self._num(previous_volume.iloc[index])
                current_volume = self._num(current.get("vol"))
                signal_pct_chg = self._num(current.get("pct_chg"))
                if signal_pct_chg == 0:
                    previous_close = (
                        self._num(frame.iloc[index - 1].get("close"))
                        if index > 0
                        else self._num(current.get("pre_close"))
                    )
                    if previous_close > 0 and signal_close > 0:
                        signal_pct_chg = (signal_close / previous_close - 1) * 100
                if (
                    signal_low <= 0
                    or prior_low <= 0
                    or signal_low > prior_low + 0.000001
                    or signal_close <= 0
                    or next_close <= 0
                    or average_volume <= 0
                ):
                    continue
                patterns = self.detect_price_codes(signal_low)
                if use_patterns and not patterns:
                    continue

                volume_ratio = current_volume / average_volume
                volume_change_pct = (volume_ratio - 1) * 100
                if not self._price_volume_match(
                    signal_pct_chg,
                    volume_change_pct,
                    volume_ratio,
                    minimum_volume_pct,
                    maximum_volume_pct,
                ):
                    continue
                if not self._volume_filter_match(
                    volume_filter,
                    volume_change_pct,
                    minimum_volume_pct,
                    maximum_volume_pct,
                ):
                    continue
                next_return_pct = (next_close / signal_close - 1) * 100
                low_gap_pct = (signal_low / prior_low - 1) * 100
                categories = self._unique([item["category"] for item in patterns])
                details = self._unique([item["detail"] for item in patterns])
                formatted_price, digits = self._price_parts(signal_low)
                symbol = str(info.get("symbol", "") or ts_code[:6]).zfill(6)
                rows.append(
                    {
                        "signal_date": signal_date,
                        "next_date": expected_next_date,
                        "code": symbol,
                        "ts_code": ts_code,
                        "name": str(info.get("name", "") or ""),
                        "stock_type": str(info.get("stock_type", "") or ""),
                        "industry": str(info.get("industry", "") or ""),
                        "rank_industry": str(
                            rank_info.get("rank_industry", "") or ""
                        ),
                        "rank_industry_code": str(
                            rank_info.get("rank_industry_code", "") or ""
                        ),
                        "market_cap_yi": self._round(
                            self._num(rank_info.get("market_cap_yi"))
                        ),
                        "industry_mv_rank": industry_mv_rank,
                        "signal_low": self._round(signal_low),
                        "price_text": formatted_price,
                        "digit_string": digits,
                        "code_categories": "、".join(categories),
                        "code_details": "；".join(details),
                        "previous_window_low": self._round(prior_low),
                        "low_gap_pct": self._round(low_gap_pct),
                        "signal_close": self._round(signal_close),
                        "signal_pct_chg": self._round(signal_pct_chg),
                        "next_close": self._round(next_close),
                        "next_return_pct": self._round(next_return_pct),
                        "next_result": (
                            "上涨"
                            if next_return_pct > 0.000001
                            else "下跌"
                            if next_return_pct < -0.000001
                            else "平盘"
                        ),
                        "volume_state": self._volume_state(
                            volume_change_pct,
                            volume_ratio,
                            minimum_volume_pct,
                            maximum_volume_pct,
                        ),
                        "volume_ratio": self._round(volume_ratio),
                        "volume_change_pct": self._round(volume_change_pct),
                    }
                )
        return sorted(
            rows,
            key=lambda row: (
                str(row.get("signal_date", "")),
                float(row.get("next_return_pct", 0) or 0),
            ),
            reverse=True,
        )

    @staticmethod
    def _backtest_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        returns = pd.Series(
            [float(row.get("next_return_pct", 0) or 0) for row in rows],
            dtype=float,
        )
        signal_count = len(rows)
        up_count = int((returns > 0).sum()) if not returns.empty else 0
        category_rows: dict[str, list[float]] = {}
        for row in rows:
            categories = [
                item.strip()
                for item in str(row.get("code_categories", "")).split("、")
                if item.strip()
            ]
            value = float(row.get("next_return_pct", 0) or 0)
            for category in categories:
                category_rows.setdefault(category, []).append(value)
        category_stats = []
        for category, values in category_rows.items():
            category_returns = pd.Series(values, dtype=float)
            category_up_count = int((category_returns > 0).sum())
            category_stats.append(
                {
                    "category": category,
                    "signal_count": len(values),
                    "up_count": category_up_count,
                    "down_count": int((category_returns < 0).sum()),
                    "win_rate": round(category_up_count / len(values) * 100, 2),
                    "average_return_pct": round(float(category_returns.mean()), 2),
                    "median_return_pct": round(float(category_returns.median()), 2),
                }
            )
        category_stats.sort(
            key=lambda item: (
                float(item["win_rate"]),
                int(item["signal_count"]),
                float(item["average_return_pct"]),
            ),
            reverse=True,
        )
        best_category = category_stats[0] if category_stats else None
        return {
            "signal_count": signal_count,
            "stock_count": len({str(row.get("ts_code", "")) for row in rows}),
            "up_count": up_count,
            "down_count": int((returns < 0).sum()) if not returns.empty else 0,
            "flat_count": int((returns == 0).sum()) if not returns.empty else 0,
            "win_rate": round(up_count / signal_count * 100, 2)
            if signal_count
            else 0.0,
            "average_return_pct": round(float(returns.mean()), 2)
            if not returns.empty
            else 0.0,
            "median_return_pct": round(float(returns.median()), 2)
            if not returns.empty
            else 0.0,
            "best_return_pct": round(float(returns.max()), 2)
            if not returns.empty
            else 0.0,
            "worst_return_pct": round(float(returns.min()), 2)
            if not returns.empty
            else 0.0,
            "category_stats": category_stats,
            "best_category": best_category,
        }

    def _sw_l2_members(self) -> pd.DataFrame:
        members = self.engine._sw_industry_members("sw_l2")
        required = {"ts_code", "rank_industry", "rank_industry_code"}
        if members.empty or not required.issubset(members.columns):
            raise RuntimeError(
                "申万二级行业成分为空，无法应用二级行业市值排名条件。"
            )
        result = members[
            ["ts_code", "rank_industry", "rank_industry_code"]
        ].copy()
        result["ts_code"] = result["ts_code"].astype(str)
        result["rank_industry"] = (
            result["rank_industry"].fillna("").astype(str).str.strip()
        )
        result["rank_industry_code"] = (
            result["rank_industry_code"].fillna("").astype(str)
        )
        return result[
            result["rank_industry"] != ""
        ].drop_duplicates("ts_code", keep="first")

    def _latest_daily_basic_for_rank(
        self,
        trade_dates: list[str],
    ) -> tuple[pd.DataFrame, str]:
        for trade_date in reversed(trade_dates[-5:]):
            daily_basic = self.engine._daily_basic(trade_date)
            if not self.engine._daily_basic_is_empty(daily_basic):
                return daily_basic, trade_date
        raise RuntimeError("最近交易日总市值数据为空，无法计算二级行业市值排名。")

    @staticmethod
    def _industry_market_cap_snapshot(
        basic: pd.DataFrame,
        daily_basic: pd.DataFrame,
        industry_members: pd.DataFrame,
        *,
        trade_date: str = "",
    ) -> pd.DataFrame:
        columns = [
            "ts_code",
            "rank_industry",
            "rank_industry_code",
            "market_cap_yi",
            "industry_mv_rank",
        ]
        if trade_date:
            columns.insert(0, "trade_date")
        if basic.empty or daily_basic.empty or industry_members.empty:
            return pd.DataFrame(columns=columns)

        universe = basic[["ts_code"]].copy()
        universe["ts_code"] = universe["ts_code"].astype(str)
        members = industry_members[
            ["ts_code", "rank_industry", "rank_industry_code"]
        ].copy()
        members["ts_code"] = members["ts_code"].astype(str)
        market_cap = daily_basic[["ts_code", "total_mv"]].copy()
        market_cap["ts_code"] = market_cap["ts_code"].astype(str)
        market_cap["total_mv"] = pd.to_numeric(
            market_cap["total_mv"],
            errors="coerce",
        ).fillna(0)

        ranked = universe.merge(members, on="ts_code", how="inner")
        ranked = ranked.merge(
            market_cap.drop_duplicates("ts_code", keep="last"),
            on="ts_code",
            how="inner",
        )
        ranked = ranked[
            (ranked["rank_industry"].fillna("").astype(str).str.strip() != "")
            & (ranked["total_mv"] > 0)
        ].copy()
        ranked = ranked.sort_values(
            ["rank_industry", "total_mv", "ts_code"],
            ascending=[True, False, True],
        )
        ranked["industry_mv_rank"] = (
            ranked.groupby("rank_industry", sort=False).cumcount() + 1
        )
        ranked["market_cap_yi"] = ranked["total_mv"] / 10000
        if trade_date:
            ranked["trade_date"] = trade_date
        return ranked[columns].reset_index(drop=True)

    @staticmethod
    def _merge_industry_market_cap_ranks(
        basic: pd.DataFrame,
        ranks: pd.DataFrame,
    ) -> pd.DataFrame:
        rank_columns = [
            "rank_industry",
            "rank_industry_code",
            "market_cap_yi",
            "industry_mv_rank",
        ]
        data = basic.drop(
            columns=[column for column in rank_columns if column in basic.columns]
        ).copy()
        if ranks.empty:
            data["rank_industry"] = ""
            data["rank_industry_code"] = ""
            data["market_cap_yi"] = 0.0
            data["industry_mv_rank"] = 0
            return data
        use = ranks[["ts_code", *rank_columns]].copy()
        use["ts_code"] = use["ts_code"].astype(str)
        data["ts_code"] = data["ts_code"].astype(str)
        data = data.merge(use, on="ts_code", how="left")
        data["rank_industry"] = data["rank_industry"].fillna("")
        data["rank_industry_code"] = data["rank_industry_code"].fillna("")
        data["market_cap_yi"] = pd.to_numeric(
            data["market_cap_yi"],
            errors="coerce",
        ).fillna(0)
        data["industry_mv_rank"] = pd.to_numeric(
            data["industry_mv_rank"],
            errors="coerce",
        ).fillna(0).astype(int)
        return data

    def _industry_market_cap_history(
        self,
        basic: pd.DataFrame,
        signal_dates: list[str],
    ) -> pd.DataFrame:
        industry_members = self._sw_l2_members()
        frames: list[pd.DataFrame] = []
        missing_dates: list[str] = []
        total_dates = len(signal_dates)
        for index, trade_date in enumerate(signal_dates, start=1):
            daily_basic = self.engine._daily_basic(trade_date)
            if self.engine._daily_basic_is_empty(daily_basic):
                missing_dates.append(trade_date)
                continue
            frames.append(
                self._industry_market_cap_snapshot(
                    basic,
                    daily_basic,
                    industry_members,
                    trade_date=trade_date,
                )
            )
            if index % 20 == 0 or index == total_dates:
                self.log(
                    f"二级行业历史市值排名：已准备 {index}/{total_dates} 个信号日。"
                )
        if not frames:
            raise RuntimeError("回测区间总市值数据为空，无法计算二级行业市值排名。")
        if missing_dates:
            self.log(
                f"二级行业历史市值排名缺少 {len(missing_dates)} 个交易日，"
                "这些日期不会产生市值排名信号。"
            )
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _filter_stock_universe(
        basic: pd.DataFrame,
        *,
        exclude_st: bool,
        exclude_chinext: bool,
        exclude_star: bool,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        data = basic.copy()
        symbols = data.get(
            "symbol",
            data.get("ts_code", pd.Series("", index=data.index)).astype(str).str[:6],
        ).astype(str).str.zfill(6)
        names = data.get("name", pd.Series("", index=data.index)).fillna("").astype(str)
        normalized_names = names.str.upper().str.replace(" ", "", regex=False)

        st_mask = normalized_names.str.match(r"^(?:S\*)?ST|^\*ST|^SST", na=False)
        chinext_mask = symbols.str.startswith(("300", "301"))
        star_mask = symbols.str.startswith(("688", "689"))
        excluded_counts = {
            "st": int(st_mask.sum()),
            "chinext": int(chinext_mask.sum()),
            "star": int(star_mask.sum()),
        }

        keep = pd.Series(True, index=data.index)
        if exclude_st:
            keep &= ~st_mask
        if exclude_chinext:
            keep &= ~chinext_mask
        if exclude_star:
            keep &= ~star_mask
        return data.loc[keep].copy(), excluded_counts

    def _realtime_quotes(self, basic: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        realtime_raw = self.engine._realtime_list()
        if not realtime_raw.empty:
            normalized = self.engine._normalize_realtime(
                realtime_raw,
                basic,
                trade_date,
            )
            if not normalized.empty:
                normalized["data_source"] = "全市场 realtime_list"
                self.log("实时数据源：全市场 realtime_list。")
                return normalized

        self.log("全市场 realtime_list 为空，改用本地 Tushare rt_k 分批读取...")
        codes = basic.get("ts_code", pd.Series(dtype=str)).dropna().astype(str).tolist()
        frames: list[pd.DataFrame] = []
        for start in range(0, len(codes), 500):
            chunk = codes[start : start + 500]
            if not chunk:
                continue
            try:
                frame = self.engine.pro.rt_k(ts_code=",".join(chunk))
            except Exception as exc:
                self.log(
                    f"rt_k 第 {start // 500 + 1} 批读取失败"
                    f"（{len(chunk)}只）: {exc}"
                )
                continue
            if frame is not None and not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        normalized = self._normalize_rt_k(
            pd.concat(frames, ignore_index=True),
            trade_date,
        )
        self.log(f"实时数据源：Tushare rt_k，共返回 {len(normalized)} 只。")
        return normalized

    @staticmethod
    def _normalize_rt_k(frame: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        data = frame.copy()
        data.columns = [str(column).lower() for column in data.columns]
        if "ts_code" not in data.columns or "close" not in data.columns:
            return pd.DataFrame()
        for column in ["open", "high", "low", "close", "pre_close", "vol", "amount"]:
            if column not in data.columns:
                data[column] = 0
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
        data["pct_chg"] = (
            data["close"] / data["pre_close"].replace(0, pd.NA) - 1
        ).fillna(0) * 100
        data["amount"] = data["amount"] / 1000
        data["trade_date"] = trade_date
        data["ts_code"] = data["ts_code"].astype(str)
        data["data_source"] = "Tushare rt_k"
        data = data[data["close"] > 0].copy()
        columns = [
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "pct_chg",
            "vol",
            "amount",
            "data_source",
        ]
        return data[columns]

    def _build_rows(
        self,
        basic: pd.DataFrame,
        daily: pd.DataFrame,
        realtime: pd.DataFrame,
        trade_date: str,
        low_days: int,
        volume_days: int,
        volume_filter: str = "all",
        volume_threshold_pct: float = 20,
        volume_min_pct: float | None = None,
        volume_max_pct: float | None = None,
        use_patterns: bool = True,
    ) -> list[dict[str, Any]]:
        minimum_volume_pct, maximum_volume_pct = self._effective_volume_range(
            volume_threshold_pct,
            volume_min_pct,
            volume_max_pct,
        )
        history = daily[daily["trade_date"].astype(str) != trade_date].copy()
        history["ts_code"] = history["ts_code"].astype(str)
        for col in ["low", "vol"]:
            history[col] = pd.to_numeric(history[col], errors="coerce")
        hist_by_code = {
            code: frame.sort_values("trade_date")
            for code, frame in history.groupby("ts_code", sort=False)
        }

        basic_columns = [
            "ts_code",
            "symbol",
            "name",
            "industry",
            "stock_type",
            "rank_industry",
            "rank_industry_code",
            "market_cap_yi",
            "industry_mv_rank",
        ]
        basic_data = basic[[col for col in basic_columns if col in basic.columns]].copy()
        basic_data["ts_code"] = basic_data["ts_code"].astype(str)
        live = realtime.merge(basic_data, on="ts_code", how="left")

        rows: list[dict[str, Any]] = []
        required_history = low_days - 1
        for _, quote in live.iterrows():
            ts_code = str(quote.get("ts_code", "") or "")
            frame = hist_by_code.get(ts_code)
            if frame is None or len(frame) < required_history:
                continue

            current_low = self._num(quote.get("low"))
            if current_low <= 0:
                continue
            patterns = self.detect_price_codes(current_low)
            if use_patterns and not patterns:
                continue

            low_window = pd.to_numeric(
                frame["low"].tail(required_history),
                errors="coerce",
            ).dropna()
            low_window = low_window[low_window > 0]
            if len(low_window) < required_history:
                continue
            previous_low = float(low_window.min())
            if current_low > previous_low + 0.000001:
                continue

            previous_volumes = pd.to_numeric(
                frame["vol"].tail(volume_days),
                errors="coerce",
            ).dropna()
            previous_volumes = previous_volumes[previous_volumes > 0]
            average_volume = float(previous_volumes.mean()) if not previous_volumes.empty else 0.0
            live_volume = self._live_volume_in_hands(quote)
            volume_ratio = live_volume / average_volume if live_volume > 0 and average_volume > 0 else 0.0
            volume_change_pct = (volume_ratio - 1) * 100 if volume_ratio > 0 else 0.0
            volume_state = self._volume_state(
                volume_change_pct,
                volume_ratio,
                minimum_volume_pct,
                maximum_volume_pct,
            )

            close = self._num(quote.get("close"))
            pre_close = self._num(quote.get("pre_close"))
            pct_chg = self._num(quote.get("pct_chg"))
            if pct_chg == 0 and close > 0 and pre_close > 0:
                pct_chg = (close / pre_close - 1) * 100
            if not self._price_volume_match(
                pct_chg,
                volume_change_pct,
                volume_ratio,
                minimum_volume_pct,
                maximum_volume_pct,
            ):
                continue
            if not self._volume_filter_match(
                volume_filter,
                volume_change_pct,
                minimum_volume_pct,
                maximum_volume_pct,
            ):
                continue
            amount_yi = self._num(quote.get("amount")) / 100000
            formatted_price, digits = self._price_parts(current_low)
            categories = self._unique([item["category"] for item in patterns])
            details = self._unique([item["detail"] for item in patterns])
            low_gap_pct = (
                (current_low / previous_low - 1) * 100
                if previous_low > 0
                else 0.0
            )

            symbol = str(quote.get("symbol", "") or ts_code[:6]).zfill(6)
            rows.append(
                {
                    "code": symbol,
                    "ts_code": ts_code,
                    "name": str(quote.get("name", "") or ""),
                    "stock_type": str(quote.get("stock_type", "") or ""),
                    "industry": str(quote.get("industry", "") or ""),
                    "rank_industry": str(
                        quote.get("rank_industry", "") or ""
                    ),
                    "rank_industry_code": str(
                        quote.get("rank_industry_code", "") or ""
                    ),
                    "market_cap_yi": self._round(
                        self._num(quote.get("market_cap_yi"))
                    ),
                    "industry_mv_rank": int(
                        self._num(quote.get("industry_mv_rank"))
                    ),
                    "data_source": str(quote.get("data_source", "") or "实时行情"),
                    "low": self._round(current_low),
                    "price_text": formatted_price,
                    "digit_string": digits,
                    "code_categories": "、".join(categories),
                    "code_details": "；".join(details),
                    "code_count": len(patterns),
                    "low_days": low_days,
                    "previous_window_low": self._round(previous_low),
                    "low_gap_pct": self._round(low_gap_pct),
                    "latest_price": self._round(close),
                    "open": self._round(self._num(quote.get("open"))),
                    "high": self._round(self._num(quote.get("high"))),
                    "pct_chg": self._round(pct_chg),
                    "volume_state": volume_state,
                    "volume_ratio": self._round(volume_ratio),
                    "volume_change_pct": self._round(volume_change_pct),
                    "live_volume_wan_hands": self._round(live_volume / 10000),
                    "avg_volume_wan_hands": self._round(average_volume / 10000),
                    "volume_days": volume_days,
                    "amount_yi": self._round(amount_yi),
                }
            )

        return sorted(
            rows,
            key=lambda row: (
                float(row.get("low_gap_pct", 0) or 0),
                -int(row.get("code_count", 0) or 0),
                -float(row.get("volume_ratio", 0) or 0),
            ),
        )

    @classmethod
    def detect_price_codes(cls, price: float) -> list[dict[str, str]]:
        formatted, digits = cls._price_parts(price)
        if not formatted or len(digits) < 3:
            return []
        integer, cents = formatted.split(".")
        matches: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def add(category: str, detail: str) -> None:
            key = (category, detail)
            if key not in seen:
                seen.add(key)
                matches.append({"category": category, "detail": detail})

        if len(set(digits)) == 1:
            add("叠数", f"全同数 {digits}")

        trailing_run = 1
        for index in range(len(digits) - 2, -1, -1):
            if digits[index] != digits[-1]:
                break
            trailing_run += 1
        if trailing_run >= 2:
            add("叠数", f"尾部连续{trailing_run}位相同 {digits[-trailing_run:]}")

        if cents[0] == cents[1]:
            add("叠数", f"尾数对子 .{cents}")

        if len(digits) >= 4 and len(digits) % 2 == 0:
            pairs = [digits[index : index + 2] for index in range(0, len(digits), 2)]
            if len(pairs) >= 2 and all(pair[0] == pair[1] for pair in pairs):
                add("叠数", f"多组对子 {'-'.join(pairs)}")

        repeated = cls._whole_repeated_block(digits)
        if repeated is not None:
            block, repeat_count, sample = repeated
            label = "交替重复" if len(block) == 2 else "片段重复"
            add("重复片段", f"{label} {block}×{repeat_count}（{sample}）")

        if digits == digits[::-1]:
            add("对称回文", f"整体回文 {digits}")
        if len(integer) >= 3 and integer == integer[::-1]:
            add("对称回文", f"整数部分回文 {integer}")
        if len(integer) >= 2 and integer[-2:] == cents[::-1]:
            add("对称回文", f"整数尾部与小数倒序 {integer[-2:]}.{cents}")

        if cls._is_whole_sequence(digits, 1):
            add("顺子", f"整体正顺子 {digits}")
        if cls._is_whole_sequence(digits, -1):
            add("顺子", f"整体倒顺子 {digits}")

        return matches

    @staticmethod
    def _whole_repeated_block(digits: str) -> tuple[str, int, str] | None:
        for block_length in range(2, len(digits) // 2 + 1):
            if len(digits) % block_length != 0:
                continue
            block = digits[:block_length]
            repeat_count = len(digits) // block_length
            if repeat_count >= 2 and block * repeat_count == digits:
                return block, repeat_count, digits
        return None

    @staticmethod
    def _is_whole_sequence(digits: str, step: int) -> bool:
        if len(digits) < 3:
            return False
        return all(
            int(digits[index]) - int(digits[index - 1]) == step
            for index in range(1, len(digits))
        )

    @classmethod
    def _price_parts(cls, price: float) -> tuple[str, str]:
        try:
            decimal_price = Decimal(str(price)).quantize(
                Decimal("0.01"),
                rounding=ROUND_HALF_UP,
            )
        except (InvalidOperation, ValueError):
            return "", ""
        if decimal_price <= 0:
            return "", ""
        formatted = format(decimal_price, ".2f")
        return formatted, formatted.replace(".", "")

    @staticmethod
    def _live_volume_in_hands(quote: pd.Series) -> float:
        volume = PriceCodeLowStrategy._num(quote.get("vol"))
        amount_thousand = PriceCodeLowStrategy._num(quote.get("amount"))
        close = PriceCodeLowStrategy._num(quote.get("close"))
        if volume <= 0:
            return 0.0
        if amount_thousand > 0 and close > 0:
            expected_hands = amount_thousand * 10 / close
            unit_ratio = volume / expected_hands if expected_hands > 0 else 0
            if 30 <= unit_ratio <= 300:
                return volume / 100
        return volume

    @staticmethod
    def _volume_state(
        change_pct: float,
        ratio: float,
        volume_threshold_pct: float = 20,
        volume_max_pct: float = math.inf,
    ) -> str:
        if ratio <= 0:
            return "量能缺失"
        if volume_threshold_pct <= change_pct <= volume_max_pct:
            return f"放量 {change_pct:.2f}%"
        if -volume_max_pct <= change_pct <= -volume_threshold_pct:
            return f"缩量 {abs(change_pct):.2f}%"
        return f"基本持平 {change_pct:+.2f}%"

    @staticmethod
    def _price_volume_match(
        pct_chg: float,
        volume_change_pct: float,
        volume_ratio: float,
        volume_threshold_pct: float = 20,
        volume_max_pct: float = math.inf,
    ) -> bool:
        if volume_ratio <= 0:
            return False
        if pct_chg > 0.000001:
            return (
                volume_change_pct > 0
                and volume_threshold_pct <= volume_change_pct <= volume_max_pct
            )
        if pct_chg < -0.000001:
            return (
                volume_change_pct < 0
                and -volume_max_pct <= volume_change_pct <= -volume_threshold_pct
            )
        return False

    @staticmethod
    def _volume_filter(value: Any) -> str:
        normalized = str(value or "all").strip().lower()
        return normalized if normalized in {"all", "expand", "shrink"} else "all"

    @staticmethod
    def _volume_filter_match(
        volume_filter: str,
        volume_change_pct: float,
        volume_threshold_pct: float = 20,
        volume_max_pct: float = math.inf,
    ) -> bool:
        if volume_filter == "expand":
            return (
                volume_change_pct > 0
                and volume_threshold_pct <= volume_change_pct <= volume_max_pct
            )
        if volume_filter == "shrink":
            return (
                volume_change_pct < 0
                and -volume_max_pct <= volume_change_pct <= -volume_threshold_pct
            )
        return True

    @classmethod
    def _volume_range(cls, minimum: Any, maximum: Any) -> tuple[float, float]:
        minimum_pct = cls._bounded_float(
            minimum,
            default=10,
            minimum=0,
            maximum=10000,
        )
        maximum_pct = cls._bounded_float(
            maximum,
            default=30,
            minimum=0,
            maximum=10000,
        )
        if minimum_pct > maximum_pct:
            raise RuntimeError("量能变化最小值不能大于最大值。")
        return minimum_pct, maximum_pct

    @classmethod
    def _effective_volume_range(
        cls,
        legacy_threshold: Any,
        minimum: Any | None,
        maximum: Any | None,
    ) -> tuple[float, float]:
        if minimum is None and maximum is None:
            minimum_pct = cls._bounded_float(
                legacy_threshold,
                default=20,
                minimum=0,
                maximum=10000,
            )
            return minimum_pct, math.inf
        return cls._volume_range(
            legacy_threshold if minimum is None else minimum,
            10000 if maximum is None else maximum,
        )

    @staticmethod
    def _selected_codes(values: Any) -> set[str]:
        if not isinstance(values, (list, tuple, set)):
            return set()
        return {
            text
            for value in values
            if (text := "".join(char for char in str(value) if char.isdigit())[:6])
            and len(text) == 6
        }

    @staticmethod
    def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError):
            result = default
        return max(minimum, min(maximum, result))

    @staticmethod
    def _bounded_float(
        value: Any,
        default: float,
        minimum: float,
        maximum: float,
    ) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            result = default
        if not math.isfinite(result):
            result = default
        return max(minimum, min(maximum, result))

    @staticmethod
    def _num(value: Any) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return 0.0
        return result if math.isfinite(result) else 0.0

    @staticmethod
    def _round(value: float) -> float:
        return round(float(value), 2)

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))
