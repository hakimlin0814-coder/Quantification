from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


LOCAL_DATA_SOURCE_FILE = Path(__file__).resolve().parent.parent / "data_source.local.json"


def _load_local_data_source() -> dict[str, str]:
    if not LOCAL_DATA_SOURCE_FILE.exists():
        return {}
    try:
        data = json.loads(LOCAL_DATA_SOURCE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "token": str(data.get("token", "")).strip(),
        "tushare_url": str(data.get("tushare_url", "")).strip(),
    }


def get_data_source_defaults() -> dict[str, str]:
    local = _load_local_data_source()
    return {
        "token": os.getenv("TUSHARE_TOKEN", "").strip() or local.get("token", ""),
        "tushare_url": os.getenv("TUSHARE_URL", "").strip() or local.get("tushare_url", ""),
    }


_DATA_SOURCE_DEFAULTS = get_data_source_defaults()
DEFAULT_TUSHARE_TOKEN = _DATA_SOURCE_DEFAULTS["token"]
DEFAULT_TUSHARE_URL = _DATA_SOURCE_DEFAULTS["tushare_url"]


def default_cross_momentum_conditions() -> list[dict]:
    """横截面动量策略的默认主策略。

    条件保持结构化，页面、API 和历史回测共用同一份定义。旧版条件未提供
    calculation 时会自动按 ``value``（单日原值）解释。
    """
    return [
        {"enabled": True, "object": "stock", "time_type": "offset", "offset": -3, "indicator": "limit_up", "calculation": "value", "operator": "eq", "value": "limit_up", "threshold": ""},
        {"enabled": True, "object": "stock", "time_type": "offset", "offset": -2, "indicator": "limit_up", "calculation": "value", "operator": "eq", "value": "not_limit_up", "threshold": ""},
        {"enabled": True, "object": "stock", "time_type": "offset_range", "range_start_offset": -5, "range_end_offset": -4, "indicator": "pct_chg", "calculation": "sum", "operator": "le", "value": "", "threshold": 10},
        {"enabled": True, "object": "stock", "time_type": "offset", "offset": -2, "indicator": "pct_chg", "calculation": "value", "operator": "lt", "value": "", "threshold": 0},
        {"enabled": True, "object": "stock", "time_type": "offset", "offset": -2, "compare_offset": -3, "indicator": "turnover_rate_f", "calculation": "ratio", "operator": "ge", "value": "", "threshold": 1.3},
        {"enabled": True, "object": "stock", "time_type": "offset", "offset": -1, "indicator": "low_pct", "calculation": "value", "operator": "le", "value": "", "threshold": -9.5},
        {"enabled": False, "object": "stock", "time_type": "window", "offset": 0, "window_days": 10, "indicator": "stock_strength", "calculation": "value", "rank_scope": "current_pool", "rank_scope_label": "当前有效股票池", "rank_mode": "count", "rank_value": 20, "operator": "top", "value": "", "threshold": ""},
        {"enabled": False, "object": "concept", "time_type": "window", "offset": 0, "window_days": 8, "indicator": "concept_strength", "calculation": "value", "rank_scope": "all_concepts", "rank_scope_label": "全部有效概念", "rank_mode": "count", "rank_value": 10, "operator": "top", "value": "", "threshold": ""},
    ]


@dataclass
class ReviewConfig:
    """网页表单传入的复盘参数。

    这些字段基本都对应页面顶部的输入框。
    后续如果要调策略，优先把阈值放到这里，而不是写死在函数内部。
    """

    token: str = DEFAULT_TUSHARE_TOKEN
    end_date: str = date.today().strftime("%Y%m%d")
    tushare_url: str = DEFAULT_TUSHARE_URL
    review_mode: str = "after_close"
    lookback_days: int = 70
    index_period: str = "daily"
    board_source: str = "concept"
    board_scan_limit: int = 80
    top_board_count: int = 12
    top_stock_count: int = 80
    top_stock_percent: float = 0.30
    max_distance_ma10: float = 12.0
    min_amount_yi: float = 3.0
    min_board_up_ratio: float = 0.55
    min_board_score: float = 58.0
    min_member_count: int = 8
    max_board_ret5: float = 18.0
    max_board_ret10: float = 32.0
    min_stock_score: float = 55.0
    first_limit_exclude_st: bool = True
    first_limit_exclude_star: bool = True
    first_limit_exclude_chinext: bool = True
    first_limit_board_range: str = "1-1"
    first_limit_min_market_cap_yi: float = 100.0
    first_limit_max_turnover_rate: float = 10.0
    first_limit_turnover_field: str = "turnover_rate_f"
    first_limit_industry_mv_rank_top: int = 5
    first_limit_industry_rank_level: str = "sw_l3"
    first_wave_top_count: int = 80
    first_wave_entry_mode: str = "both"
    first_wave_stock_scope: str = "main_board"
    first_wave_min_score: float = 70.0
    first_wave_min_amount_yi: float = 1.5
    first_wave_min_amount_ratio: float = 1.8
    first_wave_max_amount_ratio: float = 5.5
    first_wave_min_launch_pct: float = 5.0
    first_wave_breakout_pct: float = 0.5
    first_wave_close_near_high: float = 0.97
    first_wave_max_range20: float = 30.0
    first_wave_max_range60: float = 75.0
    first_wave_max_pre_ret20: float = 16.0
    first_wave_max_pre_ret60: float = 32.0
    first_wave_max_since_low120: float = 55.0
    first_wave_min_rps20: float = 68.0
    first_wave_max_rps20: float = 93.0
    first_wave_min_rps20_delta10: float = 15.0
    first_wave_max_rps120: float = 96.0
    first_wave_min_breadth20: float = 50.0
    first_wave_min_breadth60: float = 40.0
    first_wave_min_up_ratio: float = 45.0
    first_wave_min_industry_rps5: float = 60.0
    first_wave_min_industry_amt_rps: float = 45.0
    first_wave_pullback_wait_days: int = 5
    first_wave_pullback_max_above: float = 3.5
    first_wave_pullback_max_below: float = 3.5
    first_wave_max_buy_gap_pct: float = 3.0
    first_wave_min_buy_gap_pct: float = -3.0
    first_wave_exclude_st: bool = True
    first_wave_exclude_star: bool = True
    first_wave_exclude_chinext: bool = True
    short_mainline_top_industry_count: int = 5
    short_mainline_stocks_per_industry: int = 4
    short_mainline_top_count: int = 20
    short_mainline_min_industry_members: int = 8
    short_mainline_min_industry_rps5: float = 88.0
    short_mainline_min_industry_rps10: float = 82.0
    short_mainline_min_industry_rps20: float = 70.0
    short_mainline_max_industry_rps60: float = 90.0
    short_mainline_min_industry_rps_gap: float = 8.0
    short_mainline_min_industry_amount_ratio: float = 1.12
    short_mainline_min_industry_up_ratio: float = 52.0
    short_mainline_min_strong_stocks: int = 3
    short_mainline_min_stock_rps5: float = 85.0
    short_mainline_min_stock_rps10: float = 80.0
    short_mainline_min_stock_rps20: float = 65.0
    short_mainline_min_stock_rps60: float = 40.0
    short_mainline_min_stock_amount_yi: float = 1.0
    short_mainline_min_stock_amount_ratio: float = 1.05
    short_mainline_max_stock_distance_ma20: float = 15.0
    short_mainline_max_stock_ret10: float = 35.0
    short_mainline_concept_enabled: bool = True
    short_mainline_min_concept_stocks: int = 2
    short_mainline_exclude_st: bool = True
    short_mainline_exclude_star: bool = True
    short_mainline_exclude_chinext: bool = True
    canslim_top_count: int = 80
    # all: any route; canslim: CANSLIM growth only; tao: Tao RPS only; both: overlap only.
    canslim_channel_mode: str = "canslim"
    canslim_min_score: float = 60.0
    canslim_min_rps: float = 90.0
    canslim_core_rps: float = 96.0
    canslim_near_high_ratio: float = 0.85
    canslim_min_amount_yi: float = 1.0
    # CANSLIM growth stocks need enough liquidity, but overly large caps may reduce elasticity.
    # A non-positive max value means "no upper limit".
    canslim_min_market_cap_yi: float = 100.0
    canslim_max_market_cap_yi: float = 0.0
    # Turnover is treated as a quality range: too low can mean no demand, too high can mean churn.
    # A non-positive max value means "no upper limit".
    canslim_min_turnover_rate: float = 0.0
    canslim_max_turnover_rate: float = 30.0
    # CANSLIM growth route thresholds. Set a value to 0 to disable that gate.
    canslim_min_eps_yoy: float = 25.0
    canslim_min_dt_eps_yoy: float = 0.0
    canslim_min_profit_yoy: float = 20.0
    canslim_min_sales_yoy: float = 20.0
    canslim_min_annual_eps_yoy: float = 20.0
    canslim_min_roe: float = 5.0
    # Industry mainline overlay: score first, optionally filter later.
    canslim_mainline_enabled: bool = True
    canslim_mainline_industry_level: str = "sw_l2"
    canslim_mainline_filter: str = "all"
    canslim_mainline_min_score: float = 60.0
    canslim_mainline_weight: float = 0.25
    # Tao-style RPS route thresholds. Financial values are defensive "quality floor" gates.
    tao_min_rps: float = 90.0
    tao_core_rps: float = 96.0
    tao_near_high_ratio: float = 0.85
    tao_min_eps_yoy: float = 0.0
    tao_min_profit_yoy: float = 0.0
    tao_min_sales_yoy: float = 0.0
    tao_min_roe: float = 0.0
    tao_institution_enabled: bool = True
    tao_institution_mode: str = "any"
    tao_fund_hold_enabled: bool = True
    tao_fund_hold_min_ratio: float = 2.0
    tao_north_hold_enabled: bool = True
    tao_north_hold_min_value_wan: float = 3000.0
    tao_institution_missing_pass: bool = False
    canslim_include_fundamental: bool = True
    canslim_exclude_st: bool = True
    canslim_exclude_star: bool = True
    canslim_exclude_chinext: bool = True
    value_top_count: int = 100
    value_max_pb: float = 1.0
    value_max_pe_ttm: float = 15.0
    value_min_roe: float = 15.0
    value_min_fcf_yield: float = 4.0
    value_exclude_st: bool = True
    value_exclude_star: bool = True
    value_exclude_chinext: bool = True
    value_probe_query: str = ""
    dividend_top_count: int = 100
    dividend_min_yield: float = 3.5
    dividend_max_yield: float = 12.0
    dividend_min_market_cap_yi: float = 150.0
    dividend_min_amount_yi: float = 1.0
    dividend_max_volatility: float = 30.0
    dividend_max_beta: float = 0.90
    dividend_max_drawdown: float = 25.0
    dividend_min_roe: float = 6.0
    dividend_max_pb: float = 2.0
    dividend_max_pe_ttm: float = 25.0
    dividend_max_distance_ma60: float = 18.0
    dividend_min_rs20: float = -8.0
    dividend_min_ma60_slope: float = -8.0
    dividend_max_ret20: float = 18.0
    dividend_exclude_st: bool = True
    dividend_exclude_star: bool = True
    dividend_exclude_chinext: bool = True
    dividend_probe_query: str = ""
    stock_search_query: str = ""
    stock_search_limit: int = 500
    industry_market_cap_rank_level: str = "sw_l2"
    industry_market_cap_rank_type: str = "industry_market_cap"
    realtime_ma_input: str = ""
    # ETF 实时强度策略：所有参数都可由页面的模态框覆盖。
    etf_min_history_days: int = 60
    etf_score_threshold: float = 80.0
    etf_top_n: int = 3
    etf_weight_recent: float = 35.0
    etf_weight_intraday: float = 25.0
    etf_weight_trend: float = 15.0
    etf_weight_liquidity: float = 15.0
    etf_weight_quality: float = 10.0
    etf_overheat_ret5: float = 15.0
    etf_overheat_ma20: float = 15.0
    etf_expected_hold_days: int = 5
    etf_probe_query: str = ""
    price_code_low_days: int = 20
    price_code_volume_days: int = 5
    price_code_exclude_st: bool = True
    price_code_exclude_chinext: bool = True
    price_code_exclude_star: bool = True
    price_code_use_patterns: bool = True
    price_code_volume_filter: str = "all"
    price_code_volume_min_pct: float = 10.0
    price_code_volume_max_pct: float = 30.0
    price_code_industry_mv_rank_top: int = 5
    price_code_backtest_start_date: str = ""
    price_code_backtest_end_date: str = ""
    price_code_backtest_max_signal_days: int = 120
    price_code_backtest_codes: list[str] = field(default_factory=list)
    price_code_scan_direction: str = "both"
    price_code_scan_min_pct: float = 10.0
    price_code_scan_max_pct: float = 100.0
    price_code_scan_step_pct: float = 10.0
    price_code_scan_low_min_days: int = 10
    price_code_scan_low_max_days: int = 50
    price_code_scan_low_step_days: int = 10
    price_code_scan_industry_rank_min: int = 1
    price_code_scan_industry_rank_max: int = 5
    price_code_scan_industry_rank_step: int = 1
    price_code_scan_min_signals: int = 30
    canslim_probe_query: str = ""
    canslim_probe_only: bool = False
    canslim_early_min_rps20: float = 68.0
    canslim_early_min_rps20_delta10: float = 8.0
    canslim_early_max_ret20: float = 32.0
    canslim_early_max_ret60: float = 80.0
    canslim_early_max_distance_ma20: float = 18.0
    canslim_early_max_distance_ma50: float = 38.0
    canslim_early_max_upper_shadow_days10: int = 4
    canslim_early_max_stale_score: float = 58.0
    canslim_early_min_growth_accel_score: float = 35.0
    concept_top_count: int = 80
    concept_rank_period: int = 10
    concept_sort_field: str = "ret10"
    concept_keyword: str = ""
    concept_member_warn_threshold: int = 8
    concept_stock_limit_per_board: int = 120
    concept_recommend_count: int = 8
    concept_score_enabled: bool = False
    concept_weight_trend: float = 0.0
    concept_weight_stability: float = 0.0
    concept_weight_breadth: float = 0.0
    concept_weight_money: float = 0.0
    concept_weight_leader: float = 0.0
    concept_weight_persistence: float = 0.0
    concept_weight_canslim: float = 0.0
    concept_weight_risk: float = 0.0
    concept_overheat_ret5: float = 15.0
    concept_overheat_ret10: float = 28.0
    concept_overheat_distance_ma20: float = 15.0
    concept_low_breadth_ma20: float = 45.0
    concept_canslim_mode: str = "off"
    concept_exclude_st: bool = False
    concept_exclude_star: bool = False
    concept_exclude_chinext: bool = False
    concept_exclude_bj: bool = False
    # 强势股票×强势概念交集策略
    strong_concept_stock_days: int = 2
    strong_concept_concept_days: int = 2
    strong_concept_stock_rank_mode: str = "percentile"
    strong_concept_stock_top_percent: float = 10.0
    strong_concept_stock_top_count: int = 100
    strong_concept_industry_mv_enabled: bool = False
    strong_concept_industry_mv_rank_top: int = 2
    strong_concept_industry_rank_level: str = "sw_l3"
    strong_concept_top_count: int = 10
    strong_concept_market_scope: list[str] = field(default_factory=lambda: ["main_board"])
    strong_concept_exclude_st: bool = True
    strong_concept_exclude_star: bool = True
    strong_concept_exclude_chinext: bool = True
    strong_concept_exclude_bj: bool = True
    strong_concept_member_limit: int = 1000
    strong_concept_recommend_count: int = 10
    # 横截面动量策略
    cross_momentum_days: int = 10
    cross_momentum_stock_rank_enabled: bool = True
    cross_momentum_rank_mode: str = "count"
    cross_momentum_top_count: int = 100
    cross_momentum_top_percent: float = 10.0
    cross_momentum_concept_enabled: bool = False
    cross_momentum_concept_days: int = 8
    cross_momentum_concept_rank_mode: str = "count"
    cross_momentum_concept_top_count: int = 10
    cross_momentum_concept_top_percent: float = 10.0
    cross_momentum_market_scope: list[str] = field(default_factory=lambda: ["main_board"])
    cross_momentum_exclude_st: bool = True
    cross_momentum_exclude_star: bool = True
    cross_momentum_exclude_chinext: bool = True
    cross_momentum_exclude_bj: bool = True
    cross_momentum_limit_up_enabled: bool = True
    cross_momentum_limit_up_offset: int = -3
    cross_momentum_limit_not_up_enabled: bool = True
    cross_momentum_limit_not_up_offset: int = -2
    cross_momentum_low_enabled: bool = True
    cross_momentum_low_offset: int = -1
    cross_momentum_low_threshold: float = -9.5
    cross_momentum_open_enabled: bool = False
    cross_momentum_open_offset: int = 0
    cross_momentum_open_operator: str = "gt"
    cross_momentum_open_threshold: float = 0.0
    cross_momentum_realtime_enabled: bool = False
    cross_momentum_realtime_auto_refresh: bool = False
    cross_momentum_realtime_interval: int = 60
    cross_momentum_recommend_count: int = 10
    # 横截面动量历史事件回测：以 end_date 为截止，统计最近 N 个已完成交易日。
    cross_momentum_backtest_days: int = 60
    # 动态条件构建器提交的结构化条件行。
    cross_momentum_conditions: list[dict] = field(default_factory=default_cross_momentum_conditions)
    backtest_strategy: str = "canslim_tao"
    backtest_start_date: str = ""
    backtest_end_date: str = ""
    backtest_initial_cash: float = 1_000_000.0
    backtest_max_positions: int = 5
    backtest_max_entries_per_day: int = 2
    backtest_signal_top_n: int = 20
    backtest_max_hold_days: int = 10
    backtest_take_profit_pct: float = 15.0
    backtest_stop_loss_pct: float = 8.0
    backtest_exit_ma: str = "ma10"
    backtest_fee_rate: float = 0.0003
    backtest_sell_tax_rate: float = 0.0005
    backtest_slippage_bps: float = 5.0
    backtest_skip_limit_open: bool = True
    backtest_max_signal_days: int = 260


