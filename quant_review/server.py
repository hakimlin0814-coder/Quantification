from __future__ import annotations

import csv
import io
import json
import threading
import traceback
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from quant_review.ai_review import run_canslim_ai_review, run_concept_ai_review
from quant_review.config import DEFAULT_TUSHARE_TOKEN, DEFAULT_TUSHARE_URL, ReviewConfig, default_cross_momentum_conditions, get_data_source_defaults
from quant_review.engine import TushareReviewEngine
from src.oneil_model.config import OneilResearchConfig


HOST = "0.0.0.0"
PORT = 8765
LOCAL_CONTROL_HEADER = "X-Quant-Control"
LOCAL_CONTROL_VALUE = "quant-control-panel-v1"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
DOC_DIR = Path(__file__).resolve().parent / "docs"
INDEX_HTML = (TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")
MAINLINE_REVIEW_HTML = (TEMPLATE_DIR / "mainline_review.html").read_text(encoding="utf-8")
FIRST_LIMIT_HTML = (TEMPLATE_DIR / "first_limit.html").read_text(encoding="utf-8")
FIRST_WAVE_HTML = (TEMPLATE_DIR / "first_wave.html").read_text(encoding="utf-8")
SHORT_MAINLINE_HTML = (TEMPLATE_DIR / "short_mainline.html").read_text(encoding="utf-8")
CANSLIM_TAO_HTML = (TEMPLATE_DIR / "canslim_tao.html").read_text(encoding="utf-8")
CONCEPT_RADAR_HTML = (TEMPLATE_DIR / "concept_radar.html").read_text(encoding="utf-8")
STRONG_CONCEPT_HTML = (TEMPLATE_DIR / "strong_concept_momentum.html").read_text(encoding="utf-8")
CROSS_SECTIONAL_MOMENTUM_HTML = (TEMPLATE_DIR / "cross_sectional_momentum.html").read_text(encoding="utf-8")
VALUE_FCF_HTML = (TEMPLATE_DIR / "value_fcf.html").read_text(encoding="utf-8")
DIVIDEND_LOW_VOL_HTML = (TEMPLATE_DIR / "dividend_low_vol.html").read_text(encoding="utf-8")
STOCK_SEARCH_HTML = (TEMPLATE_DIR / "stock_search.html").read_text(encoding="utf-8")
INDUSTRY_MARKET_CAP_RANK_HTML = (TEMPLATE_DIR / "industry_market_cap_rank.html").read_text(encoding="utf-8")
REALTIME_MA_HTML = (TEMPLATE_DIR / "realtime_ma.html").read_text(encoding="utf-8")
ETF_MARKET_HTML = (TEMPLATE_DIR / "etf_market.html").read_text(encoding="utf-8")
PRICE_CODE_LOW_HTML = (TEMPLATE_DIR / "price_code_low.html").read_text(encoding="utf-8")
ONEIL_MODEL_HTML = (TEMPLATE_DIR / "oneil_model.html").read_text(encoding="utf-8")
GLOBAL_SETTINGS_JS_TEMPLATE = (TEMPLATE_DIR / "global_settings.js").read_text(encoding="utf-8")
APP_LAYOUT_CSS = (TEMPLATE_DIR / "app_layout.css").read_text(encoding="utf-8")
APP_LAYOUT_JS = (TEMPLATE_DIR / "app_layout.js").read_text(encoding="utf-8")
BACKTEST_JS = (TEMPLATE_DIR / "backtest.js").read_text(encoding="utf-8")
APP_MANIFEST = (TEMPLATE_DIR / "manifest.webmanifest").read_text(encoding="utf-8")
APP_ICON_SVG = (TEMPLATE_DIR / "app-icon.svg").read_text(encoding="utf-8")
SERVICE_WORKER_JS = (TEMPLATE_DIR / "service-worker.js").read_text(encoding="utf-8")


PWA_HEAD = """  <meta name="theme-color" content="#0d1117" />
  <meta name="application-name" content="量化复盘" />
  <meta name="apple-mobile-web-app-capable" content="yes" />
  <meta name="apple-mobile-web-app-title" content="量化复盘" />
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
  <link rel="manifest" href="/manifest.webmanifest" />
  <link rel="icon" href="/app-icon.svg" type="image/svg+xml" />
  <link rel="apple-touch-icon" href="/app-icon.svg" />
"""


def with_app_shell(html: str) -> str:
    """Inject mobile/PWA metadata into every HTML page."""
    result = html
    result = result.replace(
        '<meta name="viewport" content="width=device-width, initial-scale=1" />',
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />',
    )
    if '<link rel="manifest"' not in result:
        result = result.replace("</head>", f"{PWA_HEAD}</head>", 1)
    return result


def make_canslim_early_html() -> str:
    html = CANSLIM_TAO_HTML
    html = html.replace("<title>CANSLIM + RPS策略</title>", "<title>CANSLIM + RPS早期防钝化策略</title>")
    html = html.replace("<h1>CANSLIM + RPS策略</h1>", "<h1>CANSLIM + RPS早期防钝化策略</h1>")
    html = html.replace("CANSLIM + RPS策略说明", "CANSLIM + RPS早期防钝化策略说明")
    html = html.replace("CANSLIM+RPS 候选池", "CANSLIM早期防钝化候选池")
    html = html.replace("CANSLIM+RPS策略", "CANSLIM+RPS策略")
    html = html.replace(
        '<option value="/canslim-tao" selected>CANSLIM+RPS策略</option>',
        '<option value="/canslim-tao">CANSLIM+RPS策略</option>\n'
        '          <option value="/canslim-early" selected>CANSLIM早期防钝化</option>',
    )
    html = html.replace('const res = await fetch("/api/canslim_tao"', 'const res = await fetch("/api/canslim_early"')
    html = html.replace(
        '["code","代码"],["name","名称"],["stock_type","股票类型"],["industry","行业"],["concepts","产业题材"],["setup","形态"],["channel","通道"],["tao_institution_pass","陶机构"],["score","总分"],',
        '["code","代码"],["name","名称"],["stock_type","股票类型"],["industry","行业"],["concepts","产业题材"],["stage","阶段"],["stale_label","钝化"],["growth_trend","增速趋势"],["setup","形态"],["channel","通道"],["tao_institution_pass","陶机构"],["score","早期分"],',
    )
    html = html.replace(
        '["close","收盘"],["pct_chg","涨跌%"],["amount_yi","成交额亿"],["amount_ratio","量能比"],["turnover_rate_f","实换手%"],["market_cap_yi","总市值亿"],',
        '["close","收盘"],["pct_chg","涨跌%"],["amount_yi","成交额亿"],["amount_ratio","量能比"],["turnover_rate_f","实换手%"],["market_cap_yi","总市值亿"],["freshness_score","新鲜度"],["stale_score","钝化分"],',
    )
    html = html.replace(
        '["ret50","50日%"],["ret120","120日%"],["ret250","250日%"],["high_position_250","新高位置"],["ma5","MA5"],["ma10","MA10"],["ma20","MA20"],["distance_ma5","距5日%"],["distance_ma10","距10日%"],["distance_ma20","距20日%"],["distance_ma50","距50日%"],["ma_pullback","均线/买点"],',
        '["rps20","RPS20"],["rps20_delta10","RPS20变化"],["ret20_early","20日%"],["ret60_early","60日%"],["ret50","50日%"],["ret120","120日%"],["ret250","250日%"],["high_position_250","新高位置"],["ma5","MA5"],["ma10","MA10"],["ma20","MA20"],["distance_ma5","距5日%"],["distance_ma10","距10日%"],["distance_ma20","距20日%"],["distance_ma50","距50日%"],["ma_pullback","均线/买点"],["high_efficiency_20","20日效率"],["volume_price_efficiency","量价效率"],["upper_shadow_days_10","10日上影"],',
    )
    html = html.replace('["stop_hint","风控提示"],["reason","理由"]', '["stale_reason","钝化原因"],["stop_hint","风控提示"],["reason","理由"]')
    html = html.replace('a.download = `canslim_rps_${new Date().toISOString().slice(0,10)}.csv`;', 'a.download = `canslim_early_${new Date().toISOString().slice(0,10)}.csv`;')
    html = html.replace('strategy: "canslim_tao"', 'strategy: "canslim_early"')
    return html


CANSLIM_EARLY_HTML = make_canslim_early_html()

def payload_bool(payload: dict[str, Any], key: str, default: bool = True) -> bool:
    """把前端传入的布尔参数转成 Python bool。

    网页 fetch 正常会传 true/false，但以后如果手工调接口传入 "false"、"0"，
    这里也能按预期识别，避免 bool("false") 这种常见陷阱。
    """
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def price_code_volume_range(payload: dict[str, Any]) -> tuple[float, float]:
    if (
        "price_code_volume_min_pct" in payload
        or "price_code_volume_max_pct" in payload
    ):
        return (
            float(payload.get("price_code_volume_min_pct", 10)),
            float(payload.get("price_code_volume_max_pct", 30)),
        )
    if "price_code_volume_threshold_pct" in payload:
        return float(payload["price_code_volume_threshold_pct"]), 10000.0
    return 10.0, 30.0


class ReviewHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json({"ok": True, "service": "quant-review"})
        elif path == "/" or path == "/index.html":
            self._send(200, with_app_shell(INDEX_HTML), "text/html; charset=utf-8")
        elif path == "/global-settings.js":
            defaults = json.dumps(get_data_source_defaults(), ensure_ascii=False)
            self._send(200, GLOBAL_SETTINGS_JS_TEMPLATE.replace("__DATA_SOURCE_DEFAULTS__", defaults), "application/javascript; charset=utf-8")
        elif path == "/app-layout.css":
            self._send(200, APP_LAYOUT_CSS, "text/css; charset=utf-8")
        elif path == "/app-layout.js":
            self._send(200, APP_LAYOUT_JS, "application/javascript; charset=utf-8")
        elif path == "/manifest.webmanifest":
            self._send(200, APP_MANIFEST, "application/manifest+json; charset=utf-8")
        elif path == "/app-icon.svg":
            self._send(200, APP_ICON_SVG, "image/svg+xml; charset=utf-8")
        elif path == "/service-worker.js":
            self._send(
                200,
                SERVICE_WORKER_JS,
                "application/javascript; charset=utf-8",
                {"Cache-Control": "no-store, max-age=0", "Service-Worker-Allowed": "/"},
            )
        elif path == "/backtest.js":
            self._send(200, BACKTEST_JS, "application/javascript; charset=utf-8")
        elif path == "/docs/canslim-rps.md":
            body = (DOC_DIR / "canslim_rps_strategy.md").read_text(encoding="utf-8")
            self._send(
                200,
                body,
                "text/markdown; charset=utf-8",
                {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"},
            )
        elif path == "/docs/price-code-low.md":
            body = (DOC_DIR / "price_code_low_strategy.md").read_text(encoding="utf-8")
            self._send(
                200,
                body,
                "text/markdown; charset=utf-8",
                {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"},
            )
        elif path == "/docs/oneil-model.md":
            body = (TEMPLATE_DIR.parent.parent / "09_欧奈尔模型_策略逻辑详细设计.md").read_text(encoding="utf-8")
            self._send(
                200,
                body,
                "text/markdown; charset=utf-8",
                {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"},
            )
        elif path == "/mainline-review":
            self._send(200, with_app_shell(MAINLINE_REVIEW_HTML), "text/html; charset=utf-8")
        elif path == "/first-limit":
            self._send(200, with_app_shell(FIRST_LIMIT_HTML), "text/html; charset=utf-8")
        elif path == "/first-wave":
            self._send(200, with_app_shell(FIRST_WAVE_HTML), "text/html; charset=utf-8")
        elif path == "/short-mainline":
            self._send(200, with_app_shell(SHORT_MAINLINE_HTML), "text/html; charset=utf-8")
        elif path == "/canslim-tao":
            self._send(200, with_app_shell(CANSLIM_TAO_HTML), "text/html; charset=utf-8")
        elif path == "/canslim-early":
            self._send(200, with_app_shell(CANSLIM_EARLY_HTML), "text/html; charset=utf-8")
        elif path == "/oneil-model":
            self._send(200, with_app_shell(ONEIL_MODEL_HTML), "text/html; charset=utf-8")
        elif path == "/concept-radar":
            self._send(200, with_app_shell(CONCEPT_RADAR_HTML), "text/html; charset=utf-8")
        elif path == "/strong-concept-momentum":
            self._send(200, with_app_shell(STRONG_CONCEPT_HTML), "text/html; charset=utf-8")
        elif path == "/cross-sectional-momentum":
            self._send(200, with_app_shell(CROSS_SECTIONAL_MOMENTUM_HTML), "text/html; charset=utf-8")
        elif path == "/value-fcf":
            self._send(200, with_app_shell(VALUE_FCF_HTML), "text/html; charset=utf-8")
        elif path == "/dividend-low-vol":
            self._send(200, with_app_shell(DIVIDEND_LOW_VOL_HTML), "text/html; charset=utf-8")
        elif path == "/stock-search":
            self._send(200, with_app_shell(STOCK_SEARCH_HTML), "text/html; charset=utf-8")
        elif path == "/industry-market-cap-rank":
            self._send(200, with_app_shell(INDUSTRY_MARKET_CAP_RANK_HTML), "text/html; charset=utf-8")
        elif path == "/realtime-ma":
            self._send(200, with_app_shell(REALTIME_MA_HTML), "text/html; charset=utf-8")
        elif path == "/etf-market":
            self._send(200, with_app_shell(ETF_MARKET_HTML), "text/html; charset=utf-8")
        elif path == "/price-code-low":
            self._send(200, with_app_shell(PRICE_CODE_LOW_HTML), "text/html; charset=utf-8")
        else:
            self._send(404, "Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/shutdown":
            if (
                self.client_address[0] not in {"127.0.0.1", "::1"}
                or self.headers.get(LOCAL_CONTROL_HEADER) != LOCAL_CONTROL_VALUE
            ):
                self._send_json({"ok": False, "error": "shutdown is only available locally"}, 403)
                return
            self._send_json({"ok": True, "message": "review server is stopping"})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if path in {"/api/oneil_model/config/validate", "/api/oneil_model/screen"}:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8-sig"))
                research_config = OneilResearchConfig.from_payload(payload)
                if path.endswith("/config/validate"):
                    self._send_json({"ok": True, "status": "valid", "config": research_config.to_public_dict()})
                    return
                token = str(payload.get("token") or DEFAULT_TUSHARE_TOKEN).strip()
                tushare_url = str(payload.get("tushare_url") or DEFAULT_TUSHARE_URL).strip()
                end_date = str(payload.get("end_date", date.today().strftime("%Y%m%d"))).strip()
                if not token:
                    raise RuntimeError("缺少 Tushare Token")
                if not tushare_url:
                    raise RuntimeError("缺少 Tushare API地址")
                if len(end_date) != 8 or not end_date.isdigit():
                    raise RuntimeError("日期格式应为 YYYYMMDD")
                engine_config = ReviewConfig(
                    token=token,
                    tushare_url=tushare_url,
                    end_date=end_date,
                    review_mode="after_close",
                    lookback_days=research_config.rs_lookback + 5,
                )
                data = TushareReviewEngine(engine_config).run_oneil_model(research_config)
                self._send_json(data)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc), "trace": traceback.format_exc()}, 500)
            return
        if path == "/api/canslim_ai_review":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8-sig"))
                self._send_json(run_canslim_ai_review(payload))
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, 500)
            return
        if path == "/api/concept_ai_review":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8-sig"))
                self._send_json(run_concept_ai_review(payload))
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, 500)
            return
        if path not in {"/api/market_overview", "/api/review", "/api/mainline_review", "/api/first_limit", "/api/first_wave", "/api/short_mainline", "/api/one_to_two_live", "/api/canslim_tao", "/api/canslim_early", "/api/concept_radar", "/api/strong_concept_momentum", "/api/cross_sectional_momentum", "/api/cross_sectional_momentum_backtest", "/api/trade_date_shift", "/api/value_fcf", "/api/value_fcf_calc", "/api/dividend_low_vol", "/api/stock_search", "/api/industry_market_cap_rank", "/api/realtime_ma", "/api/etf_momentum", "/api/price_code_low", "/api/price_code_low_backtest", "/api/price_code_low_scan", "/api/backtest"}:
            self._send_json({"ok": False, "error": "unknown endpoint"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8-sig"))
            price_code_volume_min_pct, price_code_volume_max_pct = (
                price_code_volume_range(payload)
            )
            config = ReviewConfig(
                token=str(payload.get("token") or DEFAULT_TUSHARE_TOKEN).strip(),
                end_date=str(payload.get("end_date", date.today().strftime("%Y%m%d"))).strip(),
                tushare_url=str(payload.get("tushare_url") or DEFAULT_TUSHARE_URL).strip(),
                review_mode=str(payload.get("review_mode", "after_close")).strip(),
                board_source=str(payload.get("board_source", "concept")).strip(),
                lookback_days=int(payload.get("lookback_days", 70)),
                index_period=str(payload.get("index_period", "daily")).strip(),
                board_scan_limit=int(payload.get("board_scan_limit", 80)),
                top_board_count=int(payload.get("top_board_count", 12)),
                top_stock_count=int(payload.get("top_stock_count", 80)),
                top_stock_percent=float(payload.get("top_stock_percent", 0.30)),
                min_amount_yi=float(payload.get("min_amount_yi", 3)),
                min_board_up_ratio=float(payload.get("min_board_up_ratio", 0.55)),
                min_board_score=float(payload.get("min_board_score", 58)),
                min_member_count=int(payload.get("min_member_count", 8)),
                max_board_ret5=float(payload.get("max_board_ret5", 18)),
                max_board_ret10=float(payload.get("max_board_ret10", 32)),
                min_stock_score=float(payload.get("min_stock_score", 55)),
                max_distance_ma10=float(payload.get("max_distance_ma10", 12)),
                etf_min_history_days=int(payload.get("etf_min_history_days", 60)),
                etf_score_threshold=float(payload.get("etf_score_threshold", 80)),
                etf_top_n=int(payload.get("etf_top_n", 3)),
                etf_weight_recent=float(payload.get("etf_weight_recent", 35)),
                etf_weight_intraday=float(payload.get("etf_weight_intraday", 25)),
                etf_weight_trend=float(payload.get("etf_weight_trend", 15)),
                etf_weight_liquidity=float(payload.get("etf_weight_liquidity", 15)),
                etf_weight_quality=float(payload.get("etf_weight_quality", 10)),
                etf_overheat_ret5=float(payload.get("etf_overheat_ret5", 15)),
                etf_overheat_ma20=float(payload.get("etf_overheat_ma20", 15)),
                etf_expected_hold_days=int(payload.get("etf_expected_hold_days", 5)),
                etf_probe_query=str(payload.get("etf_probe_query", "")).strip(),
                first_limit_exclude_st=payload_bool(payload, "first_limit_exclude_st", True),
                first_limit_exclude_star=payload_bool(payload, "first_limit_exclude_star", True),
                first_limit_exclude_chinext=payload_bool(payload, "first_limit_exclude_chinext", True),
                first_limit_board_range=str(payload.get("first_limit_board_range", "1-1")).strip(),
                first_limit_min_market_cap_yi=float(payload.get("first_limit_min_market_cap_yi", 100)),
                first_limit_max_turnover_rate=float(payload.get("first_limit_max_turnover_rate", 10)),
                first_limit_turnover_field=str(payload.get("first_limit_turnover_field", "turnover_rate_f")).strip(),
                first_limit_industry_mv_rank_top=int(payload.get("first_limit_industry_mv_rank_top", 5)),
                first_limit_industry_rank_level=str(payload.get("first_limit_industry_rank_level", "sw_l3")).strip(),
                first_wave_top_count=int(payload.get("first_wave_top_count", payload.get("top_stock_count", 80))),
                first_wave_entry_mode=str(payload.get("first_wave_entry_mode", "both")).strip(),
                first_wave_stock_scope=str(payload.get("first_wave_stock_scope", "main_board")).strip(),
                first_wave_min_score=float(payload.get("first_wave_min_score", 70)),
                first_wave_min_amount_yi=float(payload.get("first_wave_min_amount_yi", 1.5)),
                first_wave_min_amount_ratio=float(payload.get("first_wave_min_amount_ratio", 1.8)),
                first_wave_max_amount_ratio=float(payload.get("first_wave_max_amount_ratio", 5.5)),
                first_wave_min_launch_pct=float(payload.get("first_wave_min_launch_pct", 5)),
                first_wave_breakout_pct=float(payload.get("first_wave_breakout_pct", 0.5)),
                first_wave_close_near_high=float(payload.get("first_wave_close_near_high", 0.97)),
                first_wave_max_range20=float(payload.get("first_wave_max_range20", 30)),
                first_wave_max_range60=float(payload.get("first_wave_max_range60", 75)),
                first_wave_max_pre_ret20=float(payload.get("first_wave_max_pre_ret20", 16)),
                first_wave_max_pre_ret60=float(payload.get("first_wave_max_pre_ret60", 32)),
                first_wave_max_since_low120=float(payload.get("first_wave_max_since_low120", 55)),
                first_wave_min_rps20=float(payload.get("first_wave_min_rps20", 68)),
                first_wave_max_rps20=float(payload.get("first_wave_max_rps20", 93)),
                first_wave_min_rps20_delta10=float(payload.get("first_wave_min_rps20_delta10", 15)),
                first_wave_max_rps120=float(payload.get("first_wave_max_rps120", 96)),
                first_wave_min_breadth20=float(payload.get("first_wave_min_breadth20", 50)),
                first_wave_min_breadth60=float(payload.get("first_wave_min_breadth60", 40)),
                first_wave_min_up_ratio=float(payload.get("first_wave_min_up_ratio", 45)),
                first_wave_min_industry_rps5=float(payload.get("first_wave_min_industry_rps5", 60)),
                first_wave_min_industry_amt_rps=float(payload.get("first_wave_min_industry_amt_rps", 45)),
                first_wave_pullback_wait_days=int(payload.get("first_wave_pullback_wait_days", 5)),
                first_wave_pullback_max_above=float(payload.get("first_wave_pullback_max_above", 3.5)),
                first_wave_pullback_max_below=float(payload.get("first_wave_pullback_max_below", 3.5)),
                first_wave_max_buy_gap_pct=float(payload.get("first_wave_max_buy_gap_pct", 3)),
                first_wave_min_buy_gap_pct=float(payload.get("first_wave_min_buy_gap_pct", -3)),
                first_wave_exclude_st=payload_bool(payload, "first_wave_exclude_st", True),
                first_wave_exclude_star=payload_bool(payload, "first_wave_exclude_star", True),
                first_wave_exclude_chinext=payload_bool(payload, "first_wave_exclude_chinext", True),
                short_mainline_top_industry_count=int(payload.get("short_mainline_top_industry_count", 5)),
                short_mainline_stocks_per_industry=int(payload.get("short_mainline_stocks_per_industry", 4)),
                short_mainline_top_count=int(payload.get("short_mainline_top_count", 20)),
                short_mainline_min_industry_members=int(payload.get("short_mainline_min_industry_members", 8)),
                short_mainline_min_industry_rps5=float(payload.get("short_mainline_min_industry_rps5", 88)),
                short_mainline_min_industry_rps10=float(payload.get("short_mainline_min_industry_rps10", 82)),
                short_mainline_min_industry_rps20=float(payload.get("short_mainline_min_industry_rps20", 70)),
                short_mainline_max_industry_rps60=float(payload.get("short_mainline_max_industry_rps60", 90)),
                short_mainline_min_industry_rps_gap=float(payload.get("short_mainline_min_industry_rps_gap", 8)),
                short_mainline_min_industry_amount_ratio=float(payload.get("short_mainline_min_industry_amount_ratio", 1.12)),
                short_mainline_min_industry_up_ratio=float(payload.get("short_mainline_min_industry_up_ratio", 52)),
                short_mainline_min_strong_stocks=int(payload.get("short_mainline_min_strong_stocks", 3)),
                short_mainline_min_stock_rps5=float(payload.get("short_mainline_min_stock_rps5", 85)),
                short_mainline_min_stock_rps10=float(payload.get("short_mainline_min_stock_rps10", 80)),
                short_mainline_min_stock_rps20=float(payload.get("short_mainline_min_stock_rps20", 65)),
                short_mainline_min_stock_rps60=float(payload.get("short_mainline_min_stock_rps60", 40)),
                short_mainline_min_stock_amount_yi=float(payload.get("short_mainline_min_stock_amount_yi", 1)),
                short_mainline_min_stock_amount_ratio=float(payload.get("short_mainline_min_stock_amount_ratio", 1.05)),
                short_mainline_max_stock_distance_ma20=float(payload.get("short_mainline_max_stock_distance_ma20", 15)),
                short_mainline_max_stock_ret10=float(payload.get("short_mainline_max_stock_ret10", 35)),
                short_mainline_concept_enabled=payload_bool(payload, "short_mainline_concept_enabled", True),
                short_mainline_min_concept_stocks=int(payload.get("short_mainline_min_concept_stocks", 2)),
                short_mainline_exclude_st=payload_bool(payload, "short_mainline_exclude_st", True),
                short_mainline_exclude_star=payload_bool(payload, "short_mainline_exclude_star", True),
                short_mainline_exclude_chinext=payload_bool(payload, "short_mainline_exclude_chinext", True),
                canslim_top_count=int(payload.get("canslim_top_count", payload.get("top_stock_count", 80))),
                canslim_channel_mode=str(payload.get("canslim_channel_mode", "canslim")).strip(),
                canslim_min_score=float(payload.get("canslim_min_score", 60)),
                canslim_min_rps=float(payload.get("canslim_min_rps", 90)),
                canslim_core_rps=float(payload.get("canslim_core_rps", 96)),
                canslim_near_high_ratio=float(payload.get("canslim_near_high_ratio", 0.85)),
                canslim_min_amount_yi=float(payload.get("canslim_min_amount_yi", 1)),
                canslim_min_market_cap_yi=float(payload.get("canslim_min_market_cap_yi", 100)),
                canslim_max_market_cap_yi=float(payload.get("canslim_max_market_cap_yi", 0)),
                canslim_min_turnover_rate=float(payload.get("canslim_min_turnover_rate", 0)),
                canslim_max_turnover_rate=float(payload.get("canslim_max_turnover_rate", 30)),
                canslim_min_eps_yoy=float(payload.get("canslim_min_eps_yoy", 25)),
                canslim_min_dt_eps_yoy=float(payload.get("canslim_min_dt_eps_yoy", 0)),
                canslim_min_profit_yoy=float(payload.get("canslim_min_profit_yoy", 20)),
                canslim_min_sales_yoy=float(payload.get("canslim_min_sales_yoy", 20)),
                canslim_min_annual_eps_yoy=float(payload.get("canslim_min_annual_eps_yoy", 20)),
                canslim_min_roe=float(payload.get("canslim_min_roe", 5)),
                canslim_mainline_enabled=payload_bool(payload, "canslim_mainline_enabled", True),
                canslim_mainline_industry_level=str(payload.get("canslim_mainline_industry_level", "sw_l2")).strip(),
                canslim_mainline_filter=str(payload.get("canslim_mainline_filter", "all")).strip(),
                canslim_mainline_min_score=float(payload.get("canslim_mainline_min_score", 60)),
                canslim_mainline_weight=float(payload.get("canslim_mainline_weight", 0.25)),
                tao_min_rps=float(payload.get("tao_min_rps", 90)),
                tao_core_rps=float(payload.get("tao_core_rps", payload.get("canslim_core_rps", 96))),
                tao_near_high_ratio=float(payload.get("tao_near_high_ratio", payload.get("canslim_near_high_ratio", 0.85))),
                tao_min_eps_yoy=float(payload.get("tao_min_eps_yoy", 0)),
                tao_min_profit_yoy=float(payload.get("tao_min_profit_yoy", 0)),
                tao_min_sales_yoy=float(payload.get("tao_min_sales_yoy", 0)),
                tao_min_roe=float(payload.get("tao_min_roe", 0)),
                tao_institution_enabled=payload_bool(payload, "tao_institution_enabled", True),
                tao_institution_mode=str(payload.get("tao_institution_mode", "any")).strip(),
                tao_fund_hold_enabled=payload_bool(payload, "tao_fund_hold_enabled", True),
                tao_fund_hold_min_ratio=float(payload.get("tao_fund_hold_min_ratio", 2)),
                tao_north_hold_enabled=payload_bool(payload, "tao_north_hold_enabled", True),
                tao_north_hold_min_value_wan=float(payload.get("tao_north_hold_min_value_wan", 3000)),
                tao_institution_missing_pass=payload_bool(payload, "tao_institution_missing_pass", False),
                canslim_include_fundamental=payload_bool(payload, "canslim_include_fundamental", True),
                canslim_exclude_st=payload_bool(payload, "canslim_exclude_st", True),
                canslim_exclude_star=payload_bool(payload, "canslim_exclude_star", True),
                canslim_exclude_chinext=payload_bool(payload, "canslim_exclude_chinext", True),
                value_top_count=int(payload.get("value_top_count", 100)),
                value_max_pb=float(payload.get("value_max_pb", 1)),
                value_max_pe_ttm=float(payload.get("value_max_pe_ttm", 15)),
                value_min_roe=float(payload.get("value_min_roe", 15)),
                value_min_fcf_yield=float(payload.get("value_min_fcf_yield", 4)),
                value_exclude_st=payload_bool(payload, "value_exclude_st", True),
                value_exclude_star=payload_bool(payload, "value_exclude_star", True),
                value_exclude_chinext=payload_bool(payload, "value_exclude_chinext", True),
                value_probe_query=str(payload.get("value_probe_query", "")).strip(),
                dividend_top_count=int(payload.get("dividend_top_count", 100)),
                dividend_min_yield=float(payload.get("dividend_min_yield", 3.5)),
                dividend_max_yield=float(payload.get("dividend_max_yield", 12)),
                dividend_min_market_cap_yi=float(payload.get("dividend_min_market_cap_yi", 150)),
                dividend_min_amount_yi=float(payload.get("dividend_min_amount_yi", 1)),
                dividend_max_volatility=float(payload.get("dividend_max_volatility", 30)),
                dividend_max_beta=float(payload.get("dividend_max_beta", 0.9)),
                dividend_max_drawdown=float(payload.get("dividend_max_drawdown", 25)),
                dividend_min_roe=float(payload.get("dividend_min_roe", 6)),
                dividend_max_pb=float(payload.get("dividend_max_pb", 2)),
                dividend_max_pe_ttm=float(payload.get("dividend_max_pe_ttm", 25)),
                dividend_max_distance_ma60=float(payload.get("dividend_max_distance_ma60", 18)),
                dividend_min_rs20=float(payload.get("dividend_min_rs20", -8)),
                dividend_min_ma60_slope=float(payload.get("dividend_min_ma60_slope", -8)),
                dividend_max_ret20=float(payload.get("dividend_max_ret20", 18)),
                dividend_exclude_st=payload_bool(payload, "dividend_exclude_st", True),
                dividend_exclude_star=payload_bool(payload, "dividend_exclude_star", True),
                dividend_exclude_chinext=payload_bool(payload, "dividend_exclude_chinext", True),
                dividend_probe_query=str(payload.get("dividend_probe_query", "")).strip(),
                stock_search_query=str(payload.get("stock_search_query", "")).strip(),
                stock_search_limit=int(payload.get("stock_search_limit", 500)),
                industry_market_cap_rank_level=str(payload.get("industry_market_cap_rank_level", "sw_l2")).strip(),
                industry_market_cap_rank_type=str(payload.get("industry_market_cap_rank_type", "industry_market_cap")).strip(),
                realtime_ma_input=str(payload.get("realtime_ma_input", "")).strip(),
                price_code_low_days=int(payload.get("price_code_low_days", 20)),
                price_code_volume_days=int(payload.get("price_code_volume_days", 5)),
                price_code_exclude_st=payload_bool(payload, "price_code_exclude_st", True),
                price_code_exclude_chinext=payload_bool(payload, "price_code_exclude_chinext", True),
                price_code_exclude_star=payload_bool(payload, "price_code_exclude_star", True),
                price_code_use_patterns=payload_bool(payload, "price_code_use_patterns", True),
                price_code_volume_filter=str(payload.get("price_code_volume_filter", "all")).strip(),
                price_code_volume_min_pct=price_code_volume_min_pct,
                price_code_volume_max_pct=price_code_volume_max_pct,
                price_code_industry_mv_rank_top=int(payload.get("price_code_industry_mv_rank_top", 5)),
                price_code_backtest_start_date=str(payload.get("price_code_backtest_start_date", "")).strip(),
                price_code_backtest_end_date=str(payload.get("price_code_backtest_end_date", "")).strip(),
                price_code_backtest_max_signal_days=int(payload.get("price_code_backtest_max_signal_days", 120)),
                price_code_backtest_codes=[
                    str(item).strip()
                    for item in payload.get("price_code_backtest_codes", [])
                    if str(item).strip()
                ]
                if isinstance(payload.get("price_code_backtest_codes", []), list)
                else [],
                price_code_scan_direction=str(payload.get("price_code_scan_direction", "both")).strip(),
                price_code_scan_min_pct=float(payload.get("price_code_scan_min_pct", 10)),
                price_code_scan_max_pct=float(payload.get("price_code_scan_max_pct", 100)),
                price_code_scan_step_pct=float(payload.get("price_code_scan_step_pct", 10)),
                price_code_scan_low_min_days=int(payload.get("price_code_scan_low_min_days", 10)),
                price_code_scan_low_max_days=int(payload.get("price_code_scan_low_max_days", 50)),
                price_code_scan_low_step_days=int(payload.get("price_code_scan_low_step_days", 10)),
                price_code_scan_industry_rank_min=int(payload.get("price_code_scan_industry_rank_min", 1)),
                price_code_scan_industry_rank_max=int(payload.get("price_code_scan_industry_rank_max", 5)),
                price_code_scan_industry_rank_step=int(payload.get("price_code_scan_industry_rank_step", 1)),
                price_code_scan_min_signals=int(payload.get("price_code_scan_min_signals", 30)),
                canslim_probe_query=str(payload.get("canslim_probe_query", "")).strip(),
                canslim_probe_only=payload_bool(payload, "canslim_probe_only", False),
                canslim_early_min_rps20=float(payload.get("canslim_early_min_rps20", 68)),
                canslim_early_min_rps20_delta10=float(payload.get("canslim_early_min_rps20_delta10", 8)),
                canslim_early_max_ret20=float(payload.get("canslim_early_max_ret20", 32)),
                canslim_early_max_ret60=float(payload.get("canslim_early_max_ret60", 80)),
                canslim_early_max_distance_ma20=float(payload.get("canslim_early_max_distance_ma20", 18)),
                canslim_early_max_distance_ma50=float(payload.get("canslim_early_max_distance_ma50", 38)),
                canslim_early_max_upper_shadow_days10=int(payload.get("canslim_early_max_upper_shadow_days10", 4)),
                canslim_early_max_stale_score=float(payload.get("canslim_early_max_stale_score", 58)),
                canslim_early_min_growth_accel_score=float(payload.get("canslim_early_min_growth_accel_score", 35)),
                concept_top_count=int(payload.get("concept_top_count", 80)),
                concept_rank_period=int(payload.get("concept_rank_period", 10)),
                concept_sort_field=str(payload.get("concept_sort_field", "ret10")).strip(),
                concept_keyword=str(payload.get("concept_keyword", "")).strip(),
                concept_member_warn_threshold=int(payload.get("concept_member_warn_threshold", 8)),
                concept_stock_limit_per_board=int(payload.get("concept_stock_limit_per_board", 120)),
                concept_recommend_count=int(payload.get("concept_recommend_count", 8)),
                concept_score_enabled=payload_bool(payload, "concept_score_enabled", False),
                concept_weight_trend=float(payload.get("concept_weight_trend", 0)),
                concept_weight_stability=float(payload.get("concept_weight_stability", 0)),
                concept_weight_breadth=float(payload.get("concept_weight_breadth", 0)),
                concept_weight_money=float(payload.get("concept_weight_money", 0)),
                concept_weight_leader=float(payload.get("concept_weight_leader", 0)),
                concept_weight_persistence=float(payload.get("concept_weight_persistence", 0)),
                concept_weight_canslim=float(payload.get("concept_weight_canslim", 0)),
                concept_weight_risk=float(payload.get("concept_weight_risk", 0)),
                concept_overheat_ret5=float(payload.get("concept_overheat_ret5", 15)),
                concept_overheat_ret10=float(payload.get("concept_overheat_ret10", 28)),
                concept_overheat_distance_ma20=float(payload.get("concept_overheat_distance_ma20", 15)),
                concept_low_breadth_ma20=float(payload.get("concept_low_breadth_ma20", 45)),
                concept_canslim_mode=str(payload.get("concept_canslim_mode", "off")).strip(),
                concept_exclude_st=payload_bool(payload, "concept_exclude_st", False),
                concept_exclude_star=payload_bool(payload, "concept_exclude_star", False),
                concept_exclude_chinext=payload_bool(payload, "concept_exclude_chinext", False),
                concept_exclude_bj=payload_bool(payload, "concept_exclude_bj", False),
                strong_concept_stock_days=int(payload.get("strong_concept_stock_days", 2)),
                strong_concept_concept_days=int(payload.get("strong_concept_concept_days", 2)),
                 strong_concept_stock_rank_mode=str(payload.get("strong_concept_stock_rank_mode", "percentile")).strip(),
                 strong_concept_stock_top_percent=float(payload.get("strong_concept_stock_top_percent", 10)),
                 strong_concept_stock_top_count=int(payload.get("strong_concept_stock_top_count", 100)),
                 strong_concept_industry_mv_enabled=payload_bool(payload, "strong_concept_industry_mv_enabled", False),
                 strong_concept_industry_mv_rank_top=int(payload.get("strong_concept_industry_mv_rank_top", 2)),
                 strong_concept_industry_rank_level=str(payload.get("strong_concept_industry_rank_level", "sw_l3")).strip(),
                 strong_concept_top_count=int(payload.get("strong_concept_top_count", 10)),
                strong_concept_market_scope=(
                    [str(item).strip() for item in payload.get("strong_concept_market_scope", ["main_board"]) if str(item).strip()]
                    if isinstance(payload.get("strong_concept_market_scope", ["main_board"]), list)
                    else [item.strip() for item in str(payload.get("strong_concept_market_scope", "main_board")).split(",") if item.strip()]
                ),
                strong_concept_exclude_st=payload_bool(payload, "strong_concept_exclude_st", True),
                strong_concept_exclude_star=payload_bool(payload, "strong_concept_exclude_star", True),
                strong_concept_exclude_chinext=payload_bool(payload, "strong_concept_exclude_chinext", True),
                strong_concept_exclude_bj=payload_bool(payload, "strong_concept_exclude_bj", True),
                strong_concept_member_limit=int(payload.get("strong_concept_member_limit", 1000)),
                strong_concept_recommend_count=int(payload.get("strong_concept_recommend_count", 10)),
                cross_momentum_days=int(payload.get("cross_momentum_days", 10)),
                cross_momentum_stock_rank_enabled=payload_bool(payload, "cross_momentum_stock_rank_enabled", True),
                cross_momentum_rank_mode=str(payload.get("cross_momentum_rank_mode", "count")).strip(),
                cross_momentum_top_count=int(payload.get("cross_momentum_top_count", 100)),
                cross_momentum_top_percent=float(payload.get("cross_momentum_top_percent", 10)),
                cross_momentum_concept_enabled=payload_bool(payload, "cross_momentum_concept_enabled", False),
                cross_momentum_concept_days=int(payload.get("cross_momentum_concept_days", 8)),
                cross_momentum_concept_rank_mode=str(payload.get("cross_momentum_concept_rank_mode", "count")).strip(),
                cross_momentum_concept_top_count=int(payload.get("cross_momentum_concept_top_count", 10)),
                cross_momentum_concept_top_percent=float(payload.get("cross_momentum_concept_top_percent", 10)),
                cross_momentum_market_scope=(
                    [str(item).strip() for item in payload.get("cross_momentum_market_scope", ["main_board"]) if str(item).strip()]
                    if isinstance(payload.get("cross_momentum_market_scope", ["main_board"]), list)
                    else [item.strip() for item in str(payload.get("cross_momentum_market_scope", "main_board")).split(",") if item.strip()]
                ),
                cross_momentum_exclude_st=payload_bool(payload, "cross_momentum_exclude_st", True),
                cross_momentum_exclude_star=payload_bool(payload, "cross_momentum_exclude_star", True),
                cross_momentum_exclude_chinext=payload_bool(payload, "cross_momentum_exclude_chinext", True),
                cross_momentum_exclude_bj=payload_bool(payload, "cross_momentum_exclude_bj", True),
                cross_momentum_limit_up_enabled=payload_bool(payload, "cross_momentum_limit_up_enabled", True),
                cross_momentum_limit_up_offset=int(payload.get("cross_momentum_limit_up_offset", -3)),
                cross_momentum_limit_not_up_enabled=payload_bool(payload, "cross_momentum_limit_not_up_enabled", True),
                cross_momentum_limit_not_up_offset=int(payload.get("cross_momentum_limit_not_up_offset", -2)),
                cross_momentum_low_enabled=payload_bool(payload, "cross_momentum_low_enabled", True),
                cross_momentum_low_offset=int(payload.get("cross_momentum_low_offset", -1)),
                cross_momentum_low_threshold=float(payload.get("cross_momentum_low_threshold", -9.5)),
                cross_momentum_open_enabled=payload_bool(payload, "cross_momentum_open_enabled", False),
                cross_momentum_open_offset=int(payload.get("cross_momentum_open_offset", 0)),
                cross_momentum_open_operator=str(payload.get("cross_momentum_open_operator", "gt")).strip(),
                cross_momentum_open_threshold=float(payload.get("cross_momentum_open_threshold", 0)),
                cross_momentum_realtime_enabled=payload_bool(payload, "cross_momentum_realtime_enabled", False),
                cross_momentum_realtime_auto_refresh=payload_bool(payload, "cross_momentum_realtime_auto_refresh", False),
                cross_momentum_realtime_interval=int(payload.get("cross_momentum_realtime_interval", 60)),
                cross_momentum_recommend_count=int(payload.get("cross_momentum_recommend_count", 10)),
                cross_momentum_backtest_days=int(payload.get("cross_momentum_backtest_days", 60)),
                cross_momentum_conditions=(
                    [item for item in payload["cross_momentum_conditions"] if isinstance(item, dict)]
                    if isinstance(payload.get("cross_momentum_conditions"), list)
                    else default_cross_momentum_conditions()
                ),
                backtest_strategy=str(payload.get("backtest_strategy", "canslim_tao")).strip(),
                backtest_start_date=str(payload.get("backtest_start_date", "")).strip(),
                backtest_end_date=str(payload.get("backtest_end_date", payload.get("end_date", ""))).strip(),
                backtest_initial_cash=float(payload.get("backtest_initial_cash", 1_000_000)),
                backtest_max_positions=int(payload.get("backtest_max_positions", 5)),
                backtest_max_entries_per_day=int(payload.get("backtest_max_entries_per_day", 2)),
                backtest_signal_top_n=int(payload.get("backtest_signal_top_n", 20)),
                backtest_max_hold_days=int(payload.get("backtest_max_hold_days", 10)),
                backtest_take_profit_pct=float(payload.get("backtest_take_profit_pct", 15)),
                backtest_stop_loss_pct=float(payload.get("backtest_stop_loss_pct", 8)),
                backtest_exit_ma=str(payload.get("backtest_exit_ma", "ma10")).strip(),
                backtest_fee_rate=float(payload.get("backtest_fee_rate", 0.0003)),
                backtest_sell_tax_rate=float(payload.get("backtest_sell_tax_rate", 0.0005)),
                backtest_slippage_bps=float(payload.get("backtest_slippage_bps", 5)),
                backtest_skip_limit_open=payload_bool(payload, "backtest_skip_limit_open", True),
                backtest_max_signal_days=int(payload.get("backtest_max_signal_days", 260)),
            )
            if not config.token:
                raise RuntimeError("缺少 Tushare Token")
            if not config.tushare_url:
                raise RuntimeError("缺少 Tushare API地址")
            if len(config.end_date) != 8 or not config.end_date.isdigit():
                raise RuntimeError("日期格式应为 YYYYMMDD")
            engine = TushareReviewEngine(config)
            if path == "/api/trade_date_shift":
                data = {
                    "ok": True,
                    "trade_date": engine.shift_trade_date(config.end_date, int(payload.get("direction", 0))),
                }
            elif path == "/api/backtest":
                data = engine.run_backtest()
            elif path == "/api/first_limit":
                data = engine.find_first_limit_up()
            elif path == "/api/first_wave":
                data = engine.run_first_wave()
            elif path == "/api/short_mainline":
                data = engine.run_short_mainline()
            elif path == "/api/one_to_two_live":
                data = engine.run_one_to_two_live()
            elif path == "/api/canslim_tao":
                data = engine.run_canslim_tao()
            elif path == "/api/canslim_early":
                data = engine.run_canslim_early()
            elif path == "/api/concept_radar":
                data = engine.run_concept_radar()
            elif path == "/api/strong_concept_momentum":
                data = engine.run_strong_concept_momentum()
            elif path == "/api/cross_sectional_momentum":
                data = engine.run_cross_sectional_momentum()
            elif path == "/api/cross_sectional_momentum_backtest":
                data = engine.run_cross_sectional_momentum_backtest()
            elif path == "/api/value_fcf":
                data = engine.run_value_fcf()
            elif path == "/api/value_fcf_calc":
                data = engine.calculate_value_fcf()
            elif path == "/api/dividend_low_vol":
                data = engine.run_dividend_low_vol()
            elif path == "/api/stock_search":
                data = engine.run_stock_search()
            elif path == "/api/industry_market_cap_rank":
                data = engine.run_industry_market_cap_rank()
            elif path == "/api/realtime_ma":
                data = engine.run_realtime_ma()
            elif path == "/api/etf_momentum":
                data = engine.run_etf_momentum()
            elif path == "/api/price_code_low":
                data = engine.run_price_code_low()
            elif path == "/api/price_code_low_backtest":
                data = engine.run_price_code_low_backtest()
            elif path == "/api/price_code_low_scan":
                data = engine.run_price_code_low_scan()
            elif path == "/api/market_overview":
                data = engine.market_overview()
            else:
                data = engine.run()
            self._send_json(data)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc), "trace": traceback.format_exc()}, 500)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False), "application/json; charset=utf-8")

    def _send(self, status: int, body: str, content_type: str, headers: dict[str, str] | None = None) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), ReviewHandler)
    print(f"Web review server: http://127.0.0.1:{PORT}")
    print(f"Phone/LAN URL: http://<your-computer-lan-ip>:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
