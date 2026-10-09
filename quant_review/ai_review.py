from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from html import unescape
from typing import Any

import requests


DEFAULT_OPENAI_BASE_URL = "https://mdkj.lol/v1/responses"
DEFAULT_OPENAI_MODEL = "gpt-5.5"
OPENAI_USER_AGENT = "sub2api-python-test/1.0"
DEFAULT_AI_CANDIDATES = 50
MAX_AI_CANDIDATES = 50
MAX_MESSAGE_WORKERS = 6
MAX_NEWS_PER_STOCK = 6
MAX_ANNOUNCEMENTS_PER_STOCK = 6
OPENAI_TIMEOUT_SECONDS = None
OPENAI_RETRY_DELAY_SECONDS = 120
OPENAI_CONNECTION_RETRY_SECONDS = 20
OPENAI_RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504, 524}
AI_SELECTION_WEIGHTS = {
    "消息作用": 35,
    "价格图形和买点": 30,
    "公司地位和基本面": 20,
    "行业主线": 10,
    "风险校验": 5,
}


AI_REVIEW_FIELDS = [
    "code",
    "name",
    "industry",
    "concepts",
    "setup",
    "channel",
    "score",
    "base_score",
    "mainline_industry",
    "mainline_state",
    "mainline_score",
    "industry_rank",
    "industry_rank_change_5",
    "industry_rps_combo",
    "industry_breadth_score",
    "industry_persistence_score",
    "rps_combo",
    "rps50",
    "rps120",
    "rps250",
    "c_score",
    "a_score",
    "n_score",
    "s_score",
    "l_score",
    "i_score",
    "m_score",
    "close",
    "pct_chg",
    "amount_yi",
    "amount_ratio",
    "turnover_rate_f",
    "market_cap_yi",
    "ret50",
    "ret120",
    "ret250",
    "high_position_250",
    "ma5",
    "ma10",
    "ma20",
    "distance_ma5",
    "distance_ma10",
    "distance_ma20",
    "distance_ma50",
    "ma_pullback",
    "ma_stack",
    "new_high_5",
    "pocket_pivot",
    "fund_hold_ratio",
    "north_hold_value_wan",
    "eps_yoy",
    "dt_eps_yoy",
    "basic_eps_yoy",
    "profit_yoy",
    "sales_yoy",
    "annual_eps_yoy",
    "annual_roe",
    "report_date",
    "finance_status",
    "finance_note",
    "tao_institution_note",
    "stop_hint",
    "reason",
]

CONCEPT_AI_REVIEW_FIELDS = [
    "code",
    "ts_code",
    "name",
    "industry",
    "stock_type",
    "source_concepts",
    "concept_name",
    "concept_state",
    "role",
    "stock_observe_score",
    "concept_stock_rank",
    "close",
    "pct_chg",
    "ret5",
    "ret10",
    "ret20",
    "ret30",
    "ret60",
    "rps5",
    "rps10",
    "rps20",
    "rps30",
    "rps60",
    "amount_yi",
    "amount_ratio",
    "turnover_rate_f",
    "market_cap_yi",
    "distance_ma5",
    "distance_ma10",
    "distance_ma20",
    "breakout20",
    "risk_flags",
    "canslim_pass",
    "canslim_score",
    "canslim_channel",
    "canslim_setup",
    "canslim_finance_status",
    "canslim_reason",
    "eps_yoy",
    "basic_eps_yoy",
    "dt_eps_yoy",
    "profit_yoy",
    "sales_yoy",
    "annual_eps_yoy",
    "annual_roe",
    "end_date",
    "note",
    "reason",
]


def run_canslim_ai_review(payload: dict[str, Any]) -> dict[str, Any]:
    api_key = str(payload.get("openai_api_key") or "").strip()
    if not api_key:
        raise ValueError("缺少 OpenAI API Key")

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("没有可供 AI 分析的候选股")

    review_mode = str(payload.get("review_mode") or "after_close").strip() or "after_close"
    candidate_limit = _bounded_int(payload.get("max_ai_candidates"), DEFAULT_AI_CANDIDATES, 1, MAX_AI_CANDIDATES)
    candidates = _compact_rows(rows[:candidate_limit])
    message_window = _message_date_window(str(payload.get("trade_date") or ""))
    candidates = _attach_real_messages(candidates, message_window)
    if review_mode == "intraday":
        candidates = _attach_realtime_snapshot(candidates)
    model = str(payload.get("openai_model") or DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL
    base_url = str(payload.get("openai_base_url") or DEFAULT_OPENAI_BASE_URL).strip() or DEFAULT_OPENAI_BASE_URL
    base_url = base_url.rstrip("/")

    prompt_payload = {
        "trade_date": str(payload.get("trade_date") or ""),
        "review_mode": review_mode,
        "analysis_target": str(payload.get("analysis_target") or "候选股").strip() or "候选股",
        "review_mode_note": (
            "盘中实时：candidates[].realtime 为额外抓取的实时行情快照，盘中价格、均线关系和买点都可能随行情变化。"
            if review_mode == "intraday"
            else "盘后复盘：主要使用正式日线/收盘数据，实时字段可忽略。"
        ),
        "candidate_count": len(rows),
        "analyzed_count": len(candidates),
        "market": _compact_context(payload.get("market") or {}),
        "financial": _compact_context(payload.get("financial") or {}),
        "institutional": _compact_context(payload.get("institutional") or {}),
        "mainline": _compact_context(payload.get("mainline") or {}),
        "ai_selection_weights": AI_SELECTION_WEIGHTS,
        "ai_review_policy": (
            "AI诊股是二次投研复核，不是主线硬过滤。主线确认只是加分项，主线候选必须加入讨论；"
            "若消息作用、价格图形、买点和公司质地更强，主线候选或暂未确认主线的股票可以排在主线确认股前面。"
            "排序时重点看真实新闻/公告的作用、收盘价与MA5/MA10/MA20关系、回踩/突破/偏离、止损线是否清楚。"
        ),
        "candidates": candidates,
        "data_scope_note": (
            "候选数据来自本地 CANSLIM+RPS 筛选结果，包含技术面、财务、机构、行业主线和产业题材。"
            "系统已经为每只候选股尝试抓取真实公告和东财新闻，并放在 candidates[].messages 中。"
            "消息面分析必须优先引用 messages 中的真实标题、摘要、日期和来源。"
            "没有抓到新闻/公告、抓取失败或源数据异常时必须标注数据不足，不能编造近期新闻、公告或研报。"
            "公司主营、行业地位和概念纯度可以结合已有概念、行业、财务数据和公开常识推断，"
            "但需要注明置信度和不确定性。"
            "本次尽量保留更多消息面，默认送入最多50只候选股，每只股票最多保留6条有效新闻和6条公告。"
        ),
    }
    user_prompt = json.dumps(prompt_payload, ensure_ascii=False, separators=(",", ":"))

    system_prompt = """你是一个A股CANSLIM+RPS复盘助手。你的任务不是直接给投资建议，而是对量化筛出的强势候选股做二次复核，选出3-5只最值得重点跟踪的股票。

你必须综合：
1. 技术面：RPS50/120/250、RPS综合、趋势形态、新高位置、收盘价与MA5/MA10/MA20/MA50、回踩/突破/偏离、量能、换手、风控线。若 review_mode=intraday，必须优先结合 candidates[].realtime 的实时价、涨跌幅、今高今低、成交额判断盘中买点。
2. 基本面：EPS、扣非EPS、利润、营收、ROE、财报期、财务数据是否缺失。
3. 行业和主线：申万/基础行业、行业RPS、行业广度、主线状态、同组竞争位置。主线确认、主线候选、主线暂未确认都要纳入讨论；强图形、好买点、消息作用明确的股票，不要仅因为尚未“主线确认”而剔除或降到最后。
4. 公司画像：主营业务、核心产品/服务、产业链环节、行业地位、竞争优势、概念纯度、题材和业绩匹配度。
5. 消息面和催化：必须读取 candidates[].messages.news 和 candidates[].messages.announcements；引用真实标题/日期/来源进行判断，并判断消息的作用是“验证主线、改善基本面、带来订单/业绩弹性、纯题材刺激、风险暴露或噪音”。没有抓到实时新闻或公告时必须写“消息面数据不足”，不能杜撰。
6. 风险：过热、追高、解禁/减持/监管/业绩变脸等无法确认时要写明“未接入相关数据，需人工核验”。

AI诊股排序权重必须按 ai_selection_weights 执行：消息作用35%，价格图形和买点30%，公司地位和基本面20%，行业主线10%，风险校验5%。主线确认只是加分项，不是门槛；主线候选必须进入讨论，且可以入选top_picks。不得用“主线确认/主线分最高”作为唯一排序理由。图形上优先选择刚突破、回踩5/10/20日线后收回、距20日线不过分偏离、止损线清晰的股票；距MA20过远且无强消息支撑的股票要标注“过热不追”或等待回踩。

消息面必须判断“作用”，不能只罗列标题：请明确它是验证行业主线、改善基本面、订单/项目落地、产能/价格/政策催化、机构关注、纯题材刺激、负面风险，还是与公司弱相关的噪音。价格图形必须判断“买点”，不能只说RPS高：请明确当前是突破买点、回踩5日/10日/20日线买点、等突破、等回踩、过热不追，还是跌破关键线失效。盘中实时模式下，必须说明盘中信号是临时快照，需要收盘确认。

如果 analysis_target=单股分析：不要强行凑3-5只，top_picks只输出这1只股票；重点解释它是否符合CANSLIM/RPS、是否只是主线候选、消息作用、当前买点、盘中/盘后风控线和下一步观察动作。如果单股不适合买，也要在 action 中明确写“等回踩/等突破/过热不追/剔除观察”等。

输出必须是一个JSON对象，不要Markdown，不要代码块。JSON结构：
{
  "summary": "整体结论，80字内",
  "selection_method": "必须说明本次按消息35%、图形买点30%、公司基本面20%、主线10%、风险5%优选，120字内",
  "top_picks": [
    {
      "rank": 1,
      "code": "股票代码",
      "name": "股票名称",
      "ai_score": 0-100,
      "confidence": "高/中/低",
      "action": "重点观察/回踩可看/等5日线回踩/等10日线回踩/等20日线回踩/等突破/过热不追/剔除",
      "company_profile": {
        "main_business": "主营与核心产品",
        "industry_position": "行业地位和竞争优势",
        "concept_purity": "概念纯度与业绩匹配度"
      },
      "technical_view": "技术面结论",
      "price_ma_view": "收盘价与MA5/MA10/MA20、回踩/突破/偏离、买点质量和失效条件",
      "fundamental_view": "基本面结论",
      "industry_view": "主线和行业地位结论；必须说明主线确认/候选/未确认如何影响但不硬过滤",
      "catalyst_view": "消息面/催化结论，数据不足要明说",
      "message_impact": "消息的作用和强弱，必须引用真实新闻/公告标题、日期或说明消息面数据不足",
      "risk_view": "主要风险",
      "watch_levels": "观察线、等待条件或失效条件",
      "reason": "入选核心理由"
    }
  ],
  "watchlist": [
    {"code":"股票代码","name":"股票名称","reason":"候补原因","action":"观察动作"}
  ],
  "rejected": [
    {"code":"股票代码","name":"股票名称","reason":"剔除原因"}
  ],
  "data_gaps": ["本次缺失或需要人工核验的数据"],
  "disclaimer": "仅用于复盘观察，不构成投资建议"
}

候选池优选时 top_picks 必须是3到5只；如果候选股少于3只，可以少于3只但要说明原因。单股分析时 top_picks 只输出该单股。优先选择消息作用明确、价格图形健康、买点清晰、基本面和公司地位不差的标的；主线强优先，但主线候选也可以入选，甚至在消息和买点更优时排在主线确认股前面。"""

    text = _call_openai_responses(base_url, api_key, model, system_prompt, user_prompt)
    parsed = _parse_json_object(text)
    return {
        "ok": True,
        "model": model,
        "candidate_count": len(rows),
        "analyzed_count": len(candidates),
        "candidate_limit": candidate_limit,
        "ai_review": parsed,
        "raw_text": "" if parsed else text,
    }


def run_concept_ai_review(payload: dict[str, Any]) -> dict[str, Any]:
    api_key = str(payload.get("openai_api_key") or "").strip()
    if not api_key:
        raise ValueError("缺少 OpenAI API Key")

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("没有可供 AI 分析的自选股")

    candidate_limit = _bounded_int(payload.get("max_ai_candidates"), 30, 1, 50)
    candidates = _compact_concept_rows(rows[:candidate_limit])
    message_window = _message_date_window(str(payload.get("trade_date") or ""))
    candidates = _attach_real_messages(candidates, message_window)

    model = str(payload.get("openai_model") or DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL
    base_url = str(payload.get("openai_base_url") or DEFAULT_OPENAI_BASE_URL).strip() or DEFAULT_OPENAI_BASE_URL
    base_url = base_url.rstrip("/")

    prompt_payload = {
        "trade_date": str(payload.get("trade_date") or ""),
        "analysis_target": str(payload.get("analysis_target") or "概念自选池").strip() or "概念自选池",
        "concept_context": _compact_context(payload.get("concept_context") or {}),
        "candidate_count": len(rows),
        "analyzed_count": len(candidates),
        "candidates": candidates,
        "analysis_policy": (
            "这是复盘研究，不是投资建议。请重点分析板块上涨逻辑是否真实、公司主营、行业/概念地位、概念纯度、"
            "近期新闻和公告催化的新鲜度、题材与主营是否匹配、风险和反证，以及后续适合观察什么。"
            "不要编造新闻；没有抓到新闻时必须说明消息面数据不足。"
        ),
    }
    user_prompt = json.dumps(prompt_payload, ensure_ascii=False, separators=(",", ":"))
    system_prompt = """你是A股概念复盘研究助手。用户会把来自不同概念板块的股票放入一个研究自选池，你需要基于候选数据、真实新闻/公告标题和日期，做二次研究整理。

你必须做到：
1. 公司主营：说明公司主要业务、产品或服务，无法确认时写“不确定，需要人工核验”。
2. 公司地位：判断它在来源概念里更像龙头、中军、弹性、跟风，还是只是概念沾边。
3. 概念纯度：判断主营、公告、新闻与来源概念是否强相关。
4. 板块逻辑：说明来源概念为什么涨，判断是政策/产业事件/涨价/业绩/订单/短线情绪，还是仅资金炒作；无法确认时写“板块催化需要人工核验”。
5. 新闻催化：优先引用 candidates[].messages 中真实新闻/公告的标题、日期、来源；判断近3日/10日/30日/旧逻辑；没有则明确“消息面数据不足”。
6. 技术位置：只做复盘解释，结合RPS、5/10/20/30/60日表现、均线偏离、成交额、风险提示，给出“重点观察/等回踩/等突破/过热不追/逻辑较弱”等观察动作。
7. 风险和反证：主动寻找主营不纯、旧闻、公告澄清、涨幅过热、业绩下滑、减持解禁、监管诉讼、后排跟风等反证；无法确认的数据要标注需要人工核验。

输出必须是JSON对象，不要Markdown，不要代码块。结构：
{
  "summary": "80字内整体结论",
  "selection_method": "说明本次根据主营地位、新闻催化、概念纯度、技术位置和风险做研究排序，120字内",
  "top_picks": [
    {
      "rank": 1,
      "code": "股票代码",
      "name": "股票名称",
      "ai_score": 0-100,
      "confidence": "高/中/低",
      "action": "重点观察/等回踩/等突破/过热不追/逻辑较弱/剔除观察",
      "company_profile": {
        "main_business": "主营业务",
        "industry_position": "行业或概念地位",
        "concept_purity": "概念纯度"
      },
      "sector_view": "板块上涨逻辑真实性、催化类型和持续性",
      "concept_view": "来源概念和板块地位分析",
      "news_view": "新闻公告催化及其作用，必须引用真实标题/日期或说明数据不足",
      "evidence_freshness": "近3日/10日/30日/旧逻辑/消息不足",
      "technical_view": "RPS、涨幅、均线、成交和当前位置",
      "risk_view": "主要风险",
      "counter_evidence": "反证或需要人工核验的疑点",
      "watch_levels": "后续观察条件或失效条件",
      "reason": "入选或排序理由"
    }
  ],
  "watchlist": [
    {"code":"股票代码","name":"股票名称","reason":"候补原因","action":"观察动作"}
  ],
  "rejected": [
    {"code":"股票代码","name":"股票名称","reason":"逻辑弱或风险原因"}
  ],
  "data_gaps": ["缺失或需要人工核验的数据"],
  "disclaimer": "仅用于复盘研究，不构成投资建议"
}

top_picks通常给3-8只；如果自选股数量更少，可以少于3只。不得编造新闻、公告、主营或行业地位。"""

    text = _call_openai_responses(base_url, api_key, model, system_prompt, user_prompt)
    parsed = _parse_json_object(text)
    return {
        "ok": True,
        "model": model,
        "candidate_count": len(rows),
        "analyzed_count": len(candidates),
        "candidate_limit": candidate_limit,
        "ai_review": parsed,
        "raw_text": "" if parsed else text,
    }


def _compact_rows(rows: list[Any]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = {key: _compact_field(key, row.get(key)) for key in AI_REVIEW_FIELDS if key in row}
        compacted.append(item)
    return compacted


def _compact_concept_rows(rows: list[Any]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = {key: _compact_field(key, row.get(key)) for key in CONCEPT_AI_REVIEW_FIELDS if key in row}
        compacted.append(item)
    return compacted


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _compact_field(key: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    limits = {
        "concepts": 120,
        "source_concepts": 140,
        "concept_name": 80,
        "concept_state": 40,
        "mainline_reason": 100,
        "finance_note": 120,
        "tao_institution_note": 120,
        "stop_hint": 100,
        "reason": 140,
        "canslim_reason": 140,
        "note": 120,
    }
    return _limit_text(value, limits.get(key, 80))


def _compact_context(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, list):
                result[key] = [_compact_context(x) for x in item[:8]]
            else:
                result[key] = _compact_context(item)
        return result
    if isinstance(value, list):
        return [_compact_context(x) for x in value[:8]]
    if isinstance(value, str):
        return _limit_text(value, 160)
    return value


def _message_date_window(trade_date: str) -> tuple[str, str]:
    try:
        end = datetime.strptime(str(trade_date)[:8], "%Y%m%d").date()
    except (TypeError, ValueError):
        end = date.today()
    start = end - timedelta(days=180)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _attach_real_messages(candidates: list[dict[str, Any]], date_window: tuple[str, str]) -> list[dict[str, Any]]:
    if not candidates:
        return candidates

    def enrich(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        code = _clean_symbol(row.get("code"))
        return code, _fetch_stock_messages(code, str(row.get("name") or ""), date_window)

    code_to_messages: dict[str, dict[str, Any]] = {}
    unique_rows = []
    seen = set()
    for row in candidates:
        code = _clean_symbol(row.get("code"))
        if code and code not in seen:
            seen.add(code)
            unique_rows.append(row)

    with ThreadPoolExecutor(max_workers=min(MAX_MESSAGE_WORKERS, max(1, len(unique_rows)))) as pool:
        futures = [pool.submit(enrich, row) for row in unique_rows]
        for future in as_completed(futures):
            try:
                code, messages = future.result()
            except Exception as exc:
                code, messages = "", {"status": "抓取失败", "notes": [str(exc)], "news": [], "announcements": []}
            if code:
                code_to_messages[code] = messages

    for row in candidates:
        code = _clean_symbol(row.get("code"))
        row["messages"] = code_to_messages.get(code, {"status": "无代码", "notes": ["股票代码缺失"], "news": [], "announcements": []})
    return candidates


def _attach_realtime_snapshot(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    codes = []
    seen = set()
    for row in candidates:
        code = _clean_symbol(row.get("code"))
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    snapshots = _fetch_eastmoney_realtime(codes)
    snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for row in candidates:
        code = _clean_symbol(row.get("code"))
        realtime = snapshots.get(code)
        if realtime:
            row["realtime"] = {"snapshot_time": snapshot_time, **realtime}
        else:
            row["realtime"] = {"snapshot_time": snapshot_time, "status": "实时行情缺失"}
    return candidates


def _fetch_eastmoney_realtime(codes: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not codes:
        return result
    session = requests.Session()
    session.trust_env = False
    for chunk in _chunks(codes, 80):
        secids = ",".join(_eastmoney_secid(code) for code in chunk)
        try:
            response = session.get(
                "https://push2.eastmoney.com/api/qt/ulist.np/get",
                params={
                    "fltt": "2",
                    "invt": "2",
                    "fields": "f12,f14,f2,f3,f4,f5,f6,f8,f15,f16,f17,f18",
                    "secids": secids,
                },
                timeout=15,
            )
            response.raise_for_status()
            items = (response.json().get("data") or {}).get("diff") or []
        except Exception:
            continue
        for item in items:
            code = str(item.get("f12", "")).zfill(6)
            latest = _to_float(item.get("f2"))
            pre_close = _to_float(item.get("f18"))
            result[code] = {
                "status": "实时行情",
                "source": "东方财富实时",
                "latest_price": latest,
                "pct_chg": _to_float(item.get("f3")),
                "price_chg": _to_float(item.get("f4")),
                "open": _to_float(item.get("f17")),
                "high": _to_float(item.get("f15")),
                "low": _to_float(item.get("f16")),
                "pre_close": pre_close,
                "turnover_rate": _to_float(item.get("f8")),
                "volume_hand": _to_float(item.get("f5")),
                "amount_yi": round((_to_float(item.get("f6")) or 0) / 100000000, 2),
            }
    return result


def _fetch_stock_messages(code: str, name: str, date_window: tuple[str, str]) -> dict[str, Any]:
    notes: list[str] = []
    news = _fetch_eastmoney_news(code, name, notes)
    announcements = _fetch_cninfo_announcements(code, date_window, notes)
    status_parts = []
    status_parts.append(f"新闻{len(news)}条" if news else "新闻缺失")
    status_parts.append(f"公告{len(announcements)}条" if announcements else "公告缺失")
    return {
        "code": code,
        "name": name,
        "window": {"start": date_window[0], "end": date_window[1]},
        "status": "；".join(status_parts),
        "notes": [_limit_text(note, 90) for note in notes[:3]],
        "news": news,
        "announcements": announcements,
    }


def _fetch_eastmoney_news(code: str, name: str, notes: list[str]) -> list[dict[str, Any]]:
    if not code:
        return []
    try:
        import akshare as ak

        frame = ak.stock_news_em(symbol=code)
    except Exception as exc:
        notes.append(f"东财新闻抓取失败: {exc}")
        return []
    if frame is None or frame.empty:
        notes.append("东财新闻为空")
        return []

    rows = []
    for _, item in frame.head(MAX_NEWS_PER_STOCK * 2).iterrows():
        values = list(item.values)
        title = _clean_message_text(values[1] if len(values) > 1 else "")
        summary = _clean_message_text(values[2] if len(values) > 2 else "")
        publish_time = _clean_message_text(values[3] if len(values) > 3 else "")
        source = _clean_message_text(values[4] if len(values) > 4 else "")
        if not _is_readable_message(title):
            continue
        relevance = _news_relevance(code, name, title, summary)
        if relevance == "弱相关":
            continue
        rows.append(
            {
                "date": publish_time,
                "source": source or "东方财富",
                "relevance": relevance,
                "title": _limit_text(title, 90),
                "summary": _limit_text(summary, 120),
            }
        )
        if len(rows) >= MAX_NEWS_PER_STOCK:
            break
    if not rows:
        notes.append("东财新闻源返回内容不可读或无有效标题")
    return rows


def _news_relevance(code: str, name: str, title: str, summary: str) -> str:
    haystack_title = f"{title}"
    haystack_all = f"{title} {summary}"
    clean_name = str(name or "").strip()
    if clean_name and clean_name in haystack_title:
        return "标题相关"
    if code and code in haystack_title:
        return "标题相关"
    if clean_name and clean_name in haystack_all:
        return "正文提及"
    if code and code in haystack_all:
        return "正文提及"
    return "弱相关"


def _fetch_cninfo_announcements(code: str, date_window: tuple[str, str], notes: list[str]) -> list[dict[str, Any]]:
    if not code:
        return []
    stock_item = _cninfo_stock_item(code)
    if not stock_item:
        notes.append("巨潮公告暂不支持该市场代码")
        return []
    payload = {
        "pageNum": "1",
        "pageSize": str(MAX_ANNOUNCEMENTS_PER_STOCK),
        "column": "szse",
        "tabName": "fulltext",
        "plate": "",
        "stock": stock_item,
        "searchkey": "",
        "secid": "",
        "category": "",
        "trade": "",
        "seDate": f"{_format_date(date_window[0])}~{_format_date(date_window[1])}",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    try:
        response = requests.post(
            "http://www.cninfo.com.cn/new/hisAnnouncement/query",
            data=payload,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
            },
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        notes.append(f"巨潮公告抓取失败: {exc}")
        return []

    rows = []
    for item in data.get("announcements") or []:
        title = _clean_message_text(item.get("announcementTitle", ""))
        if not _is_readable_message(title):
            continue
        publish_time = _cninfo_time(item.get("announcementTime"))
        rows.append(
            {
                "date": publish_time,
                "source": "巨潮资讯",
                "title": _limit_text(title, 100),
            }
        )
    if not rows:
        notes.append("巨潮公告为空或公告标题不可读")
    return rows[:MAX_ANNOUNCEMENTS_PER_STOCK]


def _clean_symbol(value: Any) -> str:
    text = str(value or "").strip()
    match = re.search(r"\d{6}", text)
    return match.group(0) if match else ""


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _eastmoney_secid(code: str) -> str:
    symbol = str(code or "").zfill(6)
    market = "1" if symbol.startswith(("6", "9")) else "0"
    return f"{market}.{symbol}"


def _to_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text or text in {"-", "None", "nan"}:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _cninfo_stock_item(code: str) -> str:
    if code.startswith(("6", "9")):
        return f"{code},gssh0{code}"
    if code.startswith(("0", "2", "3")):
        return f"{code},gssz0{code}"
    return ""


def _format_date(value: str) -> str:
    text = str(value or "")
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def _cninfo_time(value: Any) -> str:
    try:
        number = int(value)
        return datetime.fromtimestamp(number / 1000).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return _clean_message_text(value)


def _clean_message_text(value: Any) -> str:
    text = unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text.replace("\u3000", " ")).strip()
    return text


def _limit_text(value: Any, limit: int) -> str:
    text = _clean_message_text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _is_readable_message(text: str) -> bool:
    if not text or len(text) < 4:
        return False
    replacement_ratio = text.count("\ufffd") / max(1, len(text))
    if replacement_ratio > 0.08:
        return False
    chinese_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_count = sum(1 for ch in text if ch.isascii() and ch.isalnum())
    return chinese_count >= 2 or ascii_count >= 6


def _call_openai_responses(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    combined_input = f"{system_prompt}\n\n候选数据JSON：\n{user_prompt}"
    request_payload = {
        "model": model,
        "input": combined_input,
        "store": False,
    }
    responses_url, chat_url = _openai_endpoint_urls(base_url)
    responses_error = ""
    try:
        data = _post_json(responses_url, api_key, request_payload)
        return _extract_responses_text(data)
    except RuntimeError as exc:
        responses_error = str(exc)
        # Use chat fallback only for endpoint compatibility errors.
        # Cloudflare/proxy timeouts should be surfaced directly so we do not wait again.
        if not _should_try_chat_fallback(responses_error):
            raise

    chat_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 5000,
    }
    try:
        data = _post_json(chat_url, api_key, chat_payload)
        return _extract_chat_text(data)
    except RuntimeError as exc:
        raise RuntimeError(f"OpenAI接口调用失败。Responses错误: {responses_error}；Chat备用错误: {exc}") from exc


def _should_try_chat_fallback(message: str) -> bool:
    text = str(message or "").lower()
    if any(marker in text for marker in ["401", "403", "invalid api key", "unauthorized", "forbidden"]):
        return False
    return any(
        marker in text
        for marker in [
            "404",
            "unsupported",
            "not found",
            "endpoint",
            "no route",
        ]
    )


def _openai_endpoint_urls(base_url: str) -> tuple[str, str]:
    url = str(base_url or DEFAULT_OPENAI_BASE_URL).strip().rstrip("/")
    if url.endswith("/responses"):
        base = url[: -len("/responses")]
        return url, f"{base}/chat/completions"
    if url.endswith("/chat/completions"):
        base = url[: -len("/chat/completions")]
        return f"{base}/responses", url
    return f"{url}/responses", f"{url}/chat/completions"


def _post_json(url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    session = requests.Session()
    session.trust_env = False
    while True:
        try:
            response = session.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": OPENAI_USER_AGENT,
                },
                json=payload,
                timeout=OPENAI_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            break
        except requests.HTTPError as exc:
            response = exc.response
            status = response.status_code if response is not None else 0
            if status in OPENAI_RETRYABLE_HTTP_STATUSES:
                time.sleep(_retry_after_seconds(response, OPENAI_RETRY_DELAY_SECONDS))
                continue
            try:
                detail = response.text if response is not None else ""
                error_obj = json.loads(detail)
                message = error_obj.get("error", {}).get("message") or detail
            except Exception:
                message = str(exc)
            raise RuntimeError(f"OpenAI接口返回HTTP {status or '未知'}: {message}") from exc
        except requests.RequestException as exc:
            if _is_retryable_connection_error(exc):
                time.sleep(OPENAI_CONNECTION_RETRY_SECONDS)
                continue
            raise RuntimeError(f"OpenAI接口连接失败: {exc}") from exc
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError("OpenAI接口返回了无法解析的JSON") from exc


def _retry_after_seconds(response: requests.Response | None, default: int) -> int:
    if response is None:
        return default
    header = response.headers.get("Retry-After") or response.headers.get("retry-after")
    try:
        value = int(float(str(header).strip()))
        return max(5, min(600, value))
    except (TypeError, ValueError):
        return default


def _is_retryable_connection_error(exc: requests.RequestException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in [
            "remote end closed",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "timed out",
            "timeout",
        ]
    )


def _extract_responses_text(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    parts: list[str] = []
    for output in data.get("output", []) or []:
        if not isinstance(output, dict):
            continue
        for content in output.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text") or content.get("content")
            if isinstance(text, str):
                parts.append(text)
    if parts:
        return "\n".join(parts)
    raise RuntimeError("OpenAI接口没有返回文本内容")


def _extract_chat_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI接口没有返回候选内容")
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenAI接口没有返回文本内容")
    return content


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
