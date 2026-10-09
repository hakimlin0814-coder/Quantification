from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import tushare as ts

from quant_review.config import ReviewConfig
from quant_review.backtest import BacktestEngine
from quant_review.strategies.canslim_early import CanslimEarlyStrategy
from quant_review.strategies.canslim_tao import CanslimTaoStrategy
from quant_review.strategies.concept_radar import ConceptRadarStrategy
from quant_review.strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from quant_review.strategies.dividend_low_vol import DividendLowVolStrategy
from quant_review.strategies.etf_momentum import EtfMomentumStrategy
from quant_review.strategies.first_limit import FirstLimitStrategy
from quant_review.strategies.first_wave import FirstWaveStrategy
from quant_review.strategies.industry_market_cap_rank import IndustryMarketCapRankStrategy
from quant_review.strategies.mainline_review import MainlineReviewStrategy
from quant_review.strategies.one_to_two_live import OneToTwoLiveStrategy
from quant_review.strategies.oneil_model import OneilModelStrategy
from quant_review.strategies.price_code_low import PriceCodeLowStrategy
from quant_review.strategies.realtime_ma import RealtimeMaStrategy
from quant_review.strategies.short_mainline import ShortMainlineStrategy
from quant_review.strategies.stock_search import StockSearchStrategy
from quant_review.strategies.strong_concept_momentum import StrongConceptMomentumStrategy
from quant_review.strategies.value_fcf import ValueFcfStrategy


CACHE_DIR = Path(__file__).resolve().parent.parent / "review_cache_web"
CACHE_DIR.mkdir(exist_ok=True)

FINA_INDICATOR_FIELDS = (
    "ts_code,ann_date,end_date,eps,dt_eps,q_eps,basic_eps_yoy,dt_eps_yoy,"
    "roe,roe_dt,or_yoy,q_sales_yoy,netprofit_yoy,dt_netprofit_yoy,"
    "q_profit_yoy,q_netprofit_yoy,grossprofit_margin"
)
CASHFLOW_FIELDS = "ts_code,ann_date,end_date,n_cashflow_act,c_pay_acq_const_fiolta,free_cashflow"

class TushareReviewEngine:
    """复盘核心引擎。

    职责分三层：
    1. 读取数据：Tushare 日线、daily_basic、涨跌停、资金流、实时行情。
    2. 计算信号：市场状态、一级概念强度、个股评分和角色。
    3. 返回结果：给前端页面展示和导出 CSV。
    """

    NON_INDUSTRY_CONCEPT_KEYWORDS = (
        "融资融券",
        "转融券",
        "融券",
        "沪股通",
        "深股通",
        "港股通",
        "陆股通",
        "MSCI",
        "HS300",
        "富时罗素",
        "标准普尔",
        "标普",
        "标普道琼斯",
        "沪深300",
        "中证",
        "上证",
        "深证",
        "深成",
        "创业板综",
        "科创板",
        "昨日",
        "今日",
        "近期",
        "本周",
        "连板",
        "涨停",
        "跌停",
        "高振幅",
        "强势",
        "破发",
        "破净",
        "次新股",
        "新股",
        "送转",
        "填权",
        "摘帽",
        "ST板块",
        "预盈",
        "预增",
        "预减",
        "预亏",
        "年报",
        "中报",
        "季报",
        "半年报",
        "业绩",
        "财报",
        "高送转",
        "机构重仓",
        "基金重仓",
        "社保重仓",
        "证金持股",
        "QFII",
        "养老金",
        "大盘股",
        "小盘股",
        "大盘成长",
        "中盘成长",
        "小盘成长",
        "大盘价值",
        "中盘价值",
        "小盘价值",
        "百元股",
        "低价股",
        "高价股",
        "低市盈率",
        "高股息",
        "中特估",
        "中字头",
        "含H股",
    )

    def __init__(self, config: ReviewConfig) -> None:
        self.config = config
        ts.set_token(config.token)
        self.pro = ts.pro_api(config.token)
        if config.tushare_url.strip():
            self.pro._DataApi__http_url = config.tushare_url.strip()
        self.logs: list[str] = []
        self._last_realtime_snapshot: dict[str, Any] = {}

    def log(self, message: str) -> None:
        self.logs.append(f"{datetime.now():%H:%M:%S} {message}")

    def run(self) -> dict[str, Any]:
        return MainlineReviewStrategy(self).run()

    def market_overview(self) -> dict[str, Any]:
        return MainlineReviewStrategy(self).market_overview()

    def find_first_limit_up(self) -> dict[str, Any]:
        return FirstLimitStrategy(self).run()

    def run_first_wave(self) -> dict[str, Any]:
        return FirstWaveStrategy(self).run()

    def run_short_mainline(self) -> dict[str, Any]:
        return ShortMainlineStrategy(self).run()

    def run_one_to_two_live(self) -> dict[str, Any]:
        return OneToTwoLiveStrategy(self).run()

    def run_canslim_tao(self) -> dict[str, Any]:
        return CanslimTaoStrategy(self).run()

    def run_canslim_early(self) -> dict[str, Any]:
        return CanslimEarlyStrategy(self).run()

    def run_oneil_model(self, research_config) -> dict[str, Any]:
        """Run the independent Q0 O'Neil research screen (06/09 design)."""
        return OneilModelStrategy(self, research_config).run()

    def run_concept_radar(self) -> dict[str, Any]:
        return ConceptRadarStrategy(self).run()

    def run_strong_concept_momentum(self) -> dict[str, Any]:
        return StrongConceptMomentumStrategy(self).run()

    def run_cross_sectional_momentum(self) -> dict[str, Any]:
        return CrossSectionalMomentumStrategy(self).run()

    def run_cross_sectional_momentum_backtest(self) -> dict[str, Any]:
        return CrossSectionalMomentumStrategy(self).run_backtest()

    def run_value_fcf(self) -> dict[str, Any]:
        return ValueFcfStrategy(self).run()

    def calculate_value_fcf(self) -> dict[str, Any]:
        return ValueFcfStrategy(self).calculate_single_fcf()

    def run_dividend_low_vol(self) -> dict[str, Any]:
        return DividendLowVolStrategy(self).run()

    def run_stock_search(self) -> dict[str, Any]:
        return StockSearchStrategy(self).run()

    def run_industry_market_cap_rank(self) -> dict[str, Any]:
        return IndustryMarketCapRankStrategy(self).run()

    def run_realtime_ma(self) -> dict[str, Any]:
        return RealtimeMaStrategy(self).run()

    def run_etf_momentum(self) -> dict[str, Any]:
        return EtfMomentumStrategy(self).run()

    def run_price_code_low(self) -> dict[str, Any]:
        return PriceCodeLowStrategy(self).run()

    def run_price_code_low_backtest(self) -> dict[str, Any]:
        return PriceCodeLowStrategy(self).run_backtest()

    def run_price_code_low_scan(self) -> dict[str, Any]:
        return PriceCodeLowStrategy(self).run_parameter_scan()

    def run_backtest(self) -> dict[str, Any]:
        return BacktestEngine(self).run()

    def _cache_path(self, name: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
        return CACHE_DIR / f"{safe}.csv"

    def _cached(self, name: str, fetcher) -> pd.DataFrame:
        """读取缓存，缓存不存在时才调用接口。

        Tushare 接口有调用频率和积分成本，复盘时同一天的数据不需要反复拉取。
        所以每个接口结果都会保存到 review_cache_web 目录。
        """
        path = self._cache_path(name)
        if path.exists():
            try:
                return pd.read_csv(path, dtype=str)
            except pd.errors.EmptyDataError:
                self.log(f"{path.name} 缓存为空，已删除并重新读取。")
                path.unlink(missing_ok=True)
        df = fetcher()
        if df is None:
            return pd.DataFrame()
        if df.empty and len(df.columns) == 0:
            self.log(f"{name} 接口返回空表，暂不写入缓存。")
            return df
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return df

    def _trade_dates(self) -> list[str]:
        """获取复盘日期之前的最近交易日列表。

        如果用户填的是周末或节假日，会自动取之前最近的开市日。
        lookback_days 控制后面计算 5/10/20 日涨幅、均线、量能比时需要的历史窗口。
        """
        end = self.config.end_date or date.today().strftime("%Y%m%d")
        calendar_days = max(120, int(self.config.lookback_days * 2.2) + 10)
        start = (datetime.strptime(end, "%Y%m%d").date() - timedelta(days=calendar_days)).strftime("%Y%m%d")
        cal = self._cached(
            f"trade_cal_{start}_{end}",
            lambda: self.pro.trade_cal(exchange="", start_date=start, end_date=end, is_open="1"),
        )
        dates = sorted(cal["cal_date"].astype(str).tolist())
        dates = [item for item in dates if item <= end]
        if len(dates) < 25:
            raise RuntimeError("交易日数据不足，请检查 Tushare token 或日期。")
        return dates[-self.config.lookback_days :]

    def shift_trade_date(self, base_date: str, direction: int) -> str:
        """返回基准日相邻的真实交易日，自动跨过周末和节假日。"""
        try:
            base = datetime.strptime(str(base_date), "%Y%m%d").date()
        except (TypeError, ValueError) as exc:
            raise RuntimeError("日期格式应为 YYYYMMDD") from exc
        if direction not in {-1, 1}:
            raise RuntimeError("交易日切换方向无效")

        start = (base - timedelta(days=45)).strftime("%Y%m%d")
        end = (base + timedelta(days=45)).strftime("%Y%m%d")
        calendar = self._cached(
            f"trade_cal_{start}_{end}",
            lambda: self.pro.trade_cal(exchange="", start_date=start, end_date=end, is_open="1"),
        )
        if calendar.empty or "cal_date" not in calendar.columns:
            raise RuntimeError("无法读取交易日历")
        dates = sorted({str(value) for value in calendar["cal_date"].dropna()})
        if direction > 0:
            target = next((item for item in dates if item > str(base_date)), "")
        else:
            target = next((item for item in reversed(dates) if item < str(base_date)), "")
        if not target:
            label = "下一" if direction > 0 else "上一"
            raise RuntimeError(f"未找到{label}交易日")
        return target

    def _stock_basic(self) -> pd.DataFrame:
        """读取 A 股基础信息。

        主要用途：
        1. 获取股票名称、代码、上市板块等基础字段。
        2. 获取 industry 字段，作为一级概念接口失败时的行业兜底。
        """
        fields = "ts_code,symbol,name,area,industry,market,list_date"
        df = self._cached(
            "stock_basic_active",
            lambda: self.pro.stock_basic(exchange="", list_status="L", fields=fields),
        )
        df["ts_code"] = df["ts_code"].astype(str)
        df["industry"] = df["industry"].fillna("未分类").replace("", "未分类")
        df["stock_type"] = self._stock_type_series(df)
        return df

    @staticmethod
    def _stock_type_series(df: pd.DataFrame) -> pd.Series:
        symbol = df.get("symbol", df["ts_code"].astype(str).str.slice(0, 6)).astype(str).str.zfill(6)
        market = df.get("market", pd.Series("", index=df.index)).fillna("").astype(str)
        result = market.mask(market.str.strip() == "", "其他")

        result = result.mask(symbol.str.startswith(("688", "689")), "科创板")
        result = result.mask(symbol.str.startswith(("300", "301")), "创业板")
        result = result.mask(symbol.str.startswith(("8", "4", "92")), "北交所")
        result = result.mask(symbol.str.startswith(("600", "601", "603", "605")), "沪市主板")
        result = result.mask(symbol.str.startswith(("000", "001", "002", "003")), "深市主板")
        return result

    def _sw_industry_members(self, rank_level: str) -> pd.DataFrame:
        """读取申万行业成分，返回股票到申万行业的映射。

        rank_level 取值：
        - sw_l1：申万一级行业。
        - sw_l2：申万二级行业。
        - sw_l3：申万三级行业。

        Tushare 的申万行业需要先取行业列表，再逐个行业取成分。
        首次运行会慢，但每个行业成分都会缓存到 review_cache_web，后面会快很多。
        """
        level_map = {"sw_l1": "L1", "sw_l2": "L2", "sw_l3": "L3"}
        sw_level = level_map.get(rank_level)
        if not sw_level:
            return pd.DataFrame(columns=["ts_code", "rank_industry", "rank_industry_code"])

        try:
            classify = self._cached(
                f"sw2021_index_classify_{sw_level}",
                lambda: self.pro.index_classify(level=sw_level, src="SW2021"),
            )
        except Exception as exc:
            self.log(f"申万{sw_level}行业列表读取失败，回退基础行业: {exc}")
            return pd.DataFrame(columns=["ts_code", "rank_industry", "rank_industry_code"])
        if classify.empty or "index_code" not in classify.columns:
            self.log(f"申万{sw_level}行业列表为空，回退基础行业。")
            return pd.DataFrame(columns=["ts_code", "rank_industry", "rank_industry_code"])

        name_col = "industry_name" if "industry_name" in classify.columns else "name"
        if name_col not in classify.columns:
            classify[name_col] = classify["index_code"]

        frames = []
        self.log(f"读取申万{sw_level}行业成分，首次运行会慢，后续使用缓存。")
        for _, row in classify.iterrows():
            index_code = str(row.get("index_code", "")).strip()
            industry_name = str(row.get(name_col, index_code)).strip()
            if not index_code:
                continue
            try:
                members = self._cached(
                    f"sw2021_index_member_{sw_level}_{index_code}",
                    lambda index_code=index_code: self.pro.index_member(index_code=index_code),
                )
            except Exception as exc:
                self.log(f"{industry_name} 成分读取失败，跳过: {exc}")
                continue
            if members.empty:
                continue
            code_col = "con_code" if "con_code" in members.columns else "ts_code"
            if code_col not in members.columns:
                continue
            use = members.copy()
            if "is_new" in use.columns:
                is_new = use["is_new"].fillna("").astype(str).str.upper()
                use = use[(is_new == "Y") | (is_new == "1") | (is_new == "")]
            use = use[[code_col]].rename(columns={code_col: "ts_code"})
            use["ts_code"] = use["ts_code"].astype(str)
            use["rank_industry"] = industry_name
            use["rank_industry_code"] = index_code
            frames.append(use)

        if not frames:
            self.log(f"申万{sw_level}行业成分为空，回退基础行业。")
            return pd.DataFrame(columns=["ts_code", "rank_industry", "rank_industry_code"])
        result = pd.concat(frames, ignore_index=True)
        return result.drop_duplicates("ts_code", keep="first")

    def _stock_concept_map(
        self,
        trade_date: str,
        ts_codes: list[str] | set[str] | pd.Series | None = None,
        board_limit: int = 300,
        concepts_per_stock: int = 6,
        fallback_trade_dates: list[str] | None = None,
    ) -> dict[str, str]:
        """读取东财概念成分，返回股票到产业题材概念的映射。

        这个接口按“概念板块 -> 成分股”读取。原始概念里有融资融券、
        指数成分、昨日表现、业绩预告等泛标签，这里先清洗成产业题材，
        再按当日板块资金流排序取前若干个板块。
        """
        code_iterable = [] if ts_codes is None else ts_codes
        target_codes = {str(code).strip().upper() for code in code_iterable if str(code).strip()}
        if ts_codes is not None and not target_codes:
            return {}

        candidate_dates = [str(trade_date)]
        for item in fallback_trade_dates or []:
            date_text = str(item)
            if date_text and date_text not in candidate_dates:
                candidate_dates.append(date_text)

        for concept_date in candidate_dates:
            try:
                boards = self._cached(
                    f"moneyflow_ind_dc_概念_{concept_date}",
                    lambda concept_date=concept_date: self.pro.moneyflow_ind_dc(trade_date=concept_date, content_type="概念"),
                )
            except Exception as exc:
                self.log(f"{concept_date} 概念板块读取失败，尝试回退: {exc}")
                continue
            if boards.empty or "ts_code" not in boards.columns:
                continue

            data = boards.copy()
            data["ts_code"] = data["ts_code"].astype(str)
            if "name" not in data.columns:
                data["name"] = data["ts_code"]
            data["name"] = data["name"].fillna("").astype(str)
            raw_count = len(data)
            data = data[data["name"].apply(self._is_industry_theme_concept)].copy()
            if data.empty:
                self.log(f"{concept_date} 概念板块清洗后为空，继续回退。")
                continue
            self.log(f"概念板块已清洗为产业题材：{raw_count} -> {len(data)} 个。")
            if "rank" in data.columns:
                data["rank"] = pd.to_numeric(data["rank"], errors="coerce")
                data = data.sort_values("rank", na_position="last")
            data = data.drop_duplicates("ts_code", keep="first").head(max(1, int(board_limit)))

            concept_names: dict[str, list[str]] = {}
            for _, board in data.iterrows():
                board_code = str(board.get("ts_code", "")).strip()
                board_name = str(board.get("name", "")).strip()
                if not board_code or not board_name:
                    continue
                try:
                    members = self._cached(
                        f"dc_member_{concept_date}_{board_code}",
                        lambda concept_date=concept_date, board_code=board_code: self.pro.dc_member(trade_date=concept_date, ts_code=board_code),
                    )
                except Exception as exc:
                    self.log(f"{board_name} 成分股读取失败，跳过: {exc}")
                    continue
                if members.empty:
                    continue
                code_col = "con_code" if "con_code" in members.columns else "ts_code"
                if code_col not in members.columns:
                    continue
                for raw_code in members[code_col].dropna().astype(str):
                    code = raw_code.strip().upper()
                    if target_codes and code not in target_codes:
                        continue
                    names = concept_names.setdefault(code, [])
                    if board_name not in names:
                        names.append(board_name)
                if target_codes and all(len(concept_names.get(code, [])) >= concepts_per_stock for code in target_codes):
                    break

            if concept_names:
                if concept_date != str(trade_date):
                    self.log(f"{trade_date} 产业题材为空，已回退使用 {concept_date} 概念成分。")
                return {code: "；".join(names[:concepts_per_stock]) for code, names in concept_names.items()}

        self.log("未匹配到产业题材概念，相关列留空。")
        return {}

    @classmethod
    def _is_industry_theme_concept(cls, name: Any) -> bool:
        text = str(name or "").strip()
        if len(text) < 2:
            return False
        upper = text.upper()
        return not any(keyword.upper() in upper for keyword in cls.NON_INDUSTRY_CONCEPT_KEYWORDS)

    def _daily_window(self, trade_dates: list[str]) -> pd.DataFrame:
        """读取最近若干交易日的个股日线。

        日线是整个系统的基础：
        1. 计算个股 5/10/20 日涨幅。
        2. 计算均线、突破、回踩、上影线。
        3. 聚合成概念板块的内部扩散和量能。
        """
        frames = []
        for td in trade_dates:
            frame = self._cached(
                f"daily_{td}",
                lambda td=td: self.pro.daily(trade_date=td),
            )
            if not frame.empty and "ts_code" in frame.columns:
                frames.append(frame)
            else:
                self.log(f"{td} 日线为空，跳过。")
        if not frames:
            raise RuntimeError("日线行情为空，请检查 Tushare 权限、日期或网络。")
        data = pd.concat(frames, ignore_index=True)
        for col in ["open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data["trade_date"] = data["trade_date"].astype(str)
        data["ts_code"] = data["ts_code"].astype(str)
        return data.sort_values(["ts_code", "trade_date"])

    def _apply_intraday_snapshot(
        self,
        daily: pd.DataFrame,
        basic: pd.DataFrame,
        trade_date: str,
        snapshot_mode: str = "live",
    ) -> pd.DataFrame:
        """用实时行情合成当天临时日 K。

        ``lunch`` 阶段市场自 11:30 起休盘，因此本次读取的行情就是
        11:30 午间快照；``close`` 和 ``history`` 不应调用实时接口，
        由调用方直接使用收盘日线。
        """
        self._last_realtime_snapshot = {}
        if snapshot_mode not in {"live", "lunch"}:
            return daily
        rt = self._realtime_list(basic, trade_date)
        if rt.empty:
            self.log("实时行情为空，使用最近可用收盘数据。")
            return self._apply_last_close_snapshot(daily, trade_date)
        rows = self._normalize_realtime(rt, basic, trade_date)
        if rows.empty:
            self.log("实时行情字段无法识别，盘中模式退回历史数据。")
            return self._apply_last_close_snapshot(daily, trade_date)
        merged = daily[daily["trade_date"] != trade_date].copy()
        merged = pd.concat([merged, rows], ignore_index=True)
        quote_at = ""
        for col in ("QUOTE_AT", "TRADE_TIME", "DATETIME", "TIME"):
            if col in rt.columns:
                values = rt[col].dropna().astype(str).str.strip()
                if not values.empty and values.ne("").any():
                    parsed = pd.to_datetime(values, errors="coerce").dropna()
                    if not parsed.empty:
                        quote_at = parsed.max().strftime("%Y-%m-%d %H:%M:%S")
                        break
        self._last_realtime_snapshot = {
            "count": len(rows),
            "quote_at": quote_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": getattr(self, "_last_realtime_source", "实时行情"),
        }
        label = "11:30午间快照" if snapshot_mode == "lunch" else "盘中实时行情"
        if snapshot_mode == "lunch":
            self._last_realtime_snapshot["quote_at"] = (
                f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 11:30:00"
            )
        self.log(f"{label}：已用 {len(rows)} 条行情覆盖 {trade_date}。")
        return merged.sort_values(["ts_code", "trade_date"])

    @staticmethod
    def _apply_last_close_snapshot(daily: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        """实时源失败时，用每只股票最近一条历史收盘价补齐当天临时日K。

        盘中当天的正式日线通常还不存在，直接返回原始 ``daily`` 会让
        后续特征的最新日期全部变成 NaN，最终表现为“0 条候选”。这里的
        回退只用于保证策略仍可计算，数据状态仍由调用方标记为实时缺失。
        """
        if daily is None or daily.empty or "ts_code" not in daily.columns:
            return daily
        data = daily.copy()
        data["ts_code"] = data["ts_code"].astype(str)
        data["trade_date"] = data["trade_date"].astype(str)
        history = data[data["trade_date"] != str(trade_date)].copy()
        if history.empty:
            return data
        history = history.sort_values(["ts_code", "trade_date"])
        latest = history.drop_duplicates("ts_code", keep="last").copy()
        latest["trade_date"] = str(trade_date)
        return pd.concat([history, latest], ignore_index=True).sort_values(
            ["ts_code", "trade_date"]
        )

    def _realtime_list(
        self,
        basic: pd.DataFrame | None = None,
        trade_date: str = "",
    ) -> pd.DataFrame:
        """读取 A 股实时行情。

        东财实时源优先；若不可用再尝试 Tushare rt_k 和新浪。每个来源都会
        校验其显式日期字段，避免把上一交易日行情误当为当日快照。
        实时行情不进入 review_cache_web，每次运行都会重新请求。
        """
        self._last_realtime_source = ""
        try:
            data = ts.realtime_list(src="dc")
            if self._realtime_frame_is_current(data, trade_date):
                self._last_realtime_source = "东财实时"
                return data
            if data is not None and not data.empty:
                self.log("东财实时行情日期不是当前 T 日，已跳过该快照。")
        except Exception as exc:
            self.log(f"东财实时行情失败，尝试其他实时源: {exc}")
        if basic is not None and not basic.empty and hasattr(self.pro, "rt_k"):
            codes = basic.get("ts_code", pd.Series(dtype=str)).astype(str).str.upper().tolist()
            frames: list[pd.DataFrame] = []
            for start in range(0, len(codes), 500):
                chunk = codes[start : start + 500]
                if not chunk:
                    continue
                try:
                    frame = self.pro.rt_k(ts_code=",".join(chunk))
                except Exception as exc:
                    self.log(f"Tushare rt_k 股票实时行情第 {start // 500 + 1} 批失败: {exc}")
                    continue
                if frame is not None and not frame.empty and "ts_code" in frame.columns:
                    frames.append(frame)
            if frames:
                result = pd.concat(frames, ignore_index=True)
                if self._realtime_frame_is_current(result, trade_date):
                    self._last_realtime_source = "Tushare rt_k"
                    return result
                self.log("Tushare rt_k 日期不是当前 T 日，已跳过该快照。")
        try:
            data = ts.realtime_list(src="sina", interval=0.1)
            if self._realtime_frame_is_current(data, trade_date):
                self._last_realtime_source = "新浪实时"
                return data
            if data is not None and not data.empty:
                self.log("新浪实时行情日期不是当前 T 日，已跳过该快照。")
        except Exception as exc:
            self.log(f"新浪实时行情失败: {exc}")
        return pd.DataFrame()

    @staticmethod
    def _realtime_frame_is_current(data: pd.DataFrame | None, trade_date: str) -> bool:
        """若供应商给出完整日期，必须与当前 T 一致；无日期字段时保守接受。"""
        if data is None or data.empty:
            return False
        if not trade_date:
            return True
        columns = {str(col).upper(): col for col in data.columns}
        date_tokens: list[str] = []
        for name in ("TRADE_DATE", "DATE", "DATETIME", "TRADE_TIME", "QUOTE_AT"):
            col = columns.get(name)
            if col is None:
                continue
            values = data[col].dropna().astype(str)
            date_tokens.extend(values.str.extract(r"(20\d{6})", expand=False).dropna().tolist())
        return not date_tokens or trade_date in set(date_tokens)

    def _normalize_realtime(self, rt: pd.DataFrame, basic: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        """把实时行情字段整理成类似 daily 的结构。

        不同实时源返回字段名不完全一致，统一成：
        ts_code、open、high、low、close、pre_close、pct_chg、vol、amount。
        后续评分函数就可以同时兼容盘后日线和盘中实时数据。
        """
        data = rt.copy()
        data.columns = [str(col).upper() for col in data.columns]

        def first_existing(*names: str) -> str:
            return next((name for name in names if name in data.columns), "")

        code_col = first_existing("TS_CODE", "CODE")
        # realtime_list 使用 PRICE，rt_k 使用 CLOSE；两者都表示当前价。
        close_col = first_existing("PRICE", "LAST", "CURRENT", "CLOSE")
        # rt_k 的 PRE_CLOSE 是昨收；realtime_list 的 CLOSE 是昨收。
        pre_close_col = first_existing("PRE_CLOSE", "PREV_CLOSE", "YEST_CLOSE", "CLOSE")
        if not code_col or not close_col:
            return pd.DataFrame()

        normalized = pd.DataFrame(index=data.index)
        normalized["ts_code"] = data[code_col].astype(str).str.upper()
        field_sources = {
            "open": ("OPEN",),
            "high": ("HIGH",),
            "low": ("LOW",),
            "close": (close_col,),
            "pre_close": (pre_close_col,) if pre_close_col else (),
            "pct_chg": ("PCT_CHANGE", "PCT_CHG", "PCT_CHANGE_PCT"),
            "vol": ("VOLUME", "VOL"),
            "amount_yuan": ("AMOUNT", "AMOUNT_YUAN"),
        }
        for target, source_names in field_sources.items():
            source = first_existing(*source_names) if source_names else ""
            normalized[target] = data[source] if source else 0
            normalized[target] = pd.to_numeric(normalized[target], errors="coerce").fillna(0)

        # 缺失 OHLC 不能按 0 展示，否则页面会出现 -100% 的伪跌幅。
        # 对只有最新价的实时源，用最新价作为未知开高低的保守临时值。
        normalized = normalized[(normalized["close"] > 0) & (normalized["pre_close"] > 0)].copy()
        if normalized.empty:
            return pd.DataFrame()
        normalized["open"] = normalized["open"].where(normalized["open"] > 0, normalized["close"])
        normalized["high"] = normalized[["high", "open", "close"]].max(axis=1)
        low_fallback = normalized[["open", "close"]].min(axis=1)
        normalized["low"] = normalized["low"].where(normalized["low"] > 0, low_fallback)
        normalized["low"] = normalized[["low", "open", "close"]].min(axis=1)
        # 始终以当前价/昨收重算，避免不同供应商 pct 字段含义或单位不一致。
        normalized["pct_chg"] = (normalized["close"] / normalized["pre_close"] - 1) * 100
        normalized["amount"] = normalized["amount_yuan"] / 1000
        normalized["trade_date"] = trade_date
        cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]
        return normalized[cols]

    def _daily_basic(self, trade_date: str) -> pd.DataFrame:
        """读取 daily_basic。

        主要用到：
        1. total_mv：总市值，用于识别中军和容量票。
        2. pe_ttm / pb：用于低估值策略筛选。
        3. turnover_rate / turnover_rate_f / volume_ratio：后续可用于增强换手和量比判断。

        盘中时当天 daily_basic 可能没更新，所以会自动回退到上一交易日。
        """
        fields = "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,total_mv,circ_mv,pe,pe_ttm,pb,dv_ratio,dv_ttm"
        try:
            df = self._cached(
                f"daily_basic_shared_{trade_date}",
                lambda: self.pro.daily_basic(trade_date=trade_date, fields=fields),
            )
        except Exception:
            df = pd.DataFrame(columns=fields.split(","))
        for col in ["turnover_rate", "turnover_rate_f", "volume_ratio", "total_mv", "circ_mv", "pe", "pe_ttm", "pb", "dv_ratio", "dv_ttm"]:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if "ts_code" in df.columns:
            df["ts_code"] = df["ts_code"].astype(str)
        return df

    def _fina_indicator(self, period: str) -> pd.DataFrame:
        """读取共享财务指标缓存。

        CANSLIM 和低估值策略都会用到 ROE、EPS、利润和收入增速。
        这里统一拉一份覆盖字段，避免每个策略按不同 cache key 重复请求。
        """
        return self._cached_financial_shared(
            f"fina_indicator_shared_{period}",
            [
                lambda: self._call_fina_indicator_vip(period, FINA_INDICATOR_FIELDS),
                lambda: self._call_fina_indicator_vip(period, "ts_code,ann_date,end_date,roe,roe_dt,eps,basic_eps_yoy,dt_eps_yoy,or_yoy,netprofit_yoy"),
                lambda: self._call_fina_indicator_vip(period, ""),
                lambda: self.pro.fina_indicator(period=period, fields=FINA_INDICATOR_FIELDS),
                lambda: self.pro.fina_indicator(period=period, fields="ts_code,ann_date,end_date,roe,roe_dt,eps,basic_eps_yoy,dt_eps_yoy,or_yoy,netprofit_yoy"),
                lambda: self.pro.fina_indicator(period=period),
            ],
        )

    def _call_fina_indicator_vip(self, period: str, fields: str) -> pd.DataFrame:
        """Call TuShare fina_indicator_vip with query fallback for SDK compatibility."""
        if hasattr(self.pro, "fina_indicator_vip"):
            if fields:
                return self.pro.fina_indicator_vip(period=period, fields=fields)
            return self.pro.fina_indicator_vip(period=period)
        params: dict[str, Any] = {"period": period}
        if fields:
            params["fields"] = fields
        return self.pro.query("fina_indicator_vip", **params)

    def _cashflow(self, period: str) -> pd.DataFrame:
        """读取共享现金流量表缓存。"""
        return self._cached_financial_shared(
            f"cashflow_shared_{period}",
            [
                lambda: self.pro.cashflow(period=period, fields=CASHFLOW_FIELDS),
                lambda: self.pro.cashflow(period=period, fields="ts_code,ann_date,end_date,n_cashflow_act,c_pay_acq_const_fiolta"),
                lambda: self.pro.cashflow(period=period),
            ],
        )

    def _cached_financial_shared(self, name: str, fetchers: list[Any]) -> pd.DataFrame:
        path = self._cache_path(name)
        if path.exists():
            try:
                cached = pd.read_csv(path, dtype=str)
            except pd.errors.EmptyDataError:
                self.log(f"{path.name} 缓存为空，已删除并重新读取。")
                path.unlink(missing_ok=True)
                cached = pd.DataFrame()
            if not cached.empty and "ts_code" in cached.columns:
                return cached
            if path.exists():
                reason = ""
                if not cached.empty and "error" in cached.columns:
                    reason = str(cached["error"].dropna().astype(str).head(1).tolist()[0] if not cached["error"].dropna().empty else "")
                self.log(f"{path.name} 缓存无有效 ts_code，已删除并重新读取。{reason[:120]}")
                path.unlink(missing_ok=True)
        last_error = ""
        for fetcher in fetchers:
            try:
                frame = fetcher()
            except Exception as exc:
                last_error = str(exc)
                continue
            if frame is not None and not frame.empty and "ts_code" in frame.columns:
                frame.to_csv(path, index=False, encoding="utf-8-sig")
                return frame
        if last_error:
            self.log(f"{name} 读取失败: {last_error}")
            if "必填参数" in last_error and "ts_code" in last_error:
                pd.DataFrame({"error": [last_error]}).to_csv(path, index=False, encoding="utf-8-sig")
        return pd.DataFrame()

    def _daily_basic_is_empty(self, daily_basic: pd.DataFrame) -> bool:
        """判断 daily_basic 是否为空或无效。

        盘中模式下，如果当天市值数据还没更新，就用上一交易日的市值数据。
        市值不会在一天内剧烈失真，这样更适合盘中决策。
        """
        if daily_basic.empty or "total_mv" not in daily_basic.columns:
            return True
        total_mv = pd.to_numeric(daily_basic["total_mv"], errors="coerce").fillna(0)
        return bool(total_mv.sum() <= 0)

    def _limit_list(self, trade_date: str) -> pd.DataFrame:
        """读取涨跌停和炸板数据。

        用途：
        1. 市场情绪：涨停数、跌停数、连板高度。
        2. 板块情绪：概念内涨停数量和炸板数量。
        3. 个股角色：识别情绪龙头、后排风险、炸板风险。
        """
        try:
            df = self._cached(
                f"limit_list_d_{trade_date}",
                lambda: self.pro.limit_list_d(trade_date=trade_date),
            )
        except Exception:
            df = pd.DataFrame()
        if "ts_code" in df.columns:
            df["ts_code"] = df["ts_code"].astype(str)
        return df

    def _market_flow(self, trade_date: str) -> pd.DataFrame:
        """读取大盘资金流。

        这不是决定性指标，只作为市场温度里的资金流辅助项。
        如果接口失败，不中断复盘，资金流分按中性处理。
        """
        try:
            return self._cached(
                f"moneyflow_mkt_dc_{trade_date}",
                lambda: self.pro.moneyflow_mkt_dc(trade_date=trade_date),
            )
        except Exception as exc:
            self.log(f"大盘资金流读取失败，市场资金分按0处理: {exc}")
            return pd.DataFrame()

