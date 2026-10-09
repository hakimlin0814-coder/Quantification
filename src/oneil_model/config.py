"""Configuration and provenance validation for 06 §5.1 and 09 §23."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .models import RuleSpec, SourceTag


class OneilConfigurationError(ValueError):
    """Raised when a requested research run would hide an unresolved choice."""


@dataclass(frozen=True)
class OneilResearchConfig:
    """Versioned Q0 research baseline.

    Defaults are visible research starting points, not claims that the book supplied
    a unique machine algorithm. Their provenance is exposed by ``rule_specs``.
    See 09_欧奈尔模型_策略逻辑详细设计.md §23.
    """

    version: str = "oneil-q0-research-v1"
    research_level: str = "Q0"
    top_n: int = 100

    # O: explicit book thresholds; exact A-share fields remain an implementation mapping.
    min_eps_yoy: float = 25.0
    min_sales_yoy: float = 25.0
    min_annual_eps_cagr: float = 25.0
    min_roe: float = 17.0
    min_rs_percentile: float = 80.0
    min_breakout_rvol: float = 1.40

    # P/A research choices. They are configurable and never labelled as original rules.
    near_high_ratio: float = 0.85
    rvol_lookback: int = 50
    rs_lookback: int = 252
    min_history_days: int = 250
    exclude_st: bool = True
    include_chinext: bool = True
    include_star: bool = True
    include_bj: bool = False
    require_positive_eps: bool = True

    # Unresolved complete-strategy modules stay explicit and cannot silently pass.
    pattern_mode: str = "not_evaluated"
    institution_mode: str = "not_evaluated"
    market_gate_mode: str = "display_only"
    sentiment_mode: str = "display_only"
    financial_source_mode: str = "tushare_akshare_fallback"

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "OneilResearchConfig":
        def as_bool(key: str, default: bool) -> bool:
            value = payload.get(key, default)
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() not in {"0", "false", "no", "off", ""}

        config = cls(
            version=str(payload.get("oneil_version", cls.version)).strip() or cls.version,
            research_level=str(payload.get("oneil_research_level", cls.research_level)).strip().upper(),
            top_n=int(payload.get("oneil_top_n", cls.top_n)),
            min_eps_yoy=float(payload.get("oneil_min_eps_yoy", cls.min_eps_yoy)),
            min_sales_yoy=float(payload.get("oneil_min_sales_yoy", cls.min_sales_yoy)),
            min_annual_eps_cagr=float(payload.get("oneil_min_annual_eps_cagr", cls.min_annual_eps_cagr)),
            min_roe=float(payload.get("oneil_min_roe", cls.min_roe)),
            min_rs_percentile=float(payload.get("oneil_min_rs", cls.min_rs_percentile)),
            min_breakout_rvol=float(payload.get("oneil_min_rvol", cls.min_breakout_rvol)),
            near_high_ratio=float(payload.get("oneil_near_high_ratio", cls.near_high_ratio)),
            rvol_lookback=int(payload.get("oneil_rvol_lookback", cls.rvol_lookback)),
            rs_lookback=int(payload.get("oneil_rs_lookback", cls.rs_lookback)),
            min_history_days=int(payload.get("oneil_min_history_days", cls.min_history_days)),
            exclude_st=as_bool("oneil_exclude_st", cls.exclude_st),
            include_chinext=as_bool("oneil_include_chinext", cls.include_chinext),
            include_star=as_bool("oneil_include_star", cls.include_star),
            include_bj=as_bool("oneil_include_bj", cls.include_bj),
            require_positive_eps=as_bool("oneil_require_positive_eps", cls.require_positive_eps),
            pattern_mode=str(payload.get("oneil_pattern_mode", cls.pattern_mode)).strip(),
            institution_mode=str(payload.get("oneil_institution_mode", cls.institution_mode)).strip(),
            market_gate_mode=str(payload.get("oneil_market_gate_mode", cls.market_gate_mode)).strip(),
            sentiment_mode=str(payload.get("oneil_sentiment_mode", cls.sentiment_mode)).strip(),
            financial_source_mode=str(payload.get("oneil_financial_source_mode", cls.financial_source_mode)).strip(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        errors: list[str] = []
        if self.research_level != "Q0":
            errors.append("首批代码只实现Q0；Q1/Q2需要形态标注、事件N和机构I确认")
        if not 1 <= self.top_n <= 1000:
            errors.append("oneil_top_n 必须在1到1000之间")
        for name, value in {
            "min_eps_yoy": self.min_eps_yoy,
            "min_sales_yoy": self.min_sales_yoy,
            "min_annual_eps_cagr": self.min_annual_eps_cagr,
            "min_roe": self.min_roe,
            "min_rs_percentile": self.min_rs_percentile,
        }.items():
            if not 0 <= value <= 500:
                errors.append(f"{name} 超出允许研究范围")
        if not 0.5 <= self.near_high_ratio <= 1.0:
            errors.append("near_high_ratio 必须在0.5到1.0之间")
        if not 1.0 <= self.min_breakout_rvol <= 10.0:
            errors.append("min_breakout_rvol 必须在1.0到10.0之间")
        if not 5 <= self.rvol_lookback <= 250:
            errors.append("rvol_lookback 必须在5到250之间")
        if not 60 <= self.rs_lookback <= 300:
            errors.append("rs_lookback 必须在60到300之间")
        if not 60 <= self.min_history_days <= self.rs_lookback:
            errors.append("min_history_days 必须在60到rs_lookback之间")
        if self.pattern_mode != "not_evaluated":
            errors.append("D11未确认：当前只允许 pattern_mode=not_evaluated")
        if self.institution_mode != "not_evaluated":
            errors.append("D09未确认：当前只允许 institution_mode=not_evaluated")
        if self.market_gate_mode != "display_only":
            errors.append("D10/D13未确认：M当前只能展示，不能门控买入或仓位")
        if self.sentiment_mode != "display_only":
            errors.append("A股情绪尚未验证：当前只能展示")
        if self.financial_source_mode != "tushare_akshare_fallback":
            errors.append("当前只支持Tushare主源、AkShare兜底")
        if errors:
            raise OneilConfigurationError("；".join(errors))

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def rule_specs(self) -> list[RuleSpec]:
        return [
            RuleSpec("C_EPS_YOY", self.min_eps_yoy, SourceTag.ORIGINAL, "03 §2；01 007#B42–B48", "季度EPS同比研究阈值"),
            RuleSpec("C_SALES_YOY", self.min_sales_yoy, SourceTag.ORIGINAL, "03 §2；01 007#B56–B60", "季度销售同比研究阈值"),
            RuleSpec("A_EPS_CAGR", self.min_annual_eps_cagr, SourceTag.ORIGINAL, "03 §3；01 008#B5–B7", "三年EPS复合增长研究阈值"),
            RuleSpec("A_ROE", self.min_roe, SourceTag.ORIGINAL, "03 §3；01 008#B13–B17", "年度ROE研究阈值"),
            RuleSpec("L_RS", self.min_rs_percentile, SourceTag.ORIGINAL, "03 §7；01 011#B24–B32", "52周相对强度最低偏好"),
            RuleSpec("S_RVOL", self.min_breakout_rvol, SourceTag.ORIGINAL, "03 §6；01 006#B43–B46", "突破量高于常态40%起点"),
            RuleSpec("N_NEAR_HIGH", self.near_high_ratio, SourceTag.DERIVED, "04 §6.3", "接近52周高点的可配置代理"),
            RuleSpec("S_RVOL_WINDOW", self.rvol_lookback, SourceTag.DERIVED, "04 §6.4", "常态成交量参考窗口"),
            RuleSpec("UNIVERSE_HISTORY", self.min_history_days, SourceTag.A_SHARE, "05 §4", "标准A股研究池最短有效历史"),
            RuleSpec("UNIVERSE_ST", self.exclude_st, SourceTag.A_SHARE, "05 §4", "风险警示股票的新买入过滤"),
            RuleSpec("PATTERN", self.pattern_mode, SourceTag.UNRESOLVED, "09 §16", "完整基底算法尚未确认"),
            RuleSpec("I", self.institution_mode, SourceTag.UNRESOLVED, "09 §13", "机构质量模块尚未确认"),
            RuleSpec("M_GATE", self.market_gate_mode, SourceTag.UNRESOLVED, "09 §14", "市场状态目前仅展示"),
            RuleSpec("A_SENTIMENT", self.sentiment_mode, SourceTag.A_SHARE, "09 §15", "A股情绪目前仅展示"),
        ]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "values": asdict(self),
            "config_hash": self.config_hash,
            "rules": [item.to_dict() for item in self.rule_specs()],
        }
