"""Shared contracts for 06 §5 and 09 §5/§17/§25."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SourceTag(str, Enum):
    """Rule provenance required by 00_项目总则.md §1."""

    ORIGINAL = "O"
    DERIVED = "P"
    A_SHARE = "A"
    UNRESOLVED = "U"


class EvaluationStatus(str, Enum):
    PASS = "PASS"
    REJECT = "REJECT"
    INCOMPLETE = "INCOMPLETE"
    UNTRADABLE = "UNTRADABLE"


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    value: Any
    source: SourceTag
    markdown: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source"] = self.source.value
        return data


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    passed: bool | None
    actual: Any
    threshold: Any
    source: SourceTag
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source"] = self.source.value
        return data


@dataclass
class DataSnapshot:
    as_of_date: str
    trade_dates: list[str]
    basic: Any
    daily: Any
    daily_basic: Any
    financials: Any
    market_indices: dict[str, Any] = field(default_factory=dict)
    limit_list: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
