from __future__ import annotations

from typing import Any

from src.oneil_model.config import OneilResearchConfig
from src.oneil_model.data import OneilDataGateway
from src.oneil_model.research import OneilResearchService


class OneilModelStrategy:
    """Thin adapter between the existing review engine and the new research package."""

    def __init__(self, engine: Any, research_config: OneilResearchConfig) -> None:
        self.engine = engine
        self.research_config = research_config

    def run(self) -> dict[str, Any]:
        self.engine.log("欧奈尔模型：加载Tushare主数据与AkShare财务兜底...")
        snapshot = OneilDataGateway(self.engine, self.research_config).load()
        self.engine.log(f"欧奈尔模型：以{snapshot.as_of_date}为观察日计算Q0研究观察池...")
        result = OneilResearchService(self.research_config).screen(snapshot)
        self.engine.log(f"欧奈尔模型：完成，展示{len(result.get('rows', []))}条研究记录。")
        result["logs"] = list(self.engine.logs)
        return result
