"""Auditable Q0 research implementation of the O'Neil model for A-shares."""

from .config import OneilResearchConfig
from .research import OneilResearchService

__all__ = ["OneilResearchConfig", "OneilResearchService"]
