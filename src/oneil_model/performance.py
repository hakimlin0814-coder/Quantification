"""Descriptive research summaries; portfolio metrics remain gated by 09 §24."""

from __future__ import annotations

from typing import Any

import pandas as pd


def screening_summary(rows: pd.DataFrame) -> dict[str, Any]:
    if rows is None or rows.empty:
        return {"total": 0, "pass": 0, "reject": 0, "incomplete": 0}
    status = rows["status"].astype(str)
    return {
        "total": int(len(rows)),
        "pass": int((status == "PASS").sum()),
        "reject": int((status == "REJECT").sum()),
        "incomplete": int((status == "INCOMPLETE").sum()),
        "complete_oneil_signals": 0,
        "note": "PASS仅表示Q0观察门通过，不是完整欧奈尔买入信号",
    }
