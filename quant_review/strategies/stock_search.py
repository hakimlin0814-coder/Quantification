from __future__ import annotations

import re
from typing import Any

import pandas as pd


class StockSearchStrategy:
    """Fuzzy stock lookup by code, Chinese name, and pinyin fragments."""

    SYLLABLE_HINTS = {
        "an": "安按鞍岸",
        "ao": "奥澳傲",
        "ba": "八巴坝霸",
        "bai": "百白柏佰",
        "bao": "宝保包报豹",
        "bei": "北贝倍",
        "ben": "本奔",
        "bi": "比必碧币",
        "bo": "波博玻铂泊",
        "cai": "财材彩才采",
        "chang": "长常昌畅厂",
        "cheng": "成城诚程承",
        "chi": "驰池赤",
        "chuang": "创窗",
        "chuan": "川传船",
        "da": "大达",
        "de": "德得",
        "di": "迪地帝低第滴",
        "dian": "电点典店",
        "dong": "东动董",
        "fang": "方防房纺",
        "fei": "飞菲非肥",
        "feng": "风丰峰锋凤",
        "fu": "富福孚甫服府复",
        "gang": "港钢刚",
        "gao": "高",
        "ge": "格歌革",
        "gong": "工公供功",
        "gu": "股谷古固顾",
        "guang": "广光",
        "gui": "贵桂硅",
        "guo": "国果",
        "hai": "海",
        "hang": "航杭行",
        "he": "合河核禾和",
        "heng": "恒衡",
        "hong": "红宏鸿虹洪",
        "hua": "华化花桦",
        "hui": "汇会辉惠",
        "huo": "火货",
        "ji": "基机集技际吉济季纪",
        "jia": "家佳嘉加",
        "jian": "建健检简",
        "jiang": "江疆",
        "jie": "捷洁杰节",
        "jin": "金今锦晋津劲",
        "jing": "京晶精景经竞",
        "kang": "康",
        "ke": "科可克客",
        "kong": "控空",
        "kuang": "矿",
        "lan": "蓝兰澜",
        "li": "力利立丽锂理里李",
        "lian": "联连莲",
        "liang": "粮良梁亮",
        "lin": "林麟临",
        "long": "龙隆",
        "lu": "路鲁露陆",
        "ma": "马码",
        "mao": "茅贸猫",
        "mei": "美梅煤媒",
        "meng": "蒙盟",
        "min": "民闽敏",
        "mu": "牧木",
        "nan": "南",
        "neng": "能",
        "ning": "宁",
        "nong": "农",
        "ou": "欧",
        "ping": "平苹瓶",
        "pu": "浦普铺",
        "qi": "汽气七奇齐旗",
        "qian": "千前钱黔",
        "qing": "青清轻",
        "rong": "融荣",
        "rui": "瑞锐",
        "san": "三",
        "shan": "山杉陕",
        "shang": "上商尚",
        "sheng": "生升胜圣盛",
        "shi": "石实世士市时视事食",
        "shou": "首手",
        "shu": "数树蜀",
        "si": "四思司",
        "su": "苏速",
        "tai": "台泰太钛",
        "tang": "唐堂",
        "te": "特",
        "tian": "天田",
        "tong": "通同铜桐童",
        "tou": "投透",
        "wan": "万",
        "wei": "威维卫伟微未",
        "wen": "文温",
        "wu": "五物务武无",
        "xi": "西息希锡稀喜",
        "xia": "夏厦",
        "xian": "先线显贤咸",
        "xiang": "湘香祥翔",
        "xin": "信新芯欣心鑫",
        "xing": "兴星行型",
        "yang": "阳洋扬羊央杨养样",
        "ye": "业叶冶",
        "yi": "一益宜医伊亿易仪",
        "yin": "银音印饮",
        "yong": "永",
        "you": "油友优邮游",
        "yuan": "源原远圆园元",
        "yun": "云运",
        "zhang": "张",
        "zhe": "浙哲",
        "zheng": "证正",
        "zhi": "智制纸知",
        "zhong": "中众",
        "zhou": "洲州舟",
        "zi": "紫资子",
    }
    PINYIN_MAP = {char: syllable for syllable, chars in SYLLABLE_HINTS.items() for char in chars}

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.config = engine.config
        self._pypinyin = None
        self._pinyin_style = None
        self._pinyin_mode = "内置轻量映射"

    def run(self) -> dict[str, Any]:
        query = str(getattr(self.config, "stock_search_query", "") or "").strip()
        limit = max(1, int(getattr(self.config, "stock_search_limit", 500) or 500))
        self.engine.log("读取股票基础信息...")
        basic = self.engine._stock_basic()
        self._load_pinyin()

        if not query:
            return {
                "ok": True,
                "query": query,
                "count": 0,
                "rows": [],
                "pinyin_mode": self._pinyin_mode,
                "logs": self.engine.logs,
            }

        self.engine.log(f"按代码、名称和拼音模糊搜索: {query}")
        rows = self._search(basic, query)
        rows = rows[:limit]
        self.engine.log(f"股票模糊搜索完成，共 {len(rows)} 条。")
        return {
            "ok": True,
            "query": query,
            "count": len(rows),
            "rows": rows,
            "pinyin_mode": self._pinyin_mode,
            "logs": self.engine.logs,
        }

    def _load_pinyin(self) -> None:
        try:
            from pypinyin import Style, lazy_pinyin
        except Exception:
            self._pypinyin = None
            self._pinyin_style = None
            self._pinyin_mode = "内置轻量映射"
            return
        self._pypinyin = lazy_pinyin
        self._pinyin_style = Style
        self._pinyin_mode = "pypinyin"

    def _search(self, basic: pd.DataFrame, query: str) -> list[dict[str, Any]]:
        normalized = self._normalize_ascii(query)
        digits = "".join(ch for ch in query if ch.isdigit())
        hint_chars = self._hint_chars(normalized)
        results: list[dict[str, Any]] = []

        for _, row in basic.iterrows():
            symbol = str(row.get("symbol", "") or str(row.get("ts_code", ""))[:6]).zfill(6)
            ts_code = str(row.get("ts_code", ""))
            name = str(row.get("name", "") or "")
            industry = str(row.get("industry", "") or "")
            pinyin_tokens = self._pinyin_tokens(name)
            pinyin = "".join(pinyin_tokens)
            spaced_pinyin = " ".join(pinyin_tokens)
            initials = "".join(token[:1] for token in pinyin_tokens if token)
            score, reasons = self._match_score(
                query=query,
                normalized=normalized,
                digits=digits,
                hint_chars=hint_chars,
                symbol=symbol,
                ts_code=ts_code,
                name=name,
                industry=industry,
                pinyin=pinyin,
                spaced_pinyin=spaced_pinyin,
                initials=initials,
            )
            if score <= 0:
                continue
            results.append(
                {
                    "code": symbol,
                    "ts_code": ts_code,
                    "name": name,
                    "stock_type": row.get("stock_type", ""),
                    "market": row.get("market", ""),
                    "area": row.get("area", ""),
                    "industry": industry,
                    "list_date": row.get("list_date", ""),
                    "pinyin": spaced_pinyin,
                    "initials": initials,
                    "match_score": score,
                    "match_reason": "；".join(reasons),
                }
            )

        return sorted(results, key=lambda item: (-float(item["match_score"]), item["code"]))

    def _match_score(
        self,
        query: str,
        normalized: str,
        digits: str,
        hint_chars: set[str],
        symbol: str,
        ts_code: str,
        name: str,
        industry: str,
        pinyin: str,
        spaced_pinyin: str,
        initials: str,
    ) -> tuple[int, list[str]]:
        score = 0
        reasons: list[str] = []
        query_upper = query.upper()

        if digits:
            if symbol == digits.zfill(6) or ts_code.upper().startswith(digits.zfill(6)):
                score = max(score, 100)
                reasons.append("代码精确匹配")
            elif symbol.startswith(digits):
                score = max(score, 94)
                reasons.append("代码前缀匹配")
            elif digits in symbol:
                score = max(score, 82)
                reasons.append("代码包含匹配")

        if query and query in name:
            score = max(score, 90 if query == name else 84)
            reasons.append("名称包含匹配")
        if query and query in industry:
            score = max(score, 45)
            reasons.append("行业包含匹配")
        if query_upper and query_upper in ts_code.upper():
            score = max(score, 72)
            reasons.append("TS代码包含匹配")

        if normalized:
            if normalized == pinyin:
                score = max(score, 88)
                reasons.append("拼音全名匹配")
            elif normalized in pinyin:
                score = max(score, 76)
                reasons.append("拼音片段匹配")
            elif normalized in spaced_pinyin.replace(" ", ""):
                score = max(score, 74)
                reasons.append("拼音连续匹配")
            elif normalized in initials:
                score = max(score, 68)
                reasons.append("拼音首字母匹配")

        if hint_chars:
            hit = sorted({char for char in name if char in hint_chars})
            if hit:
                score = max(score, 70)
                reasons.append(f"拼音提示字匹配: {''.join(hit)}")

        return score, reasons

    def _pinyin_tokens(self, text: str) -> list[str]:
        if self._pypinyin is not None:
            try:
                return [
                    self._normalize_ascii(item)
                    for item in self._pypinyin(text, errors="ignore")
                    if self._normalize_ascii(item)
                ]
            except Exception:
                pass
        return [self.PINYIN_MAP.get(char, "") for char in text if self.PINYIN_MAP.get(char, "")]

    @staticmethod
    def _normalize_ascii(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value or "").lower())

    def _hint_chars(self, normalized: str) -> set[str]:
        if not normalized or not normalized.isalpha():
            return set()
        direct = set(self.SYLLABLE_HINTS.get(normalized, ""))
        if direct:
            return direct
        hits: set[str] = set()
        for syllable, chars in self.SYLLABLE_HINTS.items():
            if normalized in syllable or syllable in normalized:
                hits.update(chars)
        return hits
