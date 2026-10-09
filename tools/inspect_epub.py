"""Inspect an EPUB without extracting its copyrighted text to the workspace."""

from __future__ import annotations

import argparse
import re
import zipfile
from html.parser import HTMLParser


BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "caption", "dd", "div",
    "dt", "figcaption", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
    "header", "li", "main", "nav", "p", "pre", "section", "td", "th",
    "title", "tr",
}


class BlockParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str, str | None]] = []
        self._tag: str | None = None
        self._anchor: str | None = None
        self._parts: list[str] = []
        self._ignored_depth = 0

    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self._parts)).strip()
        if text:
            self.blocks.append((self._tag or "text", text, self._anchor))
        self._tag = None
        self._anchor = None
        self._parts = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = dict(attrs)
        if tag in {"script", "style"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag in BLOCK_TAGS:
            self._flush()
            self._tag = tag
            self._anchor = attrs_dict.get("id") or attrs_dict.get("name")
        elif tag == "br":
            self._parts.append(" ")
        elif tag == "img":
            src = attrs_dict.get("src", "")
            alt = attrs_dict.get("alt", "")
            self._parts.append(f" [图片 src={src} alt={alt}] ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if not self._ignored_depth and tag in BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self._parts.append(data)

    def close(self) -> None:
        super().close()
        self._flush()


def parse_entry(epub: str, entry: str) -> list[tuple[str, str, str | None]]:
    with zipfile.ZipFile(epub) as archive:
        raw = archive.read(entry)
    text = raw.decode("utf-8", errors="replace")
    parser = BlockParser()
    parser.feed(text)
    parser.close()
    return parser.blocks


def main() -> None:
    argp = argparse.ArgumentParser()
    argp.add_argument("epub")
    argp.add_argument("entries", nargs="+")
    argp.add_argument("--headings", action="store_true")
    argp.add_argument("--search")
    argp.add_argument("--context", type=int, default=2)
    argp.add_argument("--start", type=int, default=1, help="First 1-based block to print")
    argp.add_argument("--end", type=int, help="Last 1-based block to print")
    args = argp.parse_args()

    pattern = re.compile(args.search, re.IGNORECASE) if args.search else None
    for entry in args.entries:
        blocks = parse_entry(args.epub, entry)
        print(f"=== {entry} ({len(blocks)} blocks) ===")
        selected: set[int]
        if args.headings:
            selected = {i for i, (tag, _, _) in enumerate(blocks) if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}}
        elif pattern:
            hits = [i for i, (_, text, _) in enumerate(blocks) if pattern.search(text)]
            selected = {
                j
                for i in hits
                for j in range(max(0, i - args.context), min(len(blocks), i + args.context + 1))
            }
        else:
            start = max(0, args.start - 1)
            end = min(len(blocks), args.end if args.end is not None else len(blocks))
            selected = set(range(start, end))

        previous = -2
        for i in sorted(selected):
            if i > previous + 1:
                print("...")
            tag, text, anchor = blocks[i]
            location = f"{entry}#B{i + 1:04d}"
            if anchor:
                location += f"@{anchor}"
            print(f"[{location}][{tag}] {text}")
            previous = i


if __name__ == "__main__":
    main()
