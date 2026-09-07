#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check that every relative link in the tracked markdown files resolves.

External links (http, mailto) and bare anchors are not checked — only paths that
point at another file in the repository, which are the ones that silently rot
when documents move.

    python tools/check_markdown_links.py          # check, exit 1 on failure
    python tools/check_markdown_links.py --list   # print every checked link
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

LINK = re.compile(r"(!?)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#", "tel:")


def tracked_markdown() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"], capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print every link that is checked")
    args = parser.parse_args()

    broken: list[tuple[str, str]] = []
    checked = 0

    for path in tracked_markdown():
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        base = os.path.dirname(path)
        for match in LINK.finditer(text):
            target = match.group(2)
            if target.startswith(SKIP_PREFIXES):
                continue
            file_part = target.split("#", 1)[0]
            if not file_part:
                continue
            checked += 1
            resolved = os.path.normpath(os.path.join(base, file_part))
            if args.list:
                print(f"{path} -> {target}")
            if not os.path.exists(resolved):
                broken.append((path, target))

    print(f"checked {checked} relative links in {len(tracked_markdown())} markdown files")
    if broken:
        print(f"\n{len(broken)} broken:")
        for path, target in broken:
            print(f"  {path} -> {target}")
        return 1
    print("all resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
