#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check that every relative link in the tracked markdown files resolves.

Targets are resolved against what git tracks, not against the working tree, so a
local run gives the same answer as a clean checkout. Linking a build artefact
that happens to sit in your directory — a generated .cpp, a compiled extension —
is exactly the kind of rot this is meant to catch.

External links (http, mailto) and bare anchors are not checked.

    python tools/check_markdown_links.py          # check, exit 1 on failure
    python tools/check_markdown_links.py --list   # print every checked link
"""
from __future__ import annotations

import argparse
import os
import posixpath
import re
import subprocess
import sys

LINK = re.compile(r"(!?)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#", "tel:")


def tracked_paths() -> set[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout
    return {line for line in out.splitlines() if line}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print every link that is checked")
    args = parser.parse_args()

    tracked = tracked_paths()
    # a link may point at a directory, which git does not list on its own
    directories = set()
    for path in tracked:
        parent = posixpath.dirname(path)
        while parent:
            directories.add(parent)
            parent = posixpath.dirname(parent)

    markdown = sorted(p for p in tracked if p.endswith(".md"))
    broken: list[tuple[str, str]] = []
    checked = 0

    for path in markdown:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        base = posixpath.dirname(path)
        for match in LINK.finditer(text):
            target = match.group(2)
            if target.startswith(SKIP_PREFIXES):
                continue
            file_part = target.split("#", 1)[0]
            if not file_part:
                continue
            checked += 1
            resolved = posixpath.normpath(posixpath.join(base, file_part)).rstrip("/")
            if args.list:
                print(f"{path} -> {target}")
            if resolved not in tracked and resolved not in directories:
                broken.append((path, target))

    print(f"checked {checked} relative links in {len(markdown)} markdown files")
    if broken:
        print(f"\n{len(broken)} broken:")
        for path, target in broken:
            print(f"  {path} -> {target}")
        return 1
    print("all resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
