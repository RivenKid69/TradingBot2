# -*- coding: utf-8 -*-
import os

path = r"index.html"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()
    print("live-trading-config-modal in index.html:", "live-trading-config-modal" in content)
    print("toggleCopilot in index.html:", "toggleCopilot" in content)
    print("credentials-modal in index.html:", "credentials-modal" in content)
    print("quickStartActions in index.html:", "quickStartActions" in content)
    print("Live Trading in index.html:", "Live Trading" in content)
