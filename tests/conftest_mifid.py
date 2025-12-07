# -*- coding: utf-8 -*-
"""
Conftest for MiFID II compliance tests.

Adds project root to sys.path for proper imports.
"""
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
