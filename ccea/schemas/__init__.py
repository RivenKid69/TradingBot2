# -*- coding: utf-8 -*-
"""
CCEA JSON Schemas (packaged).

These schema files mirror the canonical copies under `docs/schemas/` so that
installed distributions can still access the protocol/manifest schemas without
relying on repository-relative paths.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any, Dict


def load_schema_json(filename: str) -> Dict[str, Any]:
    """
    Load a schema JSON document by filename from the `ccea.schemas` package.
    """
    text = resources.files(__package__).joinpath(filename).read_text(encoding="utf-8")
    return json.loads(text)
