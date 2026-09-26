from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tradingsys.models.runner_env import RunnerEnviron

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / 'config'
SYSENV_PATH = CONFIG_DIR / 'sysenv.json'
CACHE_PATH = Path(".cache")

logger = logging.getLogger(__name__)


def load_sysenv_data(path: Path = SYSENV_PATH) -> Any:
    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f)


def load_env(path: Path = SYSENV_PATH) -> RunnerEnviron:
    """Load and validate the runner environment from a sysenv config file.

    This is an explicit, on-demand loader rather than an import-time side
    effect: call it from an application entrypoint (see main.py), not from
    package __init__.py. That keeps `import tradingsys` safe to do in tests,
    tooling, and anywhere else a config file might not be present or valid.

    Raises:
        json.JSONDecodeError: if the file is not valid JSON.
        pydantic.ValidationError: if the data doesn't match RunnerEnviron.
    """
    try:
        env_data = load_sysenv_data(path)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON at line %s, column %s: %s", e.lineno, e.colno, e.msg)
        raise

    try:
        return RunnerEnviron.model_validate(env_data)
    except ValidationError:
        logger.exception("Could not validate sysenv configuration")
        raise
