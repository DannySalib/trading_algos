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

    Raises:
        json.JSONDecodeError: If the file is not valid JSON.
        pydantic.ValidationError: If the data doesn't match RunnerEnviron.
    """
    try:
        env_data = load_sysenv_data(path)
    except json.JSONDecodeError as e:
        logger.error(
            "Invalid JSON in %s at line %s, column %s: %s",
            path,
            e.lineno,
            e.colno,
            e.msg,
        )
        raise

    try:
        env = RunnerEnviron.model_validate(env_data)
    except ValidationError as e:
        for error in e.errors():
            field = ".".join(str(x) for x in error["loc"])
            logger.error(
                "Invalid sysenv configuration: '%s' — %s",
                field,
                error["msg"],
            )
        raise
    
    logger.info("Loaded environment for period %s to %s", env.period.t0, env.period.tf)
    return env