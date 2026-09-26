from __future__ import annotations

import logging
import sys

from tradingsys.config import load_env
from tradingsys.runner import Runner

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

def main() -> None:
    _configure_logging()
    env = load_env()
    logger.info("Loaded environment for period %s to %s", env.period.t0, env.period.tf)

    runner = Runner(env)
    print(runner.data)


if __name__ == '__main__':
    main()
