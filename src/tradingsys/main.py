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

    runner = Runner()
    runner._apply_forecasts()
    print(runner.forecasts['absolute_momentum'].data)


if __name__ == '__main__':
    main()
