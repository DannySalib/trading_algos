from __future__ import annotations

import logging
import sys

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
    df = runner.get()
    # df.to_csv('get_unweighted_forecast_data_output.csv')
    print(df)
    # runner.close_data.to_csv('close_data.csv')
    
if __name__ == '__main__':
    main()
