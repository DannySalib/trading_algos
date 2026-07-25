import logging
import sys

from csm import main, SP500WikipediaUniverse, SpySma200MarketRegimeFilter
from csm.universe import Russell2000Universe

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout
        , level=logging.INFO
        , format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main(
        universe=SP500WikipediaUniverse
        , mrf=SpySma200MarketRegimeFilter
    )
