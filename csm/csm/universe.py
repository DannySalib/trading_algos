from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import Config

logger = logging.getLogger(__name__)


class Universe(ABC):
    @staticmethod
    @abstractmethod
    def load(cfg: Config) -> list[str]: ...

def get_tables(url: str, cfg: Config) -> pd.DataFrame:
    resp = requests.get(url, headers=cfg.headers, timeout=15)
    return pd.read_html(StringIO(resp.text))


class SP500WikipediaUniverse(Universe):
    @staticmethod
    def load(cfg: Config) -> list[str]:
        logger.info("Loading S&P 500 universe from %s", cfg.sp500_wiki_url)
        tables = get_tables(cfg.sp500_wiki_url, cfg)
        df = tables[cfg.table_index]
        tickers = df[cfg.ticker_col].astype(str).str.strip().tolist()
        tickers = [t.replace(".", "-") for t in tickers]
        tickers = sorted(set(tickers))
        logger.info("Loaded %d tickers", len(tickers))
        return tickers

class Russell2000Universe(Universe):
    @staticmethod
    def load(cfg: Config) -> list[str]:
        # URL for IWM (iShares Russell 2000 ETF) fund download endpoint
        logger.info("Loading Russell 2000 universe from BlackRock (IWM)")

        response = requests.get(cfg.russell2000_url, headers=cfg.headers, timeout=15)
        response.raise_for_status()

        # Parse SpreadsheetML (XML) with BeautifulSoup
        soup = BeautifulSoup(response.content, "xml")

        # 1. Extract specifically the FIRST cell (<Cell>) in each row
        first_column_values = []
        for row in soup.find_all("Row"):
            cells = row.find_all("Cell")
            if not cells:
                continue

            # Check ONLY the first cell (Column A)
            first_cell = cells[0]
            data_tag = first_cell.find("Data")
            first_column_values.append(data_tag.text.strip() if data_tag else "")

        # 2. Slice starting right after the "Ticker" header
        if "Ticker" in first_column_values:
            first_column_values = first_column_values[
                first_column_values.index("Ticker") + 1 :
            ]

        # 3. Clean and filter tickers strictly
        tickers = []
        for val in first_column_values:
            # Stop if we reach cash holdings or footer notes
            if val.lower().startswith(("cash", "disclaimer", "subtotal", "total")):
                break

            clean_val = val.replace(".", "-").strip().upper()

            # Valid equity ticker rules: 1-5 chars (e.g. AAPL, BRK-B), no spaces/long text
            if 1 <= len(clean_val) <= 5 and clean_val.replace("-", "").isalnum():
                tickers.append(clean_val)

        tickers = sorted(set(tickers))
        logger.info("Loaded %d tickers for Russell 2000", len(tickers))
        return tickers