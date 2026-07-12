"""
Pulls the current ticker universe from Wikipedia for use in a cross-sectional
momentum strategy. Defaults to S&P 500; swap WIKI_URL/TABLE_INDEX for other indices.

Other useful Wikipedia tables (same pattern, just change URL + table index):
  Nasdaq-100:  https://en.wikipedia.org/wiki/Nasdaq-100          (table w/ 'Ticker' col)
  Dow 30:      https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average
  S&P 400:     https://en.wikipedia.org/wiki/List_of_S%26P_400_companies
  S&P 600:     https://en.wikipedia.org/wiki/List_of_S%26P_600_companies
"""

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
TABLE_INDEX = 0  # first table on the page is the constituent list
TICKER_COL = "Symbol"


def get_sp500_universe(alpaca_format: bool = True) -> list[str]:
    """
    Scrape current S&P 500 constituents from Wikipedia.

    alpaca_format: if True, replaces '.' with '-' in tickers (e.g. BRK.B -> BRK-B)
                   to match Alpaca/most broker ticker conventions.
    """
    headers = {"User-Agent": "Mozilla/5.0"}  # wikipedia blocks default requests UA sometimes
    resp = requests.get(WIKI_URL, headers=headers, timeout=15)
    resp.raise_for_status()

    tables = pd.read_html(resp.text)
    df = tables[TABLE_INDEX]

    tickers = df[TICKER_COL].astype(str).str.strip().tolist()

    if alpaca_format:
        tickers = [t.replace(".", "-") for t in tickers]

    return sorted(set(tickers))


def get_universe_with_sector() -> pd.DataFrame:
    """
    Returns full constituent table (ticker, sector, sub-industry, etc.) in case
    you want to sector-neutralize your momentum ranking or exclude sectors.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(WIKI_URL, headers=headers, timeout=15)
    resp.raise_for_status()
    df = pd.read_html(resp.text)[TABLE_INDEX]
    df[TICKER_COL] = df[TICKER_COL].astype(str).str.replace(".", "-", regex=False)
    return df


if __name__ == "__main__":
    universe = get_sp500_universe()
    # print(f"Pulled {len(universe)} tickers")
    # print(universe[:10], "...")