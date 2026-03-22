import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

try:
    from pandas_datareader import data as pdr

    STOOQ_AVAILABLE = True
except (ImportError, TypeError):
    STOOQ_AVAILABLE = False

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

GPW_TICKERS = {
    "PKN": "PKN.PL",
    "PKO": "PKO.PL",
    "PZU": "PZU.PL",
    "CDR": "CDR.PL",
    "DNP": "DNP.PL",
    "KGH": "KGH.PL",
    "LPP": "LPP.PL",
    "ALE": "ALE.PL",
    "MBK": "MBK.PL",
    "SPL": "SPL.PL",
}

# Index benchmark
WIG20_TICKER = "^WIG20"
WIG20_YF = "WIG20.WA"

# ---------------------------------------------------------------------------------------------------------
# 1. Data Loader - abstract interface
# ---------------------------------------------------------------------------------------------------------


class DataLoader:
    """
    Abstract market data loaderl
    Primary source: Stooq
    Fallback      : yfinance

    Extends DataBatchProcessor pattern from get-and-store-market-data.py
    with a unified interface and source abstraction.
    """

    def __init__(self, source: str = "stooq", verbose: bool = True):
        """
        Params
        source: str
        'stooq' (default) or 'yfinance'
        verbose: bool
        print status messages
        """
        self.source = source
        self.verbose = verbose

    def get_ohlcv(
        self,
        ticker: str,
        start: str = "2015-01-01",
        end: str = None,
    ) -> pd.DataFrame:
        """
        Download OHLCV data for a single ticker

        Returns
        pd.DataFrame with columns: Open, High, Low, Close, Volumne
        Index: DatatimeIndex
        """
        if end is None:
            end = pd.Timestamp.today().strftime("%Y-%m-%d")

        df = None

        if self.source == "stooq" and STOOQ_AVAILABLE:
            df = self._load_stooq(ticker, start, end)

        if df is None or df.empty:
            if YFINANCE_AVAILABLE:
                if self.verbose:
                    print(f"Stooq failed for {ticker}, trying yfinance...")
                df = self._load_yfinance(ticker, start, end)
            else:
                raise RuntimeError(
                    f"Cound not load {ticker}. "
                    "Install pandas-datareader and/or yfinance."
                )

        return df

    def _load_stooq(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        try:
            df = pdr.DataReader(ticker, "stooq", start=start, end=end)
            df = df.sort_index()  # stooq returns descending
            df.index.name = "Date"
            if self.verbose:
                print(
                    f"[Stooq] {ticker}: {len(df)} rows."
                    f"({df.index[0].date()} {df.index[-1].date()})"
                )
            return df
        except Exception as e:
            if self.verbose:
                print(f"[Stooq] {ticker} error: {e}")
            return pd.DataFrame()

    def _load_yfinance(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        try:
            df = yf.download(
                ticker, start=start, end=end, progress=False, auto_adjust=True
            )
            df.index.name = "Date"
            if self.verbose:
                print(
                    f"[yfinance] {ticker}: {len(df)} rows"
                    f"({df.index[0].date()} {df.index[-1].date()})"
                )
            return df
        except Exception as e:
            if self.verbose:
                print(f"[yfinance] {ticker} error: {e}")
            return pd.DataFrame()

    def get_universe(
        self,
        ticker_dict: dict,
        start: str = "2015-01-01",
        end: str = None,
        column: str = "Close",
    ) -> pd.DataFrame:
        """
        Download Close prices for a universe of tickers.
        Returns:
        pd.DataFrame: rows = dates, columns = short names

        Extends DataBatchProcessor.process_datasets() pattern.
        """
        frames = {}
        for short_name, ticker in ticker_dict.items():
            df = self.get_ohlcv(ticker, start, end)
            if not df.empty and column in df.columns:
                frames[short_name] = df[column]

        if not frames:
            raise RuntimeError("No data loaded for any ticker.")

        combined = pd.DataFrame(frames)
        combined.index = pd.to_datetime(combined.index)
        combined.sort_index(inplace=True)
        return combined
