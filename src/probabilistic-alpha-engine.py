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


# ---------------------------------------------------------------------------------------------------------
# 2. Data Cleaner
# ---------------------------------------------------------------------------------------------------------


class MarketDataCleaner:
    """
    Financial data cleaning

    Extends DataCleaner class with methods specific to OHLCV / returns.

    Incorporate patterns from:
    - data-cleaning-and-preparation.py (interpolation, Savitzky-Golay)
    - noisy-financial-data.py          (missing values, measurement error)
    """

    def __init__(self, prices: pd.DataFrame, verbose: bool = True):
        """
        Params
        prices: pd.DataFrame
        price matrix (rows = dates, columns = tickers)
        """
        self.prices = prices.copy()
        self.verbose = verbose
        self._report_quality(prices)

    def _report_quality(self, prices: pd.DataFrame):
        """
        Print data quality summary
        """
        missing = prices.isnan().sum()
        total = len(prices)
        if self.verbose:
            print("\n-----------Data Quality Report---------------")
            for col in prices.columns:
                pct = missing[col] / total * 100
                status = "ok" if pct == 0 else f"{pct:.1f}% missing"
                print(f"{col < 8} {total} rows {status}")
            print(f"Data range: {prices.index[0].date()} {prices.index[-1].date()}")
            print("---------------------------------------------\n")

    def fill_missing(self, method: str = "linear") -> "MarketDataCleaner":
        """
        Fill missing prices
        Uses lineaer interpolation
        Falls back to forward-fill then backward-fill for edge cases.
        """
        before = self.prices.isna().sum().sum()
        self.prices = self.prices.interpolate(method=method, limit_area="both")
        self.prices = self.prices.ffill().bfill()
        after = self.prices.isna().sum().sum()
        if self.verbose and before > 0:
            print(f"fill_missing: {before} NaN -> {after} NaN (method={method})")
        return self

    def remove_outliers(
        self,
        z_threshold: float = 5.0,
    ) -> "MarketDataCleaner":
        """
        Replace daily return outliers (|z| > threshold) with NaN, then re-interpolate.
        Addresses measurement noise from noisy-financial-data.py.
        """
        returns = self.prices.pct_change()
        z = (returns - returns.mean()) / returns.std()
        mask = z.abs() > z_threshold

        if self.verbose:
            n_outliers = mask.sum().sum()
            if n_outliers > 0:
                print(
                    f"remove outliers: {n_outliers} outlier returns"
                    f"(|z| > {z_threshold}) replaced"
                )

        # Replace outlier prices with NaN then re-interpolate
        # Flag price at t when return at t (=price[t]/price[t-1] is outlier)
        price_mask = mask.fillna(False)
        self.prices = self.prices.where(~price_mask, other=np.nan)
        self.prices = self.prices.interpolate(method="linear", limit_direction="both")
        return self

    def align_to_common_dates(self) -> "MarketDataCleaner":
        """
        Drop dates where ANY ticker has missing data after filling.
        """
        before = len(self.prices)
        self.prices = self.prices.dropna()
        after = len(self.prices)
        if self.verbose and before != after:
            print(
                f"aglin_dates: {before} -> {after} rows."
                f"({before - after} dates dropped)"
            )
        return self

    def get_prices(self) -> pd.DataFrame:
        return self.prices.copy()

    def get_returns(self, log: bool = True) -> pd.DataFrame:
        """
        Compute daily returns
        Params:
        log: bool
        If True, log returns (default). Else simple returns.
        """
        if log:
            return np.log(self.prices / self.prices.shift(1)).dropna()
        return self.prices.pct_change().dropna()
