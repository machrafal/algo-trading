import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

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


# ---------------------------------------------------------------------------------------------------------
# 3. Feature Builder
# ---------------------------------------------------------------------------------------------------------


class FeatureBuilder:
    """
    Computes features used as inputs to regime detection and alpha singnals

    Connects to signal-independence.py (IC, z-scores) and alternative-to-skew-kurt.py
    (higher moment features)
    """

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns

    def realized_vol(self, window: int = 21) -> pd.DataFrame:
        """
        Annualized rolling realized volatility.
        """
        return self.returns.rolling(window).std() * np.sqrt(252)

    def z_score(self, window: int = 63) -> pd.DataFrame:
        """
        Rolling z-score of returns.
        """
        mu = self.returns.rolling(window).mean()
        sig = self.returns.rolling(window).std()
        return (self.returns - mu) / sig

    def momentum(self, lookback: int = 63, skip: int = 5) -> pd.DataFrame:
        """
        Cross-sectional momentum signal (skip most recent 'skip' days).
        """
        return self.returns.shift(skip).rolling(lookback).sum()

    def rolling_skew(self, window: int = 63) -> pd.DataFrame:
        """
        Rolling skewness.
        """
        return self.returns.rolling(window).skew()

    def rolling_kurt(self, window: int = 63) -> pd.DataFrame:
        """
        Rolling excess kurtosis.
        """
        return self.returns.rolling(window).kurt()

    def vol_ratio(self, short: int = 5, long: int = 21) -> pd.DataFrame:
        """
        Short-term vs long-term vol ratio.
        > 1 = vol expanding (regime change signal)
        < 1 = vol contracting (mean-reverting regime)
        """
        vol_s = self.returns.rolling(short).std()
        vol_l = self.returns.rolling(long).std()
        return vol_s / vol_l

    def build_all(self) -> pd.DataFrame:
        """
        Build a flat feature matrix for one ticker or the full universe.
        Returns MultiIndex or single-ticker DataFrame
        """
        tickers = self.returns.columns
        feature_frames = {}

        for t in tickers:
            r = self.returns[[t]]
            feat = pd.DataFrame(index=r.index)
            feat["ret_1d"] = r[t]
            feat["vol_21d"] = self.realized_vol(21)[t]
            feat["vol_63d"] = self.realized_vol(63)[t]
            feat["vol_ratio"] = self.vol_ratio(5, 21)[t]
            feat["z_score_63d"] = self.z_score(63)[t]
            feat["momentum_63d"] = self.momentum(63)[t]
            feat["skew_63d"] = self.rolling_skew(63)[t]
            feat["kurt_63d"] = self.rolling_kurt(63)[t]
            feature_frames[t] = feat

        return pd.concat(feature_frames, axis=1)


# ---------------------------------------------------------------------------------------------------------
# 4. Pipeline
# ---------------------------------------------------------------------------------------------------------


def main():
    print("-" * 50)
    print("probabilistic-alpha-engine")
    print("-" * 50)

    # 1) Load data
    print("\n[1/4] Loading market data from Stooq...")
    loader = DataLoader(source="stooq", verbose=True)

    # Load GPW universe
    prices = loader.get_universe(ticker_dict=GPW_TICKERS, start="2015-01-01")

    # Load WIG20 index as benchmark
    wig20 = loader.get_ohlcv(WIG20_TICKER, start="2015-01-01")

    # 2) Clean the data
    print("\n[2/4] Cleaning price data...")
    cleaner = (
        MarketDataCleaner(prices, verbose=True)
        .fill_missing(method="linear")
        .remove_outliers(z_threshold=5.0)
        .align_to_common_dates()
    )

    clean_prices = cleaner.get_prices()
    log_returns = cleaner.get_returns(log=True)
    simple_returns = cleaner.get_returns(log=False)

    print(
        f"\nFinal universe: {clean_prices.shape[1]} tickers, "
        f"{clean_prices.shape[0]} trading days"
    )

    # 3) Build features
    print("\n[3/4] Building features...")
    builder = FeatureBuilder(log_returns)
    features = builder.build_all()

    # Summary stats
    print("\nFeature summary (last available row, PKN):")
    if "PKN" in features.columns.get_level_values(0):
        print(features["PKN"].tail(1).T.to_string())

    # 4) Save and plot
    print("\n[4/4] Saving outputs and generating charts...")

    clean_prices.to_csv("data_prices.csv")
    log_returns.to_csv("data_log_returns.csv")
    features.to_csv("data_features.csv")
    print("Saved: data_prices.csv, data_log_returns.csv, data_features.csv")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle("Market Data Pipeline - GPW Universe", fontsize=10)

    # Plot 1: Normalized price series
    ax = axes[0, 0]
    (clean_prices / clean_prices.iloc[0] * 100).plot(
        ax=ax, alpha=0.8, lindewidth=1.2, legend=True
    )
    ax.set_title("Normalized prices (base = 100)")
    ax.legend(fontsize=7, ncol=2)

    # Plot 2: Rolling realised volatility (21d)
    ax = axes[0, 1]
    vol = builder.realized_vol(21)
    vol.plot(ax=ax, alpha=0.7, linewidth=1.0, legend=True)
    ax.set_title("Realized volatility - 21d rolling (annualized)")
    ax.set_ylabel("Volatility")
    ax.axhline(
        vol.mean().mean(),
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="Cross-sectional mean",
    )
    ax.legend(fontsize=7, ncol=2)

    # Plot 3: Return distribution (cross-sectional)
    ax = axes[1, 0]
    all_rets = log_returns.values.flatten()
    all_rets = all_rets[~np.isnan(all_rets)]
    ax.hist(
        all_rets, bins=100, density=True, alpha=0.7, color="skyblue", edgecolor="none"
    )
    # Overly normal
    mu, sigma = all_rets.mean(), all_rets.std()
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "r--", linewidth=1.5, label="Normal")
    ax.set_title("Log-return distribution - GPW universe")
    ax.set_xlabel("Log returns")
    ax.set_ylabel("Density")
    ax.legend()

    # Plot 4: Vol ratio heatmap (regime signal preview)
    ax = axes[1, 1]
    vr = builder.vol_ratio(5, 21).tail(252)  # last year
    im = ax.imshow(
        vr.T.values,
        aspect="auto",
        cmap="coolwarm",
        vmin=0.5,
        vmax=2.0,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(vr.columns)))
    ax.set_yticklabels(vr.columns, fontsize=8)
    n_ticks = 6
    step = max(1, len(vr) // n_ticks)
    ax.set_xticks(range(0, len(vr), step))
    ax.set_xticklabels(
        [str(d.date()) for d in vr.index[::step]], rotation=30, fontsize=7
    )
    plt.colorbar(im, ax=ax, label="Vol ratio (>1 = expanding)")
    ax.set_title("Vol ration heatmap - regime change preview")

    plt.tight_layout()

    print("\n" + "-" * 50)

    return clean_prices, log_returns, features
