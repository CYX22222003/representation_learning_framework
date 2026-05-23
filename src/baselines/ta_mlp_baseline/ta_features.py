from __future__ import annotations

import numpy as np
import pandas as pd
import talib

# Feature names in the order they are returned by compute_ta_features().
FEATURE_NAMES: list[str] = [
    "z_score",
    "rsi",
    "boll",
    "ULTOSC",
    "pct_change",
    "zsVol",
    "PR_MA_Ratio_short",
    "MA_Ratio_short",
    "MA_Ratio",
    "PR_MA_Ratio",
    "DayOfWeek",
    "Month",
    "Hourly",
    "CDL2CROWS",
    "CDL3BLACKCROWS",
    "CDL3WHITESOLDIERS",
    "CDLABANDONEDBABY",
    "CDLBELTHOLD",
    "CDLCOUNTERATTACK",
    "CDLDARKCLOUDCOVER",
    "CDLDRAGONFLYDOJI",
    "CDLENGULFING",
    "CDLEVENINGDOJISTAR",
    "CDLEVENINGSTAR",
    "CDLGRAVESTONEDOJI",
    "CDLHANGINGMAN",
    "CDLHARAMICROSS",
    "CDLINVERTEDHAMMER",
    "CDLMARUBOZU",
    "CDLMORNINGDOJISTAR",
    "CDLMORNINGSTAR",
    "CDLPIERCING",
    "CDLRISEFALL3METHODS",
    "CDLSHOOTINGSTAR",
    "CDLSPINNINGTOP",
    "CDLUPSIDEGAP2CROWS",
]

N_FEATURES = len(FEATURE_NAMES)  # 36


def compute_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 36 TA features from an OHLCV DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must have columns: open, high, low, close, volume, date.
        ``date`` is used for time-of-day / calendar features.

    Returns
    -------
    DataFrame with columns matching FEATURE_NAMES.
    Rows containing NaN (warm-up period) are dropped.
    """
    d = df.copy()
    close = d["close"].astype(float)
    high = d["high"].astype(float)
    low = d["low"].astype(float)
    open_ = d["open"].astype(float)
    volume = d["volume"].astype(float)

    # --- oscillators ---
    log_ret = np.log(close) - np.log(close.shift(1))
    d["z_score"] = (log_ret - log_ret.rolling(20).mean()) / log_ret.rolling(20).std()
    d["rsi"] = talib.RSI(close) / 100.0
    upper, _, lower = talib.BBANDS(close, nbdevup=2, nbdevdn=2, matype=0)
    d["boll"] = (close - lower) / (upper - lower)
    d["ULTOSC"] = talib.ULTOSC(high, low, close) / 100.0
    d["pct_change"] = close.pct_change()
    d["zsVol"] = (volume - volume.mean()) / volume.std()
    sma21 = talib.SMA(close, 21)
    sma50 = talib.SMA(close, 50)
    sma100 = talib.SMA(close, 100)
    d["PR_MA_Ratio_short"] = (close - sma21) / sma21
    d["MA_Ratio_short"] = (sma21 - sma50) / sma50
    d["MA_Ratio"] = (sma50 - sma100) / sma100
    d["PR_MA_Ratio"] = (close - sma50) / sma50

    # --- calendar features ---
    dates = pd.to_datetime(d["date"])
    d["DayOfWeek"] = dates.dt.dayofweek
    d["Month"] = dates.dt.month
    d["Hourly"] = dates.dt.hour / 4.0

    # --- candlestick patterns (scaled to [-1, 1]) ---
    d["CDL2CROWS"] = talib.CDL2CROWS(open_, high, low, close) / 100
    d["CDL3BLACKCROWS"] = talib.CDL3BLACKCROWS(open_, high, low, close) / 100
    d["CDL3WHITESOLDIERS"] = talib.CDL3WHITESOLDIERS(open_, high, low, close) / 100
    d["CDLABANDONEDBABY"] = talib.CDLABANDONEDBABY(open_, high, low, close) / 100
    d["CDLBELTHOLD"] = talib.CDLBELTHOLD(open_, high, low, close) / 100
    d["CDLCOUNTERATTACK"] = talib.CDLCOUNTERATTACK(open_, high, low, close) / 100
    d["CDLDARKCLOUDCOVER"] = talib.CDLDARKCLOUDCOVER(open_, high, low, close) / 100
    d["CDLDRAGONFLYDOJI"] = talib.CDLDRAGONFLYDOJI(open_, high, low, close) / 100
    d["CDLENGULFING"] = talib.CDLENGULFING(open_, high, low, close) / 100
    d["CDLEVENINGDOJISTAR"] = talib.CDLEVENINGDOJISTAR(open_, high, low, close) / 100
    d["CDLEVENINGSTAR"] = talib.CDLEVENINGSTAR(open_, high, low, close) / 100
    d["CDLGRAVESTONEDOJI"] = talib.CDLGRAVESTONEDOJI(open_, high, low, close) / 100
    d["CDLHANGINGMAN"] = talib.CDLHANGINGMAN(open_, high, low, close) / 100
    d["CDLHARAMICROSS"] = talib.CDLHARAMICROSS(open_, high, low, close) / 100
    d["CDLINVERTEDHAMMER"] = talib.CDLINVERTEDHAMMER(open_, high, low, close) / 100
    d["CDLMARUBOZU"] = talib.CDLMARUBOZU(open_, high, low, close) / 100
    d["CDLMORNINGDOJISTAR"] = talib.CDLMORNINGDOJISTAR(open_, high, low, close) / 100
    d["CDLMORNINGSTAR"] = talib.CDLMORNINGSTAR(open_, high, low, close) / 100
    d["CDLPIERCING"] = talib.CDLPIERCING(open_, high, low, close) / 100
    d["CDLRISEFALL3METHODS"] = talib.CDLRISEFALL3METHODS(open_, high, low, close) / 100
    d["CDLSHOOTINGSTAR"] = talib.CDLSHOOTINGSTAR(open_, high, low, close) / 100
    d["CDLSPINNINGTOP"] = talib.CDLSPINNINGTOP(open_, high, low, close) / 100
    d["CDLUPSIDEGAP2CROWS"] = talib.CDLUPSIDEGAP2CROWS(open_, high, low, close) / 100

    out = d[FEATURE_NAMES].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def build_ta_dataset(
    feather_paths: list[str],
    horizon: int = 1,
    train_ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build train/test arrays from a list of Polymarket OHLCV feather files.

    Each file is split chronologically (first ``train_ratio`` rows → train,
    remainder → test) before merging, matching the framework's data splits.

    Labels are binary: 1 if close price rises over the next ``horizon`` candles,
    0 otherwise.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray float32
    """
    from sklearn.preprocessing import StandardScaler

    X_trains, y_trains, X_tests, y_tests = [], [], [], []

    for path in feather_paths:
        try:
            df = pd.read_feather(path).ffill().interpolate()
        except Exception:
            continue

        features = compute_ta_features(df)
        if len(features) <= horizon:
            continue

        X = features.values[:-horizon].astype(np.float32)
        close_vals = df["close"].values
        close_aligned = close_vals[features.index]
        current = close_aligned[:-horizon]
        future = close_aligned[horizon : len(features)]
        y = (future > current).astype(np.float32)

        if len(X) < 10:
            continue

        split = int(len(X) * train_ratio)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[:split])
        X_te = scaler.transform(X[split:])

        X_trains.append(X_tr)
        y_trains.append(y[:split])
        X_tests.append(X_te)
        y_tests.append(y[split:])

    if not X_trains:
        raise RuntimeError("No valid feather files produced data.")

    return (
        np.concatenate(X_trains).astype(np.float32),
        np.concatenate(y_trains).astype(np.float32),
        np.concatenate(X_tests).astype(np.float32),
        np.concatenate(y_tests).astype(np.float32),
    )
