"""
Stock Price Direction Predictor - Model Training
================================================
Predicts whether a stock will go UP or DOWN the next trading day.

Features used:
- Technical indicators: RSI, MACD, Bollinger Bands, SMA, EMA, ATR, OBV
- Price-based features: returns, volatility, momentum
- Volume features: volume ratio, volume trend

Model: Random Forest Classifier with hyperparameter tuning
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
CONFIG = {
    "ticker": "AAPL",
    "period": "5y",   # 5 years of data
    "test_size": 0.2, # 20% for testing
    "n_estimators": 200,
    "max_depth": 8,
    "random_state": 42,
    # These are no longer used for saving (we save per‑ticker),
    # but kept for backward compatibility if you reference them elsewhere.
    "model_path": "model/stock_model.pkl",
    "scaler_path": "model/scaler.pkl",
    "feature_path": "model/features.pkl",
}

# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────

def compute_rsi(series, period=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    """MACD Line, Signal Line, Histogram"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger_bands(series, period=20, std_dev=2):
    """Bollinger Bands: upper, middle, lower"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower

def compute_atr(high, low, close, period=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def compute_obv(close, volume):
    """On-Balance Volume"""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()

def engineer_features(df):
    """Build all features from raw OHLCV data."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    f = pd.DataFrame(index=df.index)

    # ── Returns & Momentum ──────────────────────────────────────────────────
    f["return_1d"]  = close.pct_change(1)
    f["return_3d"]  = close.pct_change(3)
    f["return_5d"]  = close.pct_change(5)
    f["return_10d"] = close.pct_change(10)
    f["return_20d"] = close.pct_change(20)

    # ── Moving Averages ─────────────────────────────────────────────────────
    for w in [5, 10, 20, 50]:
        f[f"sma_{w}"] = close.rolling(w).mean()
        f[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()
        f[f"price_to_sma_{w}"] = close / f[f"sma_{w}"] - 1
        f[f"price_to_ema_{w}"] = close / f[f"ema_{w}"] - 1

    # ── RSI ─────────────────────────────────────────────────────────────────
    f["rsi_14"] = compute_rsi(close, 14)
    f["rsi_7"]  = compute_rsi(close, 7)
    f["rsi_21"] = compute_rsi(close, 21)

    # ── MACD ────────────────────────────────────────────────────────────────
    macd, signal, hist = compute_macd(close)
    f["macd"]        = macd
    f["macd_signal"] = signal
    f["macd_hist"]   = hist
    f["macd_cross"]  = (macd > signal).astype(int)

    # ── Bollinger Bands ─────────────────────────────────────────────────────
    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(close)
    f["bb_pct"]   = (close - bb_lower) / (bb_upper - bb_lower) # 0=bottom,1=top
    f["bb_width"] = (bb_upper - bb_lower) / bb_mid

    # ── Volatility ──────────────────────────────────────────────────────────
    f["volatility_5d"]  = close.pct_change().rolling(5).std()
    f["volatility_20d"] = close.pct_change().rolling(20).std()
    f["atr_14"]         = compute_atr(high, low, close, 14)
    f["atr_pct"]        = f["atr_14"] / close

    # ── Volume ──────────────────────────────────────────────────────────────
    f["volume_sma_20"] = vol.rolling(20).mean()
    f["volume_ratio"]  = vol / f["volume_sma_20"]
    f["volume_change"] = vol.pct_change()
    f["obv"]           = compute_obv(close, vol)
    f["obv_sma"]       = f["obv"].rolling(20).mean()
    f["obv_ratio"]     = f["obv"] / f["obv_sma"]

    # ── High-Low Range ─────────────────────────────────────────────────────
    f["hl_range"]      = (high - low) / close
    f["close_to_high"] = (close - low) / (high - low + 1e-9)

    # ── Lag Features ───────────────────────────────────────────────────────
    for lag in [1, 2, 3, 5]:
        f[f"return_lag_{lag}"] = f["return_1d"].shift(lag)
        f[f"rsi_lag_{lag}"]    = f["rsi_14"].shift(lag)

    # ── Target: 1 if tomorrow's close > today's close ──────────────────────
    f["target"] = (close.shift(-1) > close).astype(int)

    return f.dropna()

# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_data(ticker=CONFIG["ticker"], period=CONFIG["period"]):
    print(f" Downloading {ticker} ({period})...")
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    print(f" Downloaded {len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})")
    return df

# ─── TRAINING ────────────────────────────────────────────────────────────────

def train(ticker=CONFIG["ticker"]):
    print("\n" + "="*60)
    print(" STOCK DIRECTION PREDICTOR — MODEL TRAINING")
    print("="*60)

    # 1. Load
    raw = load_data(ticker)

    # 2. Features
    print("\n[1/5] Engineering features...")
    feat_df = engineer_features(raw)
    feature_cols = [c for c in feat_df.columns if c != "target"]
    X = feat_df[feature_cols]
    y = feat_df["target"]
    print(f" {len(feature_cols)} features | {len(X)} samples")
    print(f" Class balance UP={y.sum()} DOWN={(~y.astype(bool)).sum()}")

    # 3. Train/test split (time-aware, no shuffle!)
    print("\n[2/5] Splitting data (time-series aware)...")
    split = int(len(X) * (1 - CONFIG["test_size"]))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f" Train: {len(X_train)} | Test: {len(X_test)}")

    # 4. Pipeline: scaler + RF
    print("\n[3/5] Training Random Forest...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=CONFIG["n_estimators"],
            max_depth=CONFIG["max_depth"],
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=CONFIG["random_state"],
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)

    # 5. Evaluate
    print("\n[4/5] Evaluation...")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)

    print(f"\n ┌─────────────────────────┐")
    print(f" │ Accuracy : {acc:.4f} │")
    print(f" │ Precision : {prec:.4f} │")
    print(f" │ Recall : {rec:.4f} │")
    print(f" │ F1 Score : {f1:.4f} │")
    print(f" │ ROC-AUC : {auc:.4f} │")
    print(f" └─────────────────────────┘")

    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

    # Feature importance
    rf_clf = model.named_steps["clf"]
    importances = pd.Series(rf_clf.feature_importances_, index=feature_cols)
    top10 = importances.nlargest(10)
    print("\n Top 10 Features:")
    for name, imp in top10.items():
        bar = "█" * int(imp * 200)
        print(f" {name:<30} {imp:.4f} {bar}")

    # Cross-validation
    print("\n[5/5] Time-Series Cross-Validation (5 folds)...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring="accuracy", n_jobs=-1)
    print(f" CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save per‑ticker
    os.makedirs("model", exist_ok=True)
    model_path   = f"model/{ticker}_model.pkl"
    feature_path = f"model/{ticker}_features.pkl"
    joblib.dump(model, model_path)
    joblib.dump(feature_cols, feature_path)
    print(f"\n ✓ Model saved → {model_path}")
    print(f" ✓ Features saved → {feature_path}")

    return model, feature_cols, {
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "auc": auc,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
    }

# ─── PREDICT ─────────────────────────────────────────────────────────────────

def predict(ticker: str):
    """Load saved model and predict direction for tomorrow."""
    model_path   = f"model/{ticker}_model.pkl"
    feature_path = f"model/{ticker}_features.pkl"

    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_path)

    raw = load_data(ticker, period="1y")
    feat = engineer_features(raw)
    latest = feat[feature_cols].iloc[[-1]]  # most recent row

    prob_up = model.predict_proba(latest)[0][1]
    direction  = "UP" if prob_up >= 0.5 else "DOWN"
    confidence = prob_up if prob_up >= 0.5 else 1 - prob_up

    price_now = raw["Close"].iloc[-1]
    return {
        "ticker": ticker,
        "last_price": round(float(price_now), 2),
        "direction": direction,
        "probability_up": round(float(prob_up), 4),
        "confidence": round(float(confidence), 4),
        "date": str(raw.index[-1].date()),
    }

# ─── HELPER: TRAIN + REPORT FOR ONE TICKER ───────────────────────────────────

def train_and_report(ticker: str):
    model, feature_cols, metrics = train(ticker)
    print("\n" + "="*60)
    print(f" PREDICTION FOR: {ticker}")
    print("="*60)
    result = predict(ticker)
    arrow = "▲" if result["direction"] == "UP" else "▼"
    print(f"\n {arrow} Direction : {result['direction']}")
    print(f" Last Price : ${result['last_price']}")
    print(f" Prob UP : {result['probability_up']*100:.1f}%")
    print(f" Confidence : {result['confidence']*100:.1f}%")
    print(f" As of : {result['date']}\n")
    return metrics, result

# ─── OPTIONAL: LOAD TICKERS FROM FILE ────────────────────────────────────────
# If you want, create a tickers.txt file (one symbol per line) and
# uncomment the function + use it in __main__.

def load_tickers_from_file(path="tickers.txt"):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # tickers passed via command line:
        #   python model/train_model.py AAPL
        #   python model/train_model.py AAPL TSLA RELIANCE.NS
        tickers = sys.argv[1:]
    else:
        # No args → use this as your "market" list.
        # You can either hardcode here or load from tickers.txt.
        tickers = load_tickers_from_file()
        if not tickers:
            tickers = [
                "AAPL",
                "TSLA",
                "RELIANCE.NS",
                # add more symbols here
            ]

    for t in tickers:
        print("\n" + "#" * 80)
        print(f"# TRAINING AND PREDICTING FOR: {t}")
        print("#" * 80)
        train_and_report(t)
