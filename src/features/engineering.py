import pandas as pd
import numpy as np

def add_dividend_lags(df, lags=[1,2,3]):
    for lag in lags:
        df[f"dividend_lag_{lag}"] = df["dividend"].shift(lag)
    return df

def add_moving_averages(df):
    df["dividend_ma_3"] = df["dividend"].rolling(3).mean()
    df["dividend_ma_6"] = df["dividend"].rolling(6).mean()
    df["dividend_ma_12"] = df["dividend"].rolling(12).mean()
    return df

def add_trend_features(df):
    df["dividend_trend"] = df["dividend"].pct_change().rolling(4).mean()
    return df

def add_payout_ratio(df):
    if "eps" in df.columns:
        df["payout_ratio"] = df["dividend"] / df["eps"]
    return df

def add_liquidity_debt_ratios(df):
    df["debt_to_equity"] = np.nan
    df["current_ratio"] = np.nan
    return df

def add_volatility(df, windows=[30,90]):
    for w in windows:
        df[f"vol_{w}d"] = df["close"].pct_change().rolling(w).std()
    return df

def add_quarter_dummies(df):
    df = pd.concat([df, pd.get_dummies(df["quarter"].dt.quarter, prefix="Q")], axis=1)
    return df

def add_time_since_last_dividend(df):
    df["time_since_last_dividend"] = df["dividend"].apply(lambda x: 0 if x>0 else 1).cumsum()
    return df

def build_features(df):
    df = add_dividend_lags(df)
    df = add_moving_averages(df)
    df = add_trend_features(df)
    df = add_payout_ratio(df)
    df = add_liquidity_debt_ratios(df)
    df = add_volatility(df)
    df = add_quarter_dummies(df)
    df = add_time_since_last_dividend(df)
    return df

# Seleção top-k
def select_top_k(df_preds, k=3, by="dividend_pred"):
    """
    Seleciona os top-k tickers por previsão de dividendos.
    
    Args:
        df_preds (pd.DataFrame): DataFrame com colunas ['ticker', 'quarter', 'dividend_pred']
        k (int): número de tickers a selecionar
        by (str): coluna usada para ranking ('dividend_pred' ou 'dividend_yield')
    
    Returns:
        pd.DataFrame: top-k tickers por período
    """
    top_k_list = []
    for quarter, group in df_preds.groupby("quarter"):
        top_k = group.sort_values(by=by, ascending=False).head(k)
        top_k_list.append(top_k)
    return pd.concat(top_k_list).reset_index(drop=True)


def compute_metrics(returns):
    if returns.empty:
        return {"CAGR": np.nan, "Sharpe": np.nan, "Max Drawdown": np.nan}

    # n_years a partir do período coberto pela série
    if isinstance(returns.index, pd.DatetimeIndex) and len(returns.index) > 1:
        days = (returns.index.max() - returns.index.min()).days
        n_years = max(days / 365.25, 1/252)
    else:
        n_years = max(len(returns) / 252, 1/252)

    # Total Return
    total_return = (1 + returns).prod()
    CAGR = total_return ** (1 / n_years) - 1

    # Sharpe annualizado (assume risk-free = 0)
    sharpe = returns.mean() / (returns.std() if returns.std() > 0 else np.nan) * np.sqrt(252)

    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    max_dd = (cumulative / cumulative.cummax() - 1).min()

    return {"CAGR": CAGR, "Sharpe": sharpe, "Max Drawdown": max_dd}
