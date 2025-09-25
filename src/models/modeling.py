import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer


# -----------------------
# Treinar modelo
# -----------------------
def train_model(X, y, model_type="ridge", n_splits=5):
    X_proc = pd.get_dummies(X, drop_first=True)
    X_proc = X_proc.fillna(0)
    imputer = SimpleImputer(strategy="median")
    X_array = imputer.fit_transform(X_proc)
    X_proc = pd.DataFrame(X_array, columns=X_proc.columns, index=X_proc.index)

    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    preds = pd.Series(index=y.index, dtype=float)
    maes = []

    for train_idx, test_idx in tscv.split(X_proc):
        X_train, X_test = X_proc.iloc[train_idx], X_proc.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if model_type == "ridge":
            model = Ridge()
        elif model_type == "rf":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "xgb":
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        elif model_type == "lgb":
            model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("Modelo não suportado")

        model.fit(X_train, y_train)
        y_pred = pd.Series(model.predict(X_test), index=y_test.index)
        preds.update(y_pred)
        maes.append(mean_absolute_error(y_test, y_pred))

    return preds, np.mean(maes)


# -----------------------
# Ensemble simples
# -----------------------
def ensemble_predictions(preds_dict, weights=None):
    if weights is None:
        weights = {k: 1 for k in preds_dict.keys()}
    all_preds = pd.DataFrame(preds_dict)
    weighted_preds = sum(all_preds[col] * weights[col] for col in all_preds.columns) / sum(weights.values())
    return weighted_preds

# -----------------------
# Intervalos de confiança (bootstrap)
# -----------------------
def bootstrap_ci(y_true, y_pred, n_bootstrap=500, ci=0.9):
    preds_matrix = np.zeros((len(y_pred), n_bootstrap))
    n = len(y_true)

    for i in range(n_bootstrap):
        idx = np.random.choice(range(n), size=n, replace=True)
        sample_pred = y_pred.iloc[idx]
        preds_matrix[:, i] = sample_pred

    lower = np.percentile(preds_matrix, (1 - ci) / 2 * 100, axis=1)
    upper = np.percentile(preds_matrix, (1 + ci) / 2 * 100, axis=1)
    return lower, upper

# -----------------------
# Avaliação econômica
# -----------------------
def economic_eval(y_true, y_pred, debug=False):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()

    # evita divisões por zero ou retornos inválidos
    df = df[df["y_pred"] != 0]

    returns = df["y_pred"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    returns = returns.clip(lower=-1, upper=10) # limita retornos extremos

    if returns.empty:
        return 0.0, 0.0, 0.0

    # cumulativos para drawdown
    cum_dividends = df["y_pred"].cumsum()
    cum_max = cum_dividends.cummax()
    drawdown_series = (cum_dividends - cum_max) / cum_max
    max_drawdown = drawdown_series.min()

    sharpe = (returns.mean() / returns.std() * np.sqrt(4)) if returns.std() > 0 else 0.0
    return returns.mean(), sharpe, max_drawdown


# Função para simular retorno da estratégia
# Simula o ganho percentual caso comprasse os tickers top-k e mantivesse por hold_period dias.
def simulate_strategy(df_prices, df_topk, hold_period=30):
    """
    Simula retorno diário de uma estratégia de compra dos tickers top-k por período.

    Args:
        df_prices (pd.DataFrame): preços ajustados históricos long (colunas: ['date', 'ticker', 'adj_close'])
        df_topk (pd.DataFrame): top-k tickers por período (colunas: ['quarter', 'ticker'])
        hold_period (int): número de dias para manter posição

    Returns:
        pd.Series: retorno diário médio da estratégia
    """

    # Transformar preços para formato wide
    df_prices["date"] = pd.to_datetime(df_prices["date"])
    df_wide = df_prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()

    strategy_returns = []

    for quarter, group in df_topk.groupby("quarter"):
        tickers = group["ticker"].tolist()

        # Obter início do período (primeiro dia do trimestre)
        try:
            start_date = quarter.start_time
        except AttributeError:
            start_date = quarter

        # Garantir que start_date exista no índice ou pegar o mais próximo
        if start_date not in df_wide.index:
            # Escolher a data mais próxima
            idx = df_wide.index.searchsorted(start_date)
            if idx >= len(df_wide.index):
                idx = len(df_wide.index) - 1
            start_date = df_wide.index[idx]

        # Definir fim do período
        start_idx = df_wide.index.get_loc(start_date)
        end_idx = start_idx + hold_period
        period_prices = df_wide.iloc[start_idx:end_idx][tickers]

        # Calcular retornos diários
        daily_returns = period_prices.pct_change().dropna(how='all')
        if daily_returns.empty:
            continue

        # Média dos tickers no período
        daily_returns_mean = daily_returns.mean(axis=1)
        strategy_returns.append(daily_returns_mean)

    if strategy_returns:
        strategy_returns = pd.concat(strategy_returns).sort_index()
        strategy_returns.name = "strategy_return"
    else:
        strategy_returns = pd.Series(dtype=float)

    return strategy_returns


# Função para calcular métricas da estratégia
# Métricas clássicas para avaliar se a estratégia vale a pena: crescimento anualizado, risco ajustado e drawdown.
def strategy_metrics(returns):
    """
    Calcula métricas financeiras da estratégia.
    
    Args:
        returns (pd.Series): série de retornos percentuais (diários ou por período)
    
    Returns:
        dict: {'CAGR', 'Sharpe', 'Max Drawdown'}
    """
    if returns.empty:
        return {"CAGR": np.nan, "Sharpe": np.nan, "Max Drawdown": np.nan}
    
    # CAGR
    total_return = (1 + returns).prod()
    n_years = len(returns)/252  # assumindo série diária
    CAGR = total_return**(1/n_years) - 1

    # Sharpe ratio (assumindo risco-free = 0)
    sharpe = returns.mean() / returns.std() * np.sqrt(252)

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    max_dd = (cumulative / cumulative.cummax() - 1).min()

    return {"CAGR": CAGR, "Sharpe": sharpe, "Max Drawdown": max_dd}
