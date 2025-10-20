import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Função para treinar e avaliar modelos de séries temporais

# Treinar modelo
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
        else:
            raise ValueError("Modelo não suportado")

        model.fit(X_train, y_train)
        y_pred = pd.Series(model.predict(X_test), index=y_test.index)
        preds.update(y_pred)
        maes.append(mean_absolute_error(y_test, y_pred))

    return preds, np.mean(maes), model

# Ensemble simples
def ensemble_predictions(preds_dict, weights=None):
    if weights is None:
        weights = {k: 1 for k in preds_dict.keys()}
    all_preds = pd.DataFrame(preds_dict)
    weighted_preds = sum(all_preds[col] * weights[col] for col in all_preds.columns) / sum(weights.values())
    return weighted_preds

# Intervalos de confiança (bootstrap)
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

# Avaliação econômica
def economic_eval(y_true, y_pred):
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

# Simulação de estratégia
def simulate_strategy(df_wide, df_topk, hold_period=30):
    """
    df_wide: DataFrame (index=datetime, columns=tickers)
    df_topk: DataFrame com colunas ['quarter','ticker']
    Retorna: pd.Series de retornos DIÁRIOS (média dos tickers selecionados por dia)
    """
    series_list = []

    for quarter, group in df_topk.groupby("quarter"):
        # obter início do trimestre (pd.Period -> .start_time)
        try:
            start_date_q = quarter.start_time
        except Exception:
            start_date_q = pd.to_datetime(quarter)

        # localizar posição mais próxima no índice (searchsorted é compatível)
        pos = df_wide.index.searchsorted(start_date_q)
        if pos >= len(df_wide.index):
            continue

        end_pos = min(pos + hold_period, len(df_wide.index))
        tickers = [t for t in group["ticker"].tolist() if t in df_wide.columns]
        if not tickers:
            continue

        period_prices = df_wide.iloc[pos:end_pos][tickers]
        daily_returns = period_prices.pct_change().dropna(how="all")
        if daily_returns.empty:
            continue

        # média dos tickers selecionados por dia
        daily_mean = daily_returns.mean(axis=1)
        series_list.append(daily_mean)

    # ===== ESTAS LINHAS DEVEM ESTAR FORA DO LOOP =====
    if not series_list:
        return pd.Series(dtype=float)

    # concat e agrupa por data (média quando houver overlap), ordena
    combined = pd.concat(series_list).groupby(level=0).mean().sort_index()
    combined.name = "strategy_return"
    return combined

# Previsão futura
def forecast_future(model, last_features, n_periods=4, freq="Q"):
    """
    Gera previsões futuras trimestrais usando o modelo já treinado.

    model: modelo Ridge treinado
    last_features: DataFrame com as últimas features do ticker (última linha)
    n_periods: número de períodos futuros a prever
    freq: frequência dos períodos ('Q' para trimestral)
    """
    # Criar uma cópia e remover colunas que não estavam no treino
    current_features = last_features.copy()
    if "ticker" in current_features.columns:
        current_features = current_features.drop(columns=["ticker"])
    
    future_preds = []

    for _ in range(n_periods):
        # Previsão para o período atual
        pred = model.predict(current_features)[0]
        future_preds.append(pred)
        
        # Atualiza features para o próximo período: shift e preencher NaNs com 0
        current_features = current_features.shift(1).fill()

    # Garantir que o índice seja datetime
    last_date = pd.to_datetime(last_features.index[-1])
    print( last_date, type(last_date))
    last_date = last_date + pd.offsets.QuarterEnd(0)  # Ajusta para o final do trimestre, se necessário

    
    # Criar índice de datas futuras trimestrais
    future_index = pd.date_range(start=last_date + pd.offsets.QuarterEnd(), periods=n_periods, freq=freq)

    # DataFrame de previsões futuras
    forecast_df = pd.DataFrame({"dividend_pred": future_preds}, index=future_index)
    
    return forecast_df
