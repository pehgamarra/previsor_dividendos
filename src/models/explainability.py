import shap
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st 
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuração do SHAP para modelos lineares

# Configuração para evitar warnings desnecessários
def compute_feature_importance(model, feature_names):
    """Importância global das features"""
    if hasattr(model, "coef_"):  # Ridge
        importance = pd.Series(model.coef_, index=feature_names)
    elif hasattr(model, "feature_importances_"):  # RF, LGBM, XGB
        importance = pd.Series(model.feature_importances_, index=feature_names)
    else:
        raise ValueError("Modelo não suporta importância nativa")
    return importance.sort_values(ascending=False)

# Explicações SHAP
def shap_explain(model, X, sample_size=200):
    """Explicação SHAP global e local"""
    # fixado para evitar escolhas no Streamlit
    X_sample = X.sample(min(sample_size, len(X)), random_state=42)
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    return shap_values, X_sample

# Gráficos SHAP
def plot_shap_summary(shap_values, X):
    """Resumo global SHAP"""
    plt.clf()
    shap.summary_plot(
        shap_values, 
        X, 
        plot_type="bar", 
        show=False, 
        plot_size=(6, 4)  # controla largura x altura
    )
    fig = plt.gcf() 
    return fig

# Explicação local SHAP
def plot_shap_waterfall(shap_values, i=0): 
    """Explicação local para uma previsão fixa (primeiro índice)""" 
    fig = plt.figure() 
    shap.plots.waterfall(shap_values[i], show=False) 
    return fig

# Explicação local SHAP alternativa
def plot_shap_local(shap_values, index=0):
    """Explicação local SHAP para uma previsão específica (estático)"""
    fig = plt.figure()
    shap.plots.waterfall(shap_values[index], show=False)
    return fig

# Robustez do modelo
def robustness_by_sector(df_pred: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    """
    Avalia erro por setor.
    sector_map: dict {ticker: setor}
    """
    df = df_pred.copy()
    df["sector"] = df["ticker"].map(sector_map).fillna("Desconhecido")

    results = []
    for sector, group in df.groupby("sector"):
        mae = mean_absolute_error(group["dividend"], group["dividend_pred"])
        rmse = mean_squared_error(group["dividend"], group["dividend_pred"], squared=False)
        results.append({"sector": sector, "mae": mae, "rmse": rmse})

    return pd.DataFrame(results)

# Robustez por market cap
def robustness_by_marketcap(df_pred: pd.DataFrame, marketcap_map: dict) -> pd.DataFrame:
    """
    Avalia erro por market cap (faixa de tamanho da empresa).
    marketcap_map: dict {ticker: categoria}
    """
    df = df_pred.copy()
    df["marketcap"] = df["ticker"].map(marketcap_map).fillna("Desconhecido")

    results = []
    for mcap, group in df.groupby("marketcap"):
        mae = mean_absolute_error(group["dividend"], group["dividend_pred"])
        rmse = mean_squared_error(group["dividend"], group["dividend_pred"], squared=False)
        results.append({"marketcap": mcap, "mae": mae, "rmse": rmse})

    return pd.DataFrame(results)

# Robustez em períodos de crise
def robustness_crisis_periods(df_pred: pd.DataFrame, crises: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Avalia erro em períodos de crise.
    crises: lista de tuplas (início, fim) no formato 'YYYY-Qn'
    """
    results = []
    for start, end in crises:
        mask = (df_pred["quarter"] >= start) & (df_pred["quarter"] <= end)
        group = df_pred[mask]
        if group.empty:
            continue
        mae = mean_absolute_error(group["dividend"], group["dividend_pred"])
        rmse = mean_squared_error(group["dividend"], group["dividend_pred"], squared=False)
        results.append({"period": f"{start} to {end}", "mae": mae, "rmse": rmse})

    return pd.DataFrame(results)

# Robustez ao longo do tempo
def robustness_by_time(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Avalia erro por trimestre.
    """
    results = []
    for quarter, group in df_pred.groupby("quarter"):
        mae = mean_absolute_error(group["dividend"], group["dividend_pred"])
        rmse = mean_squared_error(group["dividend"], group["dividend_pred"], squared=False)
        results.append({"quarter": quarter, "mae": mae, "rmse": rmse})

    return pd.DataFrame(results)

# Gráfico de importância das features
def plot_feature_importance(importance: pd.Series):
    """Gráfico de importância das features"""
    fig, ax = plt.subplots()
    importance.plot(kind="bar", ax=ax)
    ax.set_ylabel("Importância")
    ax.set_title("Importância das Features")
    return fig

def plot_future_dividends(historical_df, forecast_df, ci_lower=None, ci_upper=None):
    """
    Plota dividendos históricos e futuros com intervalo de confiança e crescimento percentual.

    historical_df: DataFrame histórico, colunas ['quarter_dt', 'dividend']
    forecast_df: DataFrame futuro, colunas ['dividend_pred'], index com datas futuras
    ci_lower: Serie ou DataFrame, limite inferior do IC para previsão
    ci_upper: Serie ou DataFrame, limite superior do IC para previsão
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Linha histórica
    hist_col = 'dividend' if 'dividend' in historical_df.columns else 'dividend_real'
    ax1.plot(historical_df['quarter_dt'], historical_df[hist_col], label='Histórico', color='blue', linewidth=2)

    # Linha prevista
    ax1.plot(forecast_df.index, forecast_df['dividend_pred'], '--', label='Previsto', color='orange', linewidth=2)

    # Intervalo de confiança
    if ci_lower is not None and ci_upper is not None:
        ax1.fill_between(forecast_df.index, ci_lower, ci_upper, color='orange', alpha=0.2, label='IC 90%')

    ax1.set_xlabel("Trimestre")
    ax1.set_ylabel("Dividendos")
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Barras de crescimento percentual
    growth = forecast_df['dividend_pred'].pct_change().fillna(0) * 100
    ax2 = ax1.twinx()
    ax2.bar(forecast_df.index, growth, width=60, alpha=0.2, color='green', label='Crescimento %')
    ax2.set_ylabel("Crescimento %")
    ax2.legend(loc='upper right')

    plt.title("Dividendos Históricos e Previsões Futuras")
    plt.tight_layout()

    return fig