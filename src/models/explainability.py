import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.isotonic import IsotonicRegression



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
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_title("Dividendos Históricos e Previsões Futuras")

    plt.title("Dividendos Históricos e Previsões Futuras")
    plt.tight_layout()

    return fig

def postproc_isotonic_calibrate(df_pred, forecast_df, clip_min=0.0):

    """
    df_pred: DataFrame histórico com colunas 'dividend_real' e 'dividend_pred' (resultado do cross-val)
    forecast_df: DataFrame futuro com coluna 'dividend_pred'
    Retorna forecast calibrated
    """
    # 1) remover NaNs
    df_cal = df_pred.dropna(subset=["dividend_real", "dividend_pred"]).copy()
    if df_cal.empty:
        # nada para calibrar; retorna original
        return forecast_df.copy()
    
    # 2) treinar isotonic (monotonic non-decreasing mapping)
    ir = IsotonicRegression(out_of_bounds="clip")
    try:
        ir.fit(df_cal["dividend_pred"].values, df_cal["dividend_real"].values)
    except Exception:
        return forecast_df.copy()
    
    # 3) aplicar ao forecast
    calibrated = ir.predict(forecast_df["dividend_pred"].values)

    # Substitui NaNs (caso extrapole) por extrapolação linear suave
    if np.isnan(calibrated).any():
        valid_idx = ~np.isnan(calibrated)
        calibrated = pd.Series(calibrated).interpolate(method="linear", limit_direction="both").values

    calibrated = np.clip(calibrated, clip_min, None) if clip_min is not None else calibrated
    
    # 4) ensemble smoothing leve
    alpha = 0.8  # peso do isotônico vs. original
    smoothed = alpha * calibrated + (1 - alpha) * forecast_df["dividend_pred"].values

    out = pd.DataFrame({"dividend_pred": smoothed}, index=forecast_df.index)
    return out

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

# ===========================
# FUNÇÃO AUXILIAR
# ===========================
def get_dividend_column(df):
    """Detecta automaticamente a coluna de dividendos"""
    for col in ['dividend', 'dividend_real', 'dividend_pred', 'value']:
        if col in df.columns:
            return col
    # Se não encontrar, tenta a primeira coluna numérica
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]

def get_date_column(df):
    """Detecta automaticamente a coluna de data"""
    for col in ['quarter_dt', 'quarter', 'date', 'period']:
        if col in df.columns:
            return col
    # Se não encontrar, retorna o index se for datetime
    if isinstance(df.index, pd.DatetimeIndex):
        return None  # Usar o index
    return df.columns[0]

# ===========================
# 1. DASHBOARD COMPLETO
# ===========================
def plot_dividend_dashboard(historical_df, forecast_df):
    """
    Dashboard completo com múltiplas visualizações e métricas
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    hist_col = get_dividend_column(historical_df)
    hist_date_col = get_date_column(historical_df)
    
    # --- Gráfico Principal: Timeline ---
    ax1 = fig.add_subplot(gs[0:2, :])
    
    # Histórico
    if hist_date_col:
        hist_x = historical_df[hist_date_col]
    else:
        hist_x = historical_df.index
    
    ax1.plot(hist_x, historical_df[hist_col], 
             label='Histórico', color='#2E86AB', linewidth=2.5, marker='o', markersize=4)
    
    # Previsão
    ax1.plot(forecast_df.index, forecast_df['dividend_pred'], 
             '--', label='Previsão', color='#A23B72', linewidth=2.5, marker='s', markersize=4)
    
    # Linha de hoje
    last_date = hist_x.iloc[-1] if isinstance(hist_x, pd.Series) else hist_x[-1]
    ax1.axvline(x=last_date, color='red', 
                linestyle=':', linewidth=2, alpha=0.6, label='Hoje')
    ax1.set_xlabel("Período", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Dividendos (R$)", fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_title("📈 Evolução e Previsão de Dividendos", fontsize=14, fontweight='bold', pad=20)
    
    # --- KPI Cards ---
    last_hist = historical_df[hist_col].iloc[-1]
    avg_forecast = forecast_df['dividend_pred'].mean()
    total_forecast = forecast_df['dividend_pred'].sum()
    growth = ((avg_forecast - last_hist) / last_hist) * 100
    
    # Card 1: Último Dividendo
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.text(0.5, 0.7, f"R$ {last_hist:.2f}", ha='center', va='center', 
             fontsize=20, fontweight='bold', color='#2E86AB')
    ax2.text(0.5, 0.3, "Último Dividendo", ha='center', va='center', 
             fontsize=10, color='gray')
    ax2.axis('off')
    ax2.add_patch(Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, 
                            edgecolor='#2E86AB', linewidth=2))
    
    # Card 2: Média Prevista
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.text(0.5, 0.7, f"R$ {avg_forecast:.2f}", ha='center', va='center', 
             fontsize=20, fontweight='bold', color='#A23B72')
    ax3.text(0.5, 0.3, "Média Prevista", ha='center', va='center', 
             fontsize=10, color='gray')
    ax3.axis('off')
    ax3.add_patch(Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, 
                            edgecolor='#A23B72', linewidth=2))
    
    # Card 3: Crescimento
    color = '#27AE60' if growth >= 0 else '#E74C3C'
    ax4 = fig.add_subplot(gs[2, 2])
    ax4.text(0.5, 0.7, f"{growth:+.1f}%", ha='center', va='center', 
             fontsize=20, fontweight='bold', color=color)
    ax4.text(0.5, 0.3, "Crescimento Médio", ha='center', va='center', 
             fontsize=10, color='gray')
    ax4.axis('off')
    ax4.add_patch(Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, 
                            edgecolor=color, linewidth=2))
    
    plt.suptitle("Dashboard de Análise de Dividendos", fontsize=16, 
                 fontweight='bold', y=0.98)
    
    return fig


# ===========================
# 2. GRÁFICO DE ÁREA EMPILHADA
# ===========================
def plot_area_chart(historical_df, forecast_df):
    """
    Gráfico de área mostrando acumulação de dividendos
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    hist_col = get_dividend_column(historical_df)
    hist_date_col = get_date_column(historical_df)
    
    # Histórico
    if hist_date_col:
        hist_x = historical_df[hist_date_col]
    else:
        hist_x = historical_df.index
    
    # Área histórica
    ax.fill_between(hist_x, 0, historical_df[hist_col], 
                    color='#3498DB', alpha=0.6, label='Histórico')
    ax.plot(hist_x, historical_df[hist_col], 
            color='#2C3E50', linewidth=2)
    
    # Área prevista
    ax.fill_between(forecast_df.index, 0, forecast_df['dividend_pred'], 
                    color='#E67E22', alpha=0.6, label='Previsão')
    ax.plot(forecast_df.index, forecast_df['dividend_pred'], 
            color='#D35400', linewidth=2, linestyle='--')
    
    # Linha de hoje
    last_date = hist_x.iloc[-1] if isinstance(hist_x, pd.Series) else hist_x[-1]
    ax.axvline(x=last_date, color='red', 
               linestyle=':', linewidth=2, alpha=0.7)
    
    ax.set_xlabel("Período", fontsize=12, fontweight='bold')
    ax.set_ylabel("Dividendos (R$)", fontsize=12, fontweight='bold')
    ax.set_title("📊 Evolução de Dividendos - Visualização em Área", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


# ===========================
# 3. GRÁFICO DE BARRAS COMPARATIVO
# ===========================
def plot_bar_comparison(historical_df, forecast_df):
    """
    Comparação em barras dos últimos períodos históricos vs previsões
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    hist_col = get_dividend_column(historical_df)
    hist_date_col = get_date_column(historical_df)
    
    # Pegar últimos 8 períodos históricos
    hist_recent = historical_df.tail(8).copy()
    hist_recent['tipo'] = 'Histórico'
    hist_recent['valor'] = hist_recent[hist_col]
    
    # Obter períodos
    if hist_date_col:
        if pd.api.types.is_period_dtype(historical_df[hist_date_col]):
            hist_recent['periodo'] = hist_recent[hist_date_col].astype(str)
        else:
            hist_recent['periodo'] = pd.to_datetime(hist_recent[hist_date_col]).dt.to_period('Q').astype(str)
    else:
        hist_recent['periodo'] = pd.to_datetime(hist_recent.index).to_period('Q').astype(str)
    
    # Preparar previsões
    forecast_plot = forecast_df.head(8).copy()
    forecast_plot['tipo'] = 'Previsão'
    forecast_plot['valor'] = forecast_plot['dividend_pred']
    forecast_plot['periodo'] = forecast_plot.index.to_period('Q').astype(str)
    
    # Combinar
    combined = pd.concat([
        hist_recent[['periodo', 'valor', 'tipo']], 
        forecast_plot[['periodo', 'valor', 'tipo']]
    ])
    
    # Plot
    colors = {'Histórico': '#3498DB', 'Previsão': '#E67E22'}
    x = np.arange(len(combined))
    bars = ax.bar(x, combined['valor'], color=[colors[t] for t in combined['tipo']], 
                  edgecolor='black', linewidth=1.2, alpha=0.8)
    
    # Adicionar valores nas barras
    for i, (bar, val) in enumerate(zip(bars, combined['valor'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'R$ {val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Linha de separação
    sep_idx = len(hist_recent) - 0.5
    ax.axvline(x=sep_idx, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(sep_idx, ax.get_ylim()[1]*0.95, 'HOJE', ha='center', 
            fontsize=10, fontweight='bold', color='red')
    
    ax.set_xticks(x)
    ax.set_xticklabels(combined['periodo'], rotation=45, ha='right')
    ax.set_xlabel("Período", fontsize=12, fontweight='bold')
    ax.set_ylabel("Dividendos (R$)", fontsize=12, fontweight='bold')
    ax.set_title("📊 Comparação: Histórico vs Previsão", 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Legenda customizada
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498DB', label='Histórico'),
                      Patch(facecolor='#E67E22', label='Previsão')]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


# ===========================
# 4. GRÁFICO DE CRESCIMENTO %
# ===========================
def plot_growth_rate(historical_df, forecast_df):
    """
    Visualiza a taxa de crescimento trimestral
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    hist_col = get_dividend_column(historical_df)
    hist_date_col = get_date_column(historical_df)
    
    # Obter datas
    if hist_date_col:
        hist_dates = historical_df[hist_date_col]
    else:
        hist_dates = historical_df.index
    
    # Calcular crescimento histórico
    hist_growth = historical_df[hist_col].pct_change() * 100
    
    # Calcular crescimento previsto
    forecast_growth = forecast_df['dividend_pred'].pct_change() * 100
    
    # --- Painel 1: Valores Absolutos ---
    ax1.plot(hist_dates, historical_df[hist_col], 'o-', 
             label='Histórico', color='#2E86AB', linewidth=2, markersize=5)
    ax1.plot(forecast_df.index, forecast_df['dividend_pred'], 's--', 
             label='Previsão', color='#A23B72', linewidth=2, markersize=5)
    ax1.axvline(x=hist_dates.iloc[-1], color='red', linestyle=':', linewidth=2, alpha=0.6)
    ax1.set_ylabel("Dividendos (R$)", fontsize=11, fontweight='bold')
    ax1.set_title("📈 Dividendos e Taxa de Crescimento", fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # --- Painel 2: Crescimento % ---
    colors_hist = ['#27AE60' if x >= 0 else '#E74C3C' for x in hist_growth]
    colors_fore = ['#27AE60' if x >= 0 else '#E74C3C' for x in forecast_growth]
    
    ax2.bar(hist_dates.iloc[1:] if isinstance(hist_dates, pd.Series) else hist_dates[1:], 
            hist_growth.iloc[1:], color=colors_hist[1:], 
            alpha=0.7, edgecolor='black', linewidth=1, label='Histórico')
    ax2.bar(forecast_df.index[1:], forecast_growth[1:], color=colors_fore[1:], 
            alpha=0.7, edgecolor='black', linewidth=1, label='Previsão', hatch='//')
    
    ax2.axhline(y=0, color='black', linewidth=1)
    last_date = hist_dates.iloc[-1] if isinstance(hist_dates, pd.Series) else hist_dates[-1]
    ax2.axvline(x=last_date, color='red', linestyle=':', linewidth=2, alpha=0.6)
    ax2.set_xlabel("Período", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Crescimento (%)", fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


# ===========================
# 5. TABELA INTERATIVA
# ===========================
def plot_forecast_table(forecast_df, last_historical_value):
    """
    Cria uma visualização em formato de tabela com métricas
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Preparar dados
    table_data = []
    cumulative = 0
    
    for i, (date, row) in enumerate(forecast_df.iterrows()):
        pred_val = row['dividend_pred']
        cumulative += pred_val
        
        if i == 0:
            growth = ((pred_val - last_historical_value) / last_historical_value) * 100
        else:
            growth = ((pred_val - forecast_df['dividend_pred'].iloc[i-1]) / 
                     forecast_df['dividend_pred'].iloc[i-1]) * 100
        
        period = date.strftime('%Y-Q%q') if hasattr(date, 'strftime') else str(date.to_period('Q'))
        
        table_data.append([
            period,
            f"R$ {pred_val:.2f}",
            f"{growth:+.1f}%",
            f"R$ {cumulative:.2f}"
        ])
    
    # Criar tabela
    columns = ['Período', 'Dividendo Previsto', 'Crescimento', 'Acumulado']
    table = ax.table(cellText=table_data, colLabels=columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.2, 0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Estilizar header
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(weight='bold', color='white')
    
    # Estilizar linhas alternadas
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ECF0F1')
            else:
                cell.set_facecolor('white')
            
            # Colorir crescimento
            if j == 2:
                growth_val = float(table_data[i-1][2].replace('%', '').replace('+', ''))
                if growth_val >= 0:
                    cell.set_text_props(color='#27AE60', weight='bold')
                else:
                    cell.set_text_props(color='#E74C3C', weight='bold')
    
    plt.title("📋 Tabela Detalhada de Previsões", fontsize=14, 
              fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig


# ===========================
# 6. CONE DE INCERTEZA (CENÁRIOS)
# ===========================
def plot_uncertainty_cone(historical_df, forecast_df):
    """
    Visualiza múltiplos cenários com cone de incerteza crescente
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    hist_col = get_dividend_column(historical_df)
    hist_date_col = get_date_column(historical_df)
    
    # Obter datas
    if hist_date_col:
        hist_x = historical_df[hist_date_col]
    else:
        hist_x = historical_df.index
    
    # Histórico
    ax.plot(hist_x, historical_df[hist_col], 
            'o-', label='Histórico', color='#2C3E50', linewidth=2.5, markersize=6)
    
    # Cenário base
    ax.plot(forecast_df.index, forecast_df['dividend_pred'], 
            's-', label='Cenário Base', color='#3498DB', linewidth=2.5, markersize=6)
    
    # Cenário otimista (+15%)
    optimistic = forecast_df['dividend_pred'] * 1.15
    ax.plot(forecast_df.index, optimistic, 
            '^--', label='Cenário Otimista (+15%)', color='#27AE60', 
            linewidth=2, markersize=5, alpha=0.8)
    
    # Cenário pessimista (-15%)
    pessimistic = forecast_df['dividend_pred'] * 0.85
    ax.plot(forecast_df.index, pessimistic, 
            'v--', label='Cenário Pessimista (-15%)', color='#E74C3C', 
            linewidth=2, markersize=5, alpha=0.8)
    
    # Área entre otimista e pessimista
    ax.fill_between(forecast_df.index, pessimistic, optimistic, 
                    color='gray', alpha=0.1)
    
    # Linha de hoje
    last_date = hist_x.iloc[-1] if isinstance(hist_x, pd.Series) else hist_x[-1]
    ax.axvline(x=last_date, color='red', 
               linestyle=':', linewidth=2, alpha=0.7, label='Hoje')
    
    ax.set_xlabel("Período", fontsize=12, fontweight='bold')
    ax.set_ylabel("Dividendos (R$)", fontsize=12, fontweight='bold')
    ax.set_title("🎯 Projeção com Múltiplos Cenários", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig
