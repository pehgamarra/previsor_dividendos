import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import mean_absolute_error

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.modeling import bootstrap_ci, economic_eval, train_model, strategy_metrics, simulate_strategy
from src.data.fetch import fetch_data, fetch_price_history
from src.data.preprocess import preprocess_quarterly
from src.features.engineering import build_features, select_top_k
from src.evaluation.baselines import evaluate_baselines

# --------------------
# Configuração inicial
# --------------------
st.set_page_config(page_title="Previsor de Dividendos", page_icon="💰", layout="wide")

st.title("💰 Previsor de Dividendos")
st.markdown("Explore dividendos históricos de empresas e prepare terreno para previsões.")

# --------------------
# Sidebar - Inputs
# --------------------
st.sidebar.header("Configurações")

available_tickers = {
    "PETR4.SA": "Petrobras PN",
    "VALE3.SA": "Vale ON",
    "ITUB4.SA": "Itaú Unibanco PN",
    "BBAS3.SA": "Banco do Brasil ON",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "KO": "Coca-Cola",
    "TSLA": "Tesla",
}

ticker = st.sidebar.selectbox(
    "Escolha um ticker:",
    options=list(available_tickers.keys()),
    format_func=lambda x: f"{x} - {available_tickers[x]}"
)

years = st.sidebar.slider(
    "Período do histórico (anos):",
    min_value=1,
    max_value=30,
    value=10,
    step=1
)
period = f"{years}y"
run = st.sidebar.button("🔎 Analisar")

# --------------------
# Execução
# --------------------
if "run_analysis" not in st.session_state:
    with st.spinner("Coletando dados..."):
        raw = fetch_data(ticker, period)
        quarterly = preprocess_quarterly(raw, ticker)
        features = build_features(quarterly)
        results = evaluate_baselines(quarterly, n_splits=5)
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
    st.success(f" ### {ticker} - {available_tickers[ticker]}")

    # --------------------
    # Abas
    # --------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Dividendos", "🏢 Empresa", "📖 Glossário & Dicas", "🖥️Modelagem", "🧹Backtest"])

    # --------------------
    # Aba 1: Dividendos
    # --------------------
    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📈 Dividendos ao longo do tempo")
            st.line_chart(
                quarterly.set_index("quarter")[["dividend", "close"]],
                use_container_width=True
            )

        with col2:
            st.subheader("🔢 Estatísticas")
            total = quarterly["dividend"].sum()
            avg_yield = quarterly["dividend_yield"].mean() * 100
            freq = quarterly["has_dividend"].mean() * 100

            st.metric("Dividendos pagos (total)", f"{total:.2f}")
            st.metric("Yield médio", f"{avg_yield:.2f}%")
            st.metric("Frequência de pagamentos", f"{freq:.1f}% dos trimestres")

        st.subheader("📑 Histórico Trimestral")
        df_display = quarterly.copy()
        df_display["quarter"] = df_display["quarter"].astype(str)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.download_button(
            label="💾 Baixar CSV",
            data=quarterly.to_csv(index=False),
            file_name=f"{ticker}_dividends.csv",
            mime="text/csv"
        )

    with tab2:
        st.subheader("🏢 Informações da Empresa")

        # --------------------
        # Resumo rápido
        # --------------------
        st.write(f"**Nome:** {info.get('longName', ticker)}")
        st.write(f"**Setor:** {info.get('sector', 'N/A')}")
        st.write(f"**Indústria:** {info.get('industry', 'N/A')}")
        st.write(f"**País:** {info.get('country', 'N/A')} | **Moeda:** {info.get('currency', 'N/A')}")

        st.metric("Preço atual", f"R$ {info.get('currentPrice', 'N/A')}")

        # --------------------
        # Saúde financeira
        # --------------------
        st.markdown("---")
        st.markdown("### 🏦 Saúde Financeira")
        col1, col2, col3 = st.columns(3)
        col1.metric("Valor de mercado", f"{info.get('marketCap', 'N/A'):,}")
        col2.metric("Receita Total", f"{info.get('totalRevenue', 'N/A'):,}")
        col3.metric("Margem Bruta", f"{info.get('grossMargins', 0)*100:.1f}%")
        col1.metric("Margem Operacional", f"{info.get('operatingMargins', 0)*100:.1f}%")
        col2.metric("Dívida/Patrimônio", f"{info.get('debtToEquity', 'N/A')}")
        col3.metric("Enterprise Value", f"{info.get('enterpriseValue', 'N/A'):,}")

        # --------------------
        # Valuation
        # --------------------
        st.markdown("---")
        st.markdown("### 💵 Valuation")
        col1, col2, col3 = st.columns(3)
        col1.metric("P/L (PE)", f"{info.get('trailingPE', 'N/A')}")
        col2.metric("P/VP (Price/Book)", f"{info.get('priceToBook', 'N/A')}")
        col3.metric("Forward P/E", f"{info.get('forwardPE', 'N/A')}")

        # --------------------
        # Risco e retorno
        # --------------------
        st.markdown("---")
        st.markdown("### 📊 Risco e Retorno")
        col1, col2, col3 = st.columns(3)
        col1.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
        col2.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.1f}%")
        col3.metric("Beta", f"{info.get('beta', 'N/A')}")
        st.markdown("---")
    # --------------------
    # Aba 3: Glossário & Dicas
    # --------------------

    with tab3:
        st.subheader("📖 Glossário de Métricas")

        glossario = {
            "Preço atual": "Quanto custa uma ação hoje no mercado.",
            "Valor de mercado (Market Cap)": "Valor total da empresa na bolsa (preço da ação x número de ações).",
            "Receita Total": "Quanto a empresa vendeu em dinheiro (faturamento).",
            "Margem Bruta": "Quanto sobra da receita depois de pagar custos diretos (quanto maior, melhor).",
            "Margem Operacional": "Lucro da empresa sobre as vendas após despesas operacionais.",
            "Dívida/Patrimônio": "Quanto a empresa deve em relação ao que ela tem. Número alto pode indicar risco.",
            "Enterprise Value": "Valor total da empresa incluindo dívidas. Usado em comparações.",
            "P/L (Preço/Lucro)": "Quantos anos de lucro a empresa precisaria para valer seu preço atual. Quanto menor, mais barata.",
            "P/VP (Preço/Valor Patrimonial)": "Mostra se a ação vale mais ou menos do que o patrimônio da empresa.",
            "Forward P/E": "P/L projetado com base nos lucros futuros estimados.",
            "Dividend Yield": "Quanto a empresa paga em dividendos em relação ao preço da ação (como se fosse 'juros').",
            "ROE (Retorno sobre Patrimônio)": "Mede se a empresa usa bem o dinheiro dos acionistas para gerar lucro.",
            "Beta": "Mostra o risco da ação comparado ao mercado. 1 = acompanha o mercado, >1 = mais volátil."
        }

        for metrica, explicacao in glossario.items():
            st.markdown(f"**{metrica}:** {explicacao}")

        # --------------------
        # Informações Úteis
        # --------------------
        st.markdown("---")
        st.subheader("ℹ️ Informações Úteis para Investidores")

        st.markdown("""
        - **Dividendos não são garantidos**: empresas podem reduzir ou suspender pagamentos em crises.  
        - **Ciclo do setor importa**: setores de commodities (petróleo, mineração) têm dividendos mais voláteis.  
        - **Empresas de utilidade pública e bancos** costumam ter dividendos mais estáveis.  
        - **Acompanhe o payout ratio** (quanto do lucro é distribuído) para saber se o nível de dividendos é sustentável.  
        """)

        # --------------------
        # Dicas específicas do ativo
        # --------------------
        st.markdown("---")
        st.subheader(f"💡 Dicas de Investimento para {available_tickers[ticker]}")

        if "PETR" in ticker:
            st.markdown("Petrobras depende muito do preço do petróleo e das decisões do governo. Boa pagadora de dividendos, mas volátil.")
        elif "VALE" in ticker:
            st.markdown("Vale é cíclica, ligada ao preço do minério de ferro. Dividendos altos em épocas de boom, mas pode cair bastante em crises.")
        elif "ITUB" in ticker or "BBAS" in ticker:
            st.markdown("Bancos brasileiros são tradicionalmente bons pagadores de dividendos, mas precisam ser avaliados em relação à economia e juros.")
        elif ticker in ["AAPL", "MSFT", "TSLA", "KO"]:
            st.markdown("Empresas americanas de tecnologia tendem a reinvestir lucros em crescimento, pagando menos dividendos (exceto Coca-Cola).")
        else:
            st.markdown("Analise o histórico de dividendos, compare com empresas do mesmo setor e avalie se o perfil de risco combina com você.")

   # --------------------
    # Aba 4: Modelagem
    # -------------------- 
    with tab4:
        st.header("🤖 Modelagem de Dividendos")
        st.markdown("Treinamento com o modelo Ridge avalia previsões com intervalos de confiança e impacto econômico.")

        n_splits = 10  # TimeSeriesSplit fixox

        # ------------------------
        # Preparar features
        # ------------------------
        features = build_features(quarterly)
        features = features.dropna(axis=1, how="all")
        features = features.fillna(features.median(numeric_only=True))
        X = features.drop(columns=["dividend", "quarter"])
        y = features["dividend"]

        # ------------------------
        # Treinar modelo
        # ------------------------
        with st.spinner("Treinando modelo Ridge..."):
            preds, mae = train_model(X, y, model_type="ridge", n_splits=n_splits)
            preds = preds.ffill().bfill()  # evita FutureWarning

            df_pred = pd.DataFrame({
                "quarter": features["quarter"],
                "dividend_real": y,
                "dividend_pred": preds
            })

        # Intervalos de confiança
        lower, upper = bootstrap_ci(df_pred["dividend_real"], df_pred["dividend_pred"])
        df_pred["ci_lower"] = lower
        df_pred["ci_upper"] = upper
        df_pred["quarter_dt"] = df_pred["quarter"].dt.to_timestamp()

        # Avaliação econômica
        mean_ret, sharpe, max_drawdown = economic_eval(df_pred["dividend_real"], df_pred["dividend_pred"])
        st.markdown("---")
        mae_val = mean_absolute_error(df_pred['dividend_real'], df_pred['dividend_pred'])
        mean_ret_pct = mean_ret * 100
        sharpe_val = sharpe
        max_dd_pct = max_drawdown * 100

        # Exibir 
        st.markdown("### Resultados")
        st.metric("MAE (aprox)", f"{mae_val:.4f}")
        st.metric("Retorno médio", f"{mean_ret_pct:.2f}%")
        st.metric("Sharpe ratio", f"{sharpe_val:.2f}")
        st.metric("Max Drawdown", f"{max_dd_pct:.2f}%")
        st.markdown("---")

        #Gráfico de previsões x real
        st.subheader("📊 Previsões x Real com intervalo de confiança")
        st.line_chart(df_pred.set_index("quarter_dt")[["dividend_real", "dividend_pred"]])
        # Tabela de previsões
        with st.expander("📋 Mostrar tabela completa de previsões"):
            st.dataframe(df_pred, use_container_width=True)   
        # Download
        st.download_button(
            "💾 Baixar previsões CSV",
            data=df_pred.to_csv(index=False),
            file_name=f"{ticker}_predictions.csv",
            mime="text/csv"
        )

        # --------------------
        # Aba 5: Backtest & Estratégia (versão final, sem debug)
        # --------------------
        with tab5:
            st.header("📈 Backtest da Estratégia de Dividendos")
            st.markdown("""
            Estratégia automática: a cada período, selecionamos os top-3 tickers por previsão de dividendos,
            mantemos por 30 dias, e comparamos com benchmark.
            """)

            # Config
            top_k = 3
            hold_period = 30  # em dias úteis (posições no índice de preços)
            start_date = pd.to_datetime("2018-01-01")
            end_date = pd.to_datetime("today")
            tickers_bt = list(available_tickers.keys())
            st.write(f"Tickers incluídos no backtest: {tickers_bt}")

            with st.spinner("Rodando backtest..."):
                # 1) coletar preços (formato long -> wide)
                df_prices = fetch_price_history(tickers_bt, start=start_date, end=end_date)
                df_prices["date"] = pd.to_datetime(df_prices["date"])
                df_prices = df_prices.sort_values(["ticker", "date"])
                df_prices_wide = df_prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()

                # 2) criar previsões por ticker (mesma lógica sua)
                df_preds_list = []
                for ticker in tickers_bt:
                    raw = fetch_data(ticker, f"{years}y")
                    quarterly = preprocess_quarterly(raw, ticker)
                    features = build_features(quarterly).dropna(axis=1, how="all")
                    features = features.fillna(features.median(numeric_only=True))
                    X = features.drop(columns=["dividend", "quarter"])
                    y = features["dividend"]
                    preds, _ = train_model(X, y, model_type="ridge", n_splits=10)
                    df_pred = pd.DataFrame({
                        "quarter": features["quarter"],
                        "ticker": ticker,
                        "dividend_pred": preds
                    })
                    df_preds_list.append(df_pred)

                df_all_preds = pd.concat(df_preds_list, ignore_index=True)

                # 3) selecionar top-k por período
                df_topk = select_top_k(df_all_preds, k=top_k, by="dividend_pred")

                # 4) função robusta de simulação (gera retornos DIÁRIOS, agrupa overlaps)
                def simulate_strategy_v2(df_wide, df_topk, hold_period=30):
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

                    if not series_list:
                        return pd.Series(dtype=float)

                    # concat e agrupa por data (média quando houver overlap), ordena
                    combined = pd.concat(series_list).groupby(level=0).mean().sort_index()
                    combined.name = "strategy_return"
                    return combined

                # 5) simular estratégia (usa df_prices_wide)
                strategy_returns = simulate_strategy_v2(df_prices_wide, df_topk, hold_period=hold_period)

                # 6) benchmark (IBOV)
                benchmark_prices = fetch_price_history(["^BVSP"], start=start_date, end=end_date)
                benchmark_prices["date"] = pd.to_datetime(benchmark_prices["date"])
                benchmark_prices = benchmark_prices.set_index("date").sort_index()
                benchmark_returns = benchmark_prices["adj_close"].pct_change().dropna()

                # 7) métricas (CAGR, Sharpe e Max Drawdown) - sem debug
                def compute_metrics(returns):
                    if returns.empty:
                        return {"CAGR": np.nan, "Sharpe": np.nan, "Max Drawdown": np.nan}

                    # n_years a partir do período coberto pela série
                    if isinstance(returns.index, pd.DatetimeIndex) and len(returns.index) > 1:
                        days = (returns.index.max() - returns.index.min()).days
                        n_years = max(days / 365.25, 1/252)
                    else:
                        n_years = max(len(returns) / 252, 1/252)

                    total_return = (1 + returns).prod()
                    CAGR = total_return ** (1 / n_years) - 1

                    # Sharpe annualizado (assume risk-free = 0)
                    sharpe = returns.mean() / (returns.std() if returns.std() > 0 else np.nan) * np.sqrt(252)

                    # Max Drawdown
                    cumulative = (1 + returns).cumprod()
                    max_dd = (cumulative / cumulative.cummax() - 1).min()

                    return {"CAGR": CAGR, "Sharpe": sharpe, "Max Drawdown": max_dd}

                strategy_metrics_dict = compute_metrics(strategy_returns)
                benchmark_metrics_dict = compute_metrics(benchmark_returns)

            # --------------------
            # Exibir resultados (limpo, sem debug)
            # --------------------
            # formatar: CAGR e Max Drawdown em %, Sharpe sem %
            df_metrics = pd.DataFrame([strategy_metrics_dict, benchmark_metrics_dict],
                                    index=["Estratégia", "Benchmark"])

            display = df_metrics.copy()
            # formatar percentuais
            display["CAGR"] = (display["CAGR"] * 100).round(2)
            display["Max Drawdown"] = (display["Max Drawdown"] * 100).round(2)
            display["Sharpe"] = display["Sharpe"].round(3)

            st.subheader("📊 Métricas da Estratégia vs Benchmark")
            st.table(display)

            # --------------------
            # Gráfico de retorno acumulado (alinha índices)
            # --------------------
            all_idx = strategy_returns.index.union(benchmark_returns.index).sort_values()
            cum_strategy = (1 + strategy_returns.reindex(all_idx).fillna(0)).cumprod()
            cum_benchmark = (1 + benchmark_returns.reindex(all_idx).fillna(0)).cumprod()

            cum_df = pd.DataFrame({
                "Estratégia": cum_strategy,
                "Benchmark": cum_benchmark
            }, index=all_idx)

            st.subheader("📈 Retorno Acumulado")
            st.line_chart(cum_df)

            # --------------------
            # Top-k por período (tabela)
            # --------------------
            st.subheader("📋 Top-K Tickers por Período")
            with st.expander("Mostrar tabela completa"):
                st.dataframe(df_topk, use_container_width=True)


else:
    st.info("Selecione um ticker na barra lateral e clique em **Analisar**.")
