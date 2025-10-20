import sys
import os
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import mean_absolute_error

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.modeling import bootstrap_ci, economic_eval, train_model, simulate_strategy, forecast_future
from src.data.fetch import fetch_data, fetch_price_history
from src.data.preprocess import preprocess_quarterly
from src.features.engineering import build_features, select_top_k, compute_metrics
from src.evaluation.baselines import evaluate_baselines
from src.evaluation.robustness import eval_robustness
from src.models.explainability import *

# --------------------
# Configuração inicial do Streamlit
# --------------------
st.set_page_config(page_title="Previsor de Dividendos", page_icon="💰", layout="wide")

st.title("💰 Previsor de Dividendos")
st.markdown("Explore dividendos históricos de empresas e prepare terreno para previsões.")

# --------------------
# Sidebar - Inputs
# --------------------

available_tickers = {
    "PETR4.SA": "Petrobras PN",
    "VALE3.SA": "Vale ON",
    "ITUB4.SA": "Itaú Unibanco PN",
    "BBAS3.SA": "Banco do Brasil ON",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "KO": "Coca-Cola",
}

ticker = st.sidebar.selectbox(
    "Escolha um ticker:",
    options=list(available_tickers.keys()),
    format_func=lambda x: f"{x} - {available_tickers[x]}"
)

st.sidebar.markdown("""
---""")

years = 10
period = f"{years}y"

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

    st.sidebar.subheader("🏢 Dados da Empresa")
    st.sidebar.write(f"**Nome:** {info.get('longName', ticker)}")
    st.sidebar.write(f"**Setor:** {info.get('sector', 'N/A')}")
    st.sidebar.write(f"**Indústria:** {info.get('industry', 'N/A')}")
    st.sidebar.write(f"**País:** {info.get('country', 'N/A')}") 
    st.sidebar.write(f"**Moeda:** {info.get('currency', 'N/A')}")
    st.sidebar.metric("Preço atual", f"R$ {info.get('currentPrice', 'N/A')} ")
    st.sidebar.markdown("Coletado via [yfinance](https://pypi.org/project/yfinance/) 🔎")
    

    # --------------------
    # Abas
    # --------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Dividendos", "🏢 Empresa", "🧪 Validação do Modelo", "📖 Glossário & Dicas", "🔮 Previsões"])

    # --------------------
    # Aba 1: Dividendos
    # --------------------
    with tab1:
        col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Dividendos ao longo do tempo")

        # Se faltar 'quarter' mas houver 'date', tentamos reconstruir (não quebra o app)
        if "quarter" not in quarterly.columns and "date" in quarterly.columns:
            quarterly["quarter"] = pd.to_datetime(quarterly["date"], errors="coerce").dt.to_period("Q")

        if "quarter" not in quarterly.columns:
            st.warning("Coluna 'quarter' não encontrada — impossível plotar série temporal.")
        else:
            # Usar uma cópia só para o plot, para NÃO mexer no DataFrame original
            plot_df = quarterly.copy()

            # Converter Period -> Timestamp ou strings -> datetime com segurança
            if str(plot_df["quarter"].dtype).startswith("period"):
                plot_df["quarter"] = plot_df["quarter"].dt.to_timestamp()
            else:
                plot_df["quarter"] = pd.to_datetime(plot_df["quarter"], errors="coerce")

            # Ordenar e usar o índice apenas na cópia para o gráfico
            plot_df = plot_df.sort_values("quarter").set_index("quarter")

            cols_to_plot = [c for c in ["dividend", "close"] if c in plot_df.columns]
            if cols_to_plot:
                st.line_chart(plot_df[cols_to_plot], use_container_width=True)
            else:
                st.warning("Colunas 'dividend' e 'close' não encontradas para plotar.")


        with col2:
            st.subheader("🔢 Estatísticas")
            total = quarterly["dividend"].sum() if "dividend" in quarterly else 0
            avg_yield = quarterly["dividend_yield"].mean() * 100 if "dividend_yield" in quarterly else 0
            freq = quarterly["has_dividend"].mean() * 100 if "has_dividend" in quarterly else 0

            st.metric("Dividendos pagos (total)", f"{total:.2f}")
            st.metric("Yield médio", f"{avg_yield:.2f}%")
            st.metric("Frequência de pagamentos", f"{freq:.1f}% dos trimestres")


        st.subheader("📑 Histórico Trimestral")
        df_display = quarterly.copy()
        df_display["quarter"] = df_display["quarter"].astype(str)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    with tab2:
        # --------------------
        # Saúde financeira
        # --------------------
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
    # Aba 3: Modelagem
    # -------------------- 
    with tab3:
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
            preds, mae, model = train_model(X, y, model_type="ridge", n_splits=n_splits)
            preds = preds.ffill().bfill()

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
        mae_val = mean_absolute_error(df_pred['dividend_real'], df_pred['dividend_pred'])
        mean_ret_pct = mean_ret * 100
        sharpe_val = sharpe
        max_dd_pct = max_drawdown * 100

        with st.expander("🔎 Resultado de Modelagem de Dividendos", expanded=False):   
            # Exibir 
            st.markdown("### Resultados")
            st.metric("MAE (aprox)", f"{mae_val:.4f}")
            st.metric("Retorno médio", f"{mean_ret_pct:.2f}%")
            st.metric("Sharpe ratio", f"{sharpe_val:.2f}")
            st.metric("Max Drawdown", f"{max_dd_pct:.2f}%")
            st.markdown("---")

            #Gráfico de previsões x real
            st.subheader("Previsões x Real com intervalo de confiança")
            st.line_chart(df_pred.set_index("quarter_dt")[["dividend_real", "dividend_pred"]])
            # Tabela de previsões
            with st.expander("Mostrar tabela completa de previsões"):
                st.dataframe(df_pred, use_container_width=True)   
            # Download
            st.download_button(
                "💾 Baixar previsões CSV",
                data=df_pred.to_csv(index=False),
                file_name=f"{ticker}_predictions.csv",
                mime="text/csv"
            )

        #Backtest & Estratégia
        st.markdown("---")
        st.header("📈 Backtest da Estratégia de Dividendos")
        st.markdown("""
        Estratégia automática: a cada período, selecionamos os top-3 tickers por previsão de dividendos,
        mantemos por 30 dias, e comparamos com benchmark. ***Importante:*** O ticker não altera o resultado do backtest,
        que é baseado na seleção dos top-3 dentre todos os disponíveis.""")
        with st.expander("🔎 Resultado Backtest da Estratégia de Dividendos", expanded=False):   
            # Config
            top_k = 3
            hold_period = 30
            start_date = pd.to_datetime("2018-01-01")
            end_date = pd.to_datetime("today")
            tickers_bt = list(available_tickers.keys())
            
            # Rodar backtest
            with st.spinner("Rodando backtest..."):
                # coletar preços (formato long -> wide)
                df_prices = fetch_price_history(tickers_bt, start=start_date, end=end_date)
                df_prices["date"] = pd.to_datetime(df_prices["date"])
                df_prices = df_prices.sort_values(["ticker", "date"])
                df_prices_wide = df_prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()

                # criar previsões por ticker
                df_preds_list = []
                for ticker_bt in tickers_bt:
                    raw_bt = fetch_data(ticker_bt, f"{years}y")
                    quarterly_bt = preprocess_quarterly(raw_bt, ticker_bt)
                    features_bt = build_features(quarterly_bt).dropna(axis=1, how="all")
                    features_bt = features_bt.fillna(features_bt.median(numeric_only=True))

                    X_bt = features_bt.drop(columns=["dividend", "quarter"], errors="ignore")
                    y_bt = features_bt["dividend"]

                    # Treina modelo específico para o ticker
                    preds_bt, mae_bt, model_bt = train_model(X_bt, y_bt, model_type="ridge", n_splits=n_splits)
                    preds_bt = preds_bt.ffill().bfill()

                    df_pred = pd.DataFrame({
                        "quarter": features_bt["quarter"],
                        "ticker": ticker_bt,
                        "dividend_real": y_bt.values,
                        "dividend_pred": preds_bt
                    })
                    df_pred["quarter_dt"] = df_pred["quarter"].dt.to_timestamp()
                    
                    # Fazer uma cópia antes de adicionar à lista
                    df_preds_list.append(df_pred.copy())
                    
                # Concatenar todos os tickers
                df_all_preds = pd.concat(df_preds_list, ignore_index=True)
                df_all_preds = df_all_preds.sort_values(["ticker", "quarter_dt"]).reset_index(drop=True)

                
                # selecionar top-k por período
                df_topk = select_top_k(df_all_preds, k=top_k, by="dividend_pred")
                df_topk_filtered = df_topk[df_topk["ticker"].isin(tickers_bt)].copy()

                # simular estratégia (usa df_prices_wide) - USAR df_topk_filtered!
                strategy_returns = simulate_strategy(df_prices_wide, df_topk_filtered, hold_period=hold_period)
                strategy_returns = strategy_returns.dropna()

                # benchmark (IBOV)
                benchmark_prices = fetch_price_history(["^BVSP"], start=start_date, end=end_date)
                benchmark_prices["date"] = pd.to_datetime(benchmark_prices["date"])
                benchmark_prices = benchmark_prices.set_index("date").sort_index()
                benchmark_returns = benchmark_prices["adj_close"].pct_change().dropna()

                # métricas (CAGR, Sharpe e Max Drawdown)
                strategy_metrics_dict = compute_metrics(strategy_returns)
                benchmark_metrics_dict = compute_metrics(benchmark_returns)

                # Exibir resultados
                df_metrics = pd.DataFrame([strategy_metrics_dict, benchmark_metrics_dict],
                                        index=["Estratégia", "Benchmark"])

                display = df_metrics.copy()

                # formatar percentuais
                display["CAGR"] = (display["CAGR"] * 100).round(2)
                display["Max Drawdown"] = (display["Max Drawdown"] * 100).round(2)
                display["Sharpe"] = display["Sharpe"].round(3)

                st.subheader("Métricas da Estratégia (TOP 3 TIKERS) vs Benchmark (IBOV)")
                st.table(display)

                # --------------------
                # Gráfico de retorno acumulado
                # --------------------
                all_idx = strategy_returns.index.union(benchmark_returns.index).sort_values()
                cum_strategy = (1 + strategy_returns.reindex(all_idx).fillna(0)).cumprod()
                cum_benchmark = (1 + benchmark_returns.reindex(all_idx).fillna(0)).cumprod()

                cum_df = pd.DataFrame({
                    "Estratégia TOP 3 TIKERS": cum_strategy,
                    "Benchmark": cum_benchmark
                }, index=all_idx)

                st.subheader("Retorno Acumulado")
                st.line_chart(cum_df)

        #Explainability e Robustez
        st.markdown("---")
        st.header("🧠 Explainability & Robustez")
        st.markdown("Explicações de previsões e testes de estabilidade do modelo.")

        with st.expander("🔎 Resultado Explainability & Robustez", expanded=False):
            try:
                # df_all_preds já tem todas as colunas necessárias do loop anterior
                # Apenas renomear se necessário para padronizar
                if "dividend_real" in df_all_preds.columns:
                    df_all_preds = df_all_preds.rename(columns={"dividend_real": "dividend"})
                
                # Selecionar colunas relevantes
                df_display = df_all_preds[["quarter", "ticker", "dividend", "dividend_pred"]].copy()
                
            except Exception as e:
                st.warning(f"Não foi possível gerar visualização: {e}")

            # --- Importância global
            st.subheader("Importância Global das Features")
            try:
                X_proc = pd.get_dummies(X, drop_first=True).fillna(0)
                imputer = SimpleImputer(strategy="median")
                X_array = imputer.fit_transform(X_proc)
                X_proc = pd.DataFrame(X_array, columns=X_proc.columns, index=X_proc.index)

                importance = compute_feature_importance(model, X_proc.columns)
                fig_importance = plot_feature_importance(importance)
                st.pyplot(fig_importance, bbox_inches="tight")
            except Exception as e:
                st.warning(f"Não foi possível calcular importância global: {e}")

            # --- SHAP
            st.subheader("SHAP (Global)")
            try:
                shap_values, X_sample = shap_explain(model, X_proc)
                fig_summary = plot_shap_summary(shap_values, X_sample)
                st.pyplot(fig_summary, bbox_inches="tight")
            except Exception as e:
                st.warning(f"Não foi possível gerar SHAP summary: {e}")

            st.subheader("SHAP (Explicação Local - 1ª previsão)")
            try:
                fig_local = plot_shap_local(shap_values, index=0)
                st.pyplot(fig_local, bbox_inches="tight")
            except Exception as e:
                st.warning(f"Não foi possível gerar SHAP local: {e}")

            # Avaliação de robustez baseada em períodos (ano)
            st.markdown('---')
            try:
                df_all_preds["quarter_dt"] = pd.to_datetime(df_all_preds["quarter"].dt.to_timestamp())
                df_all_preds["year"] = df_all_preds["quarter_dt"].dt.year
                df_all_preds["date"] = df_all_preds["quarter_dt"]

                crisis_df = eval_robustness(df_all_preds, group_col="year")
                st.markdown("### Performance em períodos de crise")
                st.dataframe(crisis_df, use_container_width=True)
            except Exception as e:
                st.warning(f"Não foi possível calcular robustez: {e}")
            st.markdown("---")
    
    # --------------------
    # Aba 4: Glossário & Dicas
    # --------------------

    with tab4:
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
        elif ticker in ["AAPL", "MSFT", "KO"]:
            st.markdown("Empresas americanas de tecnologia tendem a reinvestir lucros em crescimento, pagando menos dividendos (exceto Coca-Cola).")
        else:
            st.markdown("Analise o histórico de dividendos, compare com empresas do mesmo setor e avalie se o perfil de risco combina com você.")
    
    
        # --------------------
        # Aba 5: Previsões Futuras
        # --------------------

        with tab5:
            st.subheader("🔮 Previsões Futuras de Dividendos")
            st.markdown("Visualização de previsões futuras usando o modelo treinado.")

            try:
                # --- Preparar últimas features do ticker ---
                last_features = features.drop(columns=["dividend", "quarter"], errors="ignore").iloc[[-1]]
                last_date = quarterly["quarter"].iloc[-1].to_timestamp()  # converte Period -> Timestamp
                last_features.index = [last_date]  # transforma em DatetimeIndex

                # --- Gerar previsões futuras ---
                forecast_df = forecast_future(model, last_features, n_periods=10)
                
                # Combina com histórico
                historical_df = df_pred.rename(columns={"dividend_real": "dividend"})
                historical_df.index = pd.to_datetime(quarterly["quarter"].dt.to_timestamp())

                combined_df = pd.concat([historical_df[["dividend"]], forecast_df], axis=0)

                f_raw = forecast_df.copy()
                f_iso = postproc_isotonic_calibrate(df_pred, f_raw)
            
                fig = plot_future_dividends(historical_df, f_iso)
                st.pyplot(fig, bbox_inches="tight")

            except Exception as e:
                st.warning(f"Não foi possível gerar previsões futuras: {e}")
    
else:
    st.info("Selecione um ticker na barra lateral e clique em **Analisar**.")
