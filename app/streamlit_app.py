import sys
import os
import streamlit as st
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.fetch import fetch_data
from src.data.preprocess import preprocess_quarterly

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

# Lista de tickers sugeridos (pode expandir)
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

# Selectbox com busca
ticker = st.sidebar.selectbox(
    "Escolha um ticker:",
    options=list(available_tickers.keys()),
    format_func=lambda x: f"{x} - {available_tickers[x]}"
)

# Período de histórico
period = st.sidebar.radio(
    "Período do histórico em anos:",
    options=["5y", "10y", "15y", "20y", "max"],
    index=1,
    horizontal=True
)

# Botão para rodar
run = st.sidebar.button("🔎 Analisar")

# --------------------
# Execução
# --------------------
if run:
    with st.spinner("Coletando dados..."):
        raw = fetch_data(ticker, period)
        quarterly = preprocess_quarterly(raw, ticker)

    st.success(f"Dados de **{ticker} - {available_tickers[ticker]}** carregados!")

    # --------------------
    # Layout principal
    # --------------------
    col1, col2 = st.columns([2, 1])

    # Coluna 1: gráfico
    with col1:
        st.subheader("📈 Dividendos ao longo do tempo")
        st.line_chart(
            quarterly.set_index("quarter")[["dividend", "close"]],
            use_container_width=True
        )

    # Coluna 2: métricas
    with col2:
        st.subheader("🔢 Estatísticas")
        total = quarterly["dividend"].sum()
        avg_yield = quarterly["dividend_yield"].mean() * 100
        freq = quarterly["has_dividend"].mean() * 100

        st.metric("Dividendos pagos (total)", f"{total:.2f}")
        st.metric("Yield médio", f"{avg_yield:.2f}%")
        st.metric("Frequência de pagamentos", f"{freq:.1f}% dos trimestres")

    # --------------------
    # Tabela de dividendos
    # --------------------
    st.subheader("📑 Histórico Trimestral")
    df_display = quarterly.copy()
    df_display["quarter"] = df_display["quarter"].astype(str)
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Download CSV
    st.download_button(
        label="💾 Baixar CSV",
        data=quarterly.to_csv(index=False),
        file_name=f"{ticker}_dividends.csv",
        mime="text/csv"
    )

else:
    st.info("Selecione um ticker na barra lateral e clique em **Analisar**.")
