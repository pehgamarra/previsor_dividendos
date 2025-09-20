import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.process import preprocess_quarterly
from src.data.fetch import fetch_data
import streamlit as st

st.title("📊 Previsor de Dividendos")

# Input do usuário
ticker = st.text_input("Digite o ticker (ex.: PETR4.SA, AAPL):", "AAPL")
period = st.selectbox("Período de histórico:", ["5y", "10y", "15y", "20y"], index=1)

if st.button("Carregar dados"):
    with st.spinner("Coletando dados..."):
        raw = fetch_data(ticker, period)
        quarterly = preprocess_quarterly(raw, ticker)

    st.success("Dados carregados!")
    st.dataframe(quarterly)

    # Download opcional
    st.download_button(
        label="Baixar CSV",
        data=quarterly.to_csv(index=False),
        file_name=f"{ticker}_dividends.csv",
        mime="text/csv"
    )
