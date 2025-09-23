import sys
import os
import streamlit as st
import pandas as pd
import yfinance as yf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.fetch import fetch_data
from src.data.preprocess import preprocess_quarterly
from src.features.engineering import build_features
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
if run:
    with st.spinner("Coletando dados..."):
        raw = fetch_data(ticker, period)
        quarterly = preprocess_quarterly(raw, ticker)
        features = build_features(quarterly)
        results = evaluate_baselines(quarterly, n_splits=5)
        # puxar informações da empresa
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
    st.success(f" ### {ticker} - {available_tickers[ticker]}")

    # --------------------
    # Abas
    # --------------------
    tab1, tab2, tab3 = st.tabs(["📈 Dividendos", "🏢 Empresa", "📖 Glossário & Dicas"])

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

        # --------------------
        # Glossário didático
        # --------------------
        st.markdown("---")
        st.markdown("### 📚 Glossário"
                      " (para iniciantes)")
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
           
else:
    st.info("Selecione um ticker na barra lateral e clique em **Analisar**.")
 