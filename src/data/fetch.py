import yfinance as yf
import pandas as pd

#Fetching data functions

# Função para coletar dados históricos de preços e dividendos
def fetch_data(ticker: str, period) -> pd.DataFrame:
    """
    Coleta preços e dividendos de um ticker usando yfinance.
    """
    stock = yf.Ticker(ticker)

    dividends = stock.dividends.reset_index()
    dividends.columns = ["date", "dividend"]

    prices = stock.history(period=period)[["Close"]].reset_index()
    prices.columns = ["date", "close"]

    data = pd.merge(prices, dividends, on="date", how="left")
    data["dividend"] = data["dividend"].fillna(0.0)

    return data

# Função para coletar preços ajustados históricos
def fetch_price_history(tickers, start, end):
    """
    Baixa preços ajustados de um ou mais tickers.
    Tenta 'Adj Close', se não existir, usa 'Close'.
    Retorna DataFrame com datas como índice e tickers como colunas.
    """

    if isinstance(tickers, str):
        tickers = [tickers]

    price_dfs = []

    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, progress=False)

        if df.empty:
            print(f"Aviso: Nenhum dado para {ticker}")
            continue

        # Tenta 'Adj Close' primeiro
        if 'Adj Close' in df.columns:
            series = df['Adj Close']
        elif 'Close' in df.columns:
            series = df['Close']
        else:
            print(f"Aviso: '{ticker}' não tem 'Adj Close' nem 'Close'. Ignorando.")
            continue

        series.name = ticker  # nome da coluna
        price_dfs.append(series)

    if not price_dfs:
        raise ValueError("Nenhum dado válido retornado para os tickers informados.")

    # Combina todos os tickers
    df_prices = pd.concat(price_dfs, axis=1)
    df_prices = df_prices.sort_index()
    
    # Transformar para formato longo
    df_prices_long = df_prices.reset_index().melt(
        id_vars="Date",
        var_name="ticker",
        value_name="adj_close"
    ).rename(columns={"Date": "date"})
    
    return df_prices_long

