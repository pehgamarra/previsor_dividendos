import yfinance as yf
import pandas as pd

def fetch_data(ticker: str, period: str = "15y") -> pd.DataFrame:
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
    Retorna DataFrame com preços ajustados históricos.
    
    Args:
        tickers (list): lista de tickers
        start (str): data inicial (YYYY-MM-DD)
        end (str): data final (YYYY-MM-DD)
    
    Returns:
        pd.DataFrame: ['date', 'ticker', 'adj_close']
    """
    all_prices = []
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, progress=False)[["Adj Close"]]
        df = df.reset_index()
        df["ticker"] = ticker
        df.rename(columns={"Adj Close": "adj_close"}, inplace=True)
        all_prices.append(df)
    return pd.concat(all_prices, ignore_index=True)

