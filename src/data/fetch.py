import yfinance as yf
import pandas as pd
def fetch_data(ticker: str, period="max") -> pd.DataFrame:
    """
    Coleta preços e dividendos AJUSTADOS de um ticker usando yfinance.
    CRÍTICO: Usa auto_adjust=True para corrigir splits.
    """
    stock = yf.Ticker(ticker)
    
    # Baixar dados com ajuste automático
    hist = stock.history(period=period, auto_adjust=True)
    
    # CRÍTICO: Remover timezone para evitar conflitos
    hist.index = hist.index.tz_localize(None)
    
    # Extrair preços e dividendos
    prices = hist[["Close"]].reset_index()
    prices.columns = ["date", "close"]
    
    dividends = hist[["Dividends"]].reset_index()
    dividends.columns = ["date", "dividend"]
    
    # Merge
    data = pd.merge(prices, dividends, on="date", how="left")
    data["dividend"] = data["dividend"].fillna(0.0)
    
    # Remover linhas com preço zero/NaN
    data = data[data["close"] > 0].reset_index(drop=True)
    
    print(f"\n✅ Dados coletados para {ticker}:")
    print(f"   Período: {data['date'].min()} → {data['date'].max()}")
    print(f"   Total de dias: {len(data)}")
    print(f"   Dividendos pagos: {(data['dividend'] > 0).sum()}")
    print(f"   Último preço: R$ {data['close'].iloc[-1]:.2f}")
    print(f"   Último dividendo: R$ {data[data['dividend'] > 0]['dividend'].iloc[-1] if (data['dividend'] > 0).any() else 0:.4f}")
    
    return data


def fetch_price_history(tickers, start, end):
    """
    Baixa preços ajustados de um ou mais tickers.
    Usa auto_adjust=True para consistência.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    price_dfs = []

    for ticker in tickers:
        # Baixar com ajuste automático
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, auto_adjust=True)

        if df.empty:
            print(f"⚠️ Aviso: Nenhum dado para {ticker}")
            continue

        # CRÍTICO: Remover timezone
        df.index = df.index.tz_localize(None)

        # Com auto_adjust=True, a coluna é sempre 'Close' (já ajustada)
        if 'Close' in df.columns:
            series = df['Close']
        else:
            print(f"⚠️ Aviso: '{ticker}' não tem coluna 'Close'. Ignorando.")
            continue

        series.name = ticker
        price_dfs.append(series)

    if not price_dfs:
        raise ValueError("Nenhum dado válido retornado para os tickers informados.")

    # Combinar todos os tickers
    df_prices = pd.concat(price_dfs, axis=1)
    df_prices = df_prices.sort_index()
    
    # Transformar para formato longo
    df_prices_long = df_prices.reset_index().melt(
        id_vars="Date",
        var_name="ticker",
        value_name="adj_close"
    ).rename(columns={"Date": "date"})
    
    # Garantir que 'date' também está sem timezone
    df_prices_long['date'] = pd.to_datetime(df_prices_long['date']).dt.tz_localize(None)
    
    return df_prices_long