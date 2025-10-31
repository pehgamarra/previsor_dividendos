import pandas as pd

# Data preprocessing functions
def preprocess_quarterly(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Transforma dados diários em dataset trimestral com alvos.
    Remove trimestres sem dividendos ou ainda não finalizados.
    """
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["has_dividend"] = (data["dividend"] > 0).astype(int)
    data["dividend_yield"] = data["dividend"] / data["close"]
    data["quarter"] = data["date"].dt.to_period("Q")

    quarterly = data.groupby("quarter").agg({
        "close": "last",
        "dividend": "sum",
        "has_dividend": "max"
    }).reset_index()

    quarterly["dividend_yield"] = quarterly["dividend"] / quarterly["close"]
    quarterly["ticker"] = ticker

    current_quarter = pd.Timestamp.now().to_period("Q")
    quarterly = quarterly[
        quarterly["dividend"].notna() &
        (quarterly["dividend"] > 0) &
        (quarterly["quarter"] < current_quarter)
    ]

    return quarterly
