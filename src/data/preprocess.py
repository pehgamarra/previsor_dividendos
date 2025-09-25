import pandas as pd

def preprocess_quarterly(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Transforma dados diários em dataset trimestral com alvos.
    """
    data = data.copy()
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
    return quarterly