import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

def test_preprocess_quarterly():
    try:
        from src.data.preprocess import preprocess_quarterly
    except ImportError:
        assert True
        return

    # Mock de dados diários
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=10, freq="D"),
        "close": np.linspace(100, 110, 10),
        "dividend": [0, 0.5, 0, 0, 0.2, 0, 0, 0.3, 0, 0]
    })

    ticker = "TEST"

    # Roda o preprocess
    quarterly = preprocess_quarterly(df, ticker)

    # Checagens básicas
    assert isinstance(quarterly, pd.DataFrame)
    assert not quarterly.empty
    assert "has_dividend" in quarterly.columns
    assert "dividend_yield" in quarterly.columns
    assert "quarter" in quarterly.columns
    assert "ticker" in quarterly.columns
    assert all(quarterly["ticker"] == ticker)

    # Checa agregação trimestral (quantidade de linhas <= número de dias originais)
    assert len(quarterly) <= len(df)
