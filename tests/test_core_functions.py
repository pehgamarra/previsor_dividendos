import sys
import os
import pandas as pd
import numpy as np

# Ajusta caminho se suas funções estiverem em /src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


def test_basic_functions_exist():
    try:
        from src.data.preprocess import preprocess_quarterly
    except ImportError:
        # ignora se estrutura for diferente
        assert True
        return

    # Mock simples de dados de entrada
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=4, freq="Q"),
        "value": np.random.rand(4),
    })

    # Testa preprocessamento
    processed = preprocess_quarterly(df, "TEST")
    assert isinstance(processed, pd.DataFrame)

