import importlib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_streamlit_starts():
    try:
        importlib.import_module("app.streamlit_app")
    except Exception as e:
        raise AssertionError(f"Falha ao importar Streamlit app: {e}")
