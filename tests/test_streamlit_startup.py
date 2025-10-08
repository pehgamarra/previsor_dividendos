import importlib

def test_streamlit_starts():
    # Verifica se o arquivo principal existe e pode ser importado
    try:
        importlib.import_module("app.streamlit_app")
    except Exception as e:
        raise AssertionError(f"Falha ao importar Streamlit app: {e}")
