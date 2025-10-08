# Previsor de Dividendos

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-blue)](https://previsordividendos.streamlit.app/)
[![CI](https://github.com/pehgamarra/previsor_dividendos/actions/workflows/ci.yml/badge.svg)](https://github.com/pehgamarra/previsor_dividendos/actions/workflows/ci.yml)

## Descrição
O **Previsor de Dividendos** é um sistema de previsão de dividendos de ações utilizando dados históricos do Yahoo Finance.  
O projeto permite analisar a evolução dos dividendos de diferentes ações e gerar previsões trimestrais de forma automatizada.

---

## Tecnologias Utilizadas
- **Python 3.13**
- **Pandas / Numpy** – manipulação de dados
- **YahooFinance (yfinance)** – obtenção de dados históricos
- **Matplotlib** – visualização de dados
- **Streamlit** – interface web interativa
- **Docker** – reprodutibilidade e deploy
- **Git / GitHub Actions** – versionamento e CI/CD

---

## Estrutura do Projeto

```bash
previsor_dividendos/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── feature/
│   └── models/
├── tests/
├── .github/
│   └── workflows/
├── Dockerfile
├── requirements.txt
└── .gitignore
```

---

## Instalação e Execução Local

### Requisitos
- Python 3.13
- pip

### Instalação
```bash
git clone https://github.com/pehgamarra/previsor_dividendos.git
cd previsor de dividendos
pip install -r requirements.txt

### Executar localmente
streamlit run app/streamlit_app.py

### Executar com Docker
docker build -t previsor de dividendos .
docker run -p 8501:8501 previsor de dividendos
```
---

## Deploy
O projeto já está deployado no **Streamlit Cloud**:  
[https://previsordividendos.streamlit.app/](https://previsordividendos.streamlit.app/)

## CI/CD
O repositório possui **GitHub Actions** configurado para:
- Rodar testes unitários (`pytest`)
- Checar estilo de código (`flake8`)
- Garantir que o app importa e inicializa sem erros

O workflow está ativo e o badge de CI mostra o status atual.

## Exemplo de Uso
1. O usuário seleciona um ticker e o período desejado.
2. O app retorna:
   - Evolução histórica de dividendos
   - Previsão trimestral futura
   - Gráficos de distribuição de dividendos


## Autor
**Pedro Gamarra** – [GitHub](https://github.com/pehgamarra) / [Linkedin](https://www.linkedin.com/in/pedro-gamarra-9b8162181/)
