import pandas as pd
from sklearn.metrics import mean_absolute_error

#Este arquivo contém funções para avaliar a robustez do modelo

#Erro por períodos de crise
def eval_crisis_periods(df, y_col="dividend", y_pred_col="dividend_pred"):
    """Avalia períodos de crise fixos"""
    crisis_ranges = [
        ("2008-01-01", "2009-12-31", "GFC"),
        ("2020-03-01", "2020-12-31", "Covid")
    ]

    results = []
    for start, end, label in crisis_ranges:
        sub = df[(df["date"] >= start) & (df["date"] <= end)]
        if sub.empty:
            continue
        mae = mean_absolute_error(sub[y_col], sub[y_pred_col])
        results.append({"period": label, "MAE": mae, "n": len(sub)})
    return pd.DataFrame(results)

def eval_robustness(df, group_col, y_col="dividend", y_pred_col="dividend_pred"):
    """Avalia robustez do modelo em diferentes aspectos"""
    crisis = eval_crisis_periods(df, y_col, y_pred_col)
    return  crisis