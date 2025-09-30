import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Baselines simples para séries temporais de dividendos

#----- Baselines -----
def baseline_last(df):
    """Previsão = último dividendo observado (t-1)."""
    return df["dividend"].shift(1)

# Média móvel dos últimos 4 pagamentos
def baseline_mean4(df):
    """Previsão = média dos últimos 4 pagamentos (shiftada para frente)."""
    return df["dividend"].rolling(4).mean().shift(1)

# Regra simples baseada em payout (se existir EPS)
def baseline_rule_based(df):
    """
    Exemplo simples: usar payout médio (se existir EPS).
    Retorna série de previsões (alinhar com index do df).
    """
    if "eps" in df.columns and df["eps"].dropna().shape[0] > 0:
        payout = (df["dividend"] / df["eps"]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(payout) == 0:
            return baseline_last(df)
        payout_avg = payout.mean()
        return df["eps"].shift(1) * payout_avg
    else:
        return baseline_last(df)

# Funções de métrica
def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Funções de avaliação
def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Função para calcular MAPE, evitando divisão por zero
def compute_mape(y_true, y_pred):
    # evita divisão por zero
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Função segura para avaliação
def safe_eval(y_true, y_pred):
    """
    Recebe duas Series (pandas) possivelmente com NaNs, alinha e calcula MAE/RMSE/MAPE.
    Retorna (mae, rmse, mape).
    """
    df = pd.concat([y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        return np.nan, np.nan, np.nan
    y_t = df["y_true"].to_numpy()
    y_p = df["y_pred"].to_numpy()
    mae = compute_mae(y_t, y_p)
    rmse = compute_rmse(y_t, y_p)
    mape = compute_mape(y_t, y_p)
    return mae, rmse, mape

# Função TimeSeriesSplit para avaliação temporal
def evaluate_baselines(df, n_splits=5):
    """
    Avalia os baselines com validação temporal (TimeSeriesSplit).
    Retorna DataFrame com MAE/RMSE/MAPE por split e médias.
    """
    df = df.copy()
    results = []
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Precompute baseline predictions on the whole df (shift/rolling funcionam corretamente)
    pred_last = baseline_last(df)
    pred_mean4 = baseline_mean4(df)
    pred_rule = baseline_rule_based(df)

    split_no = 0
    for train_idx, test_idx in tscv.split(df):
        split_no += 1
        test = df.iloc[test_idx]
        y_true = test["dividend"]

        # pego por posição (iloc) para evitar problemas de index não-ordenado
        y_pred_last = pred_last.iloc[test_idx]
        y_pred_mean4 = pred_mean4.iloc[test_idx]
        y_pred_rule = pred_rule.iloc[test_idx]

        mae_last, rmse_last, mape_last = safe_eval(y_true, y_pred_last)
        mae_mean4, rmse_mean4, mape_mean4 = safe_eval(y_true, y_pred_mean4)
        mae_rule, rmse_rule, mape_rule = safe_eval(y_true, y_pred_rule)

        results.append({
            "split": split_no,
            "MAE_last": mae_last, "RMSE_last": rmse_last, "MAPE_last(%)": mape_last,
            "MAE_mean4": mae_mean4, "RMSE_mean4": rmse_mean4, "MAPE_mean4(%)": mape_mean4,
            "MAE_rule": mae_rule, "RMSE_rule": rmse_rule, "MAPE_rule(%)": mape_rule,
        })

    df_res = pd.DataFrame(results)

    # adicionar resumo (médias) ao final
    means = df_res.mean(numeric_only=True).to_dict()
    summary = {"split": "mean"}
    summary.update({k: v for k, v in means.items()})
    df_res = pd.concat([df_res, pd.DataFrame([summary])], ignore_index=True)

    return df_res

