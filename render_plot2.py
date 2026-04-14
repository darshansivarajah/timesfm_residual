import torch
import timesfm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean

from statsmodels.tsa.arima.model import ARIMA

# ================================
# LOAD TIMESFM
# ================================
torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

model.compile(
    timesfm.ForecastConfig(
        max_context=1024,                   # max input history length
        max_horizon=256,                    # max forecasting steps
        normalize_inputs=True,              # scale inputs internally
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

# ================================
# METRICS
# ================================
def mae(y_true, y_pred):                    # Mean Absolute Error
    return np.mean(np.abs(y_true - y_pred))

def scaled_mae(y_true, y_pred, context):    # Scaled MAE compares model error to a naive baseline
    naive = np.repeat(context[-1], len(y_true))
    return mae(y_true, y_pred) / mae(y_true, naive)

# ================================
# FORECAST METHODS
# ================================
def timesfm_forecast(context, horizon):
    context = np.array(context, dtype=float)
    context = np.nan_to_num(context)        # Replaces NAs with 0s

    pred = model.forecast(inputs=[context], horizon=horizon)
    pred = pred[0][0]  # extract mean forecast

    return np.array(pred)

def naive_forecast(context, horizon):       # fits baseline model
    return np.repeat(context[-1], horizon)

def arima_forecast(context, horizon):       # fits a simple ARIMA model
    try:
        model_arima = ARIMA(context, order=(1,1,1))
        fitted = model_arima.fit()
        return fitted.forecast(steps=horizon)
    except:
        return naive_forecast(context, horizon)

# ================================
# EVALUATION CORE
# ================================
def evaluate_series(series, context_len, horizon):
    # splits the series into contexts and ground truths
    context = series[-(context_len + horizon):-horizon]
    true = series[-horizon:]

    results = {}

    pred_tf = timesfm_forecast(context, horizon)
    results["TimesFM"] = scaled_mae(true, pred_tf, context)

    pred_naive = naive_forecast(context, horizon)
    results["Naive"] = scaled_mae(true, pred_naive, context)

    pred_arima = arima_forecast(context, horizon)
    results["ARIMA"] = scaled_mae(true, pred_arima, context)

    return results

# ================================
# DATASETS
# ================================

# ---- ETT ----
def load_ett():
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    df = pd.read_csv(url)
    return df["OT"].values.astype(float)

# ---- MONASH (proxy multi-series) ----
def load_monash_like():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    df = pd.read_csv(url)
    base = df["Passengers"].values.astype(float)

    # simulate multiple datasets
    datasets = []
    for i in range(5):
        datasets.append(base[i*50:(i+1)*50])
    return datasets

# ---- DARTS (proxy single-series set) ----
def load_darts_like():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv"
    df = pd.read_csv(url)
    return [df["Sunspots"].values.astype(float)]

# ================================
# AGGREGATION
# ================================
def aggregate_results(results_list):
    models = results_list[0].keys()
    agg = {}

    for m in models:
        vals = [r[m] for r in results_list if m in r]
        agg[m] = gmean(vals)

    return agg

# ================================
# PLOTTING
# ================================
def plot_results(results, title):
    models = list(results.keys())
    values = list(results.values())

    plt.figure()
    plt.bar(models, values)

    plt.axhline(1.0, linestyle='--')  # naive baseline
    plt.ylim(0, 1.2)

    plt.ylabel("Scaled MAE (lower is better)")
    plt.title(title)

    plt.show()

# ================================
# RUN EXPERIMENTS
# ================================
def run_all():

    # ---- MONASH ----
    monash_data = load_monash_like()
    monash_results = []

    for s in monash_data:
        if len(s) < 40:
            continue
        res = evaluate_series(s, 24, 12)
        monash_results.append(res)

    monash_agg = aggregate_results(monash_results)
    plot_results(monash_agg, "Monash (Figure 2a Approx)")

    # ---- DARTS ----
    darts_data = load_darts_like()
    darts_results = []

    for s in darts_data:
        res = evaluate_series(s, 100, 12)
        darts_results.append(res)

    darts_agg = aggregate_results(darts_results)
    plot_results(darts_agg, "Darts (Figure 2b Approx)")

    # ---- ETT ----
    ett_series = load_ett()
    ett_results = [evaluate_series(ett_series, 512, 96)]
    ett_agg = aggregate_results(ett_results)
    plot_results(ett_agg, "ETT (Figure 2c Approx)")

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    run_all()