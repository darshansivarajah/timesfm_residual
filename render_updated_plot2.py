#uv run render_updated_plot2.py

import torch
import timesfm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean
from statsmodels.tsa.arima.model import ARIMA

# ✅ IMPORT YOUR MODEL
from innovative_idea_model import ResidualForecaster

# ================================
# LOAD TIMESFM
# ================================
torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

model.compile(
    timesfm.ForecastConfig(
        max_context=1024,       # max input sequence length
        max_horizon=256,        # max forecast steps
        normalize_inputs=True,  # normalize time series internally
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

# ================================
# METRICS
# ================================
def mae(y_true, y_pred):                            # calculates Mean Absolute Error
    return np.mean(np.abs(y_true - y_pred))

def scaled_mae(y_true, y_pred, context):            # scaled MAE (compares model vs naive baseline)
    naive = np.repeat(context[-1], len(y_true))
    return mae(y_true, y_pred) / mae(y_true, naive) # ratio: model error / naive error

# ================================
# FORECAST METHODS
# ================================
def timesfm_forecast(context, horizon):
    context = np.array(context, dtype=float)
    context = np.nan_to_num(context)        # if there are NaNs in context, we replace them with 0

    pred = model.forecast(inputs=[context], horizon=horizon)
    pred = np.array(pred[0]).reshape(-1)    # model output is flattened so that the shape is (horizon, )

    return pred

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
# TRAIN RESIDUAL MODEL
# ================================
def train_residual_model(series_list, context_len, horizon):
    contexts, y_trues = [], []

    for series in series_list:                  # raw time series is converted into contexts and true y values
        if len(series) < context_len + horizon: # if the series is short, we skip it
            continue

        context = series[-(context_len + horizon):-horizon]
        future = series[-horizon:]

        contexts.append(context)
        y_trues.append(future)

    model_res = ResidualForecaster()            # residual model is initialized
    model_res.fit(contexts, y_trues, horizon)   # residual model is fitted; it learns that residual = true - TimesFM_prediction

    return model_res

# ================================
# EVALUATION CORE
# ================================
def evaluate_series(series, context_len, horizon, residual_model=None):
    # splits the series into contexts and ground truths
    context = series[-(context_len + horizon):-horizon]
    true = series[-horizon:]

    results = {}

    # TimesFM
    pred_tf = timesfm_forecast(context, horizon)
    results["TimesFM"] = scaled_mae(true, pred_tf, context)

    # TimesFM enhanced using residual layer
    if residual_model is not None:
        pred_res = residual_model.predict(context, horizon)
        results["TimesFM2"] = scaled_mae(true, pred_res, context)

    # Naive
    pred_naive = naive_forecast(context, horizon)
    results["Naive"] = scaled_mae(true, pred_naive, context)

    # ARIMA
    pred_arima = arima_forecast(context, horizon)
    results["ARIMA"] = scaled_mae(true, pred_arima, context)

    return results

# ================================
# DATASETS
# ================================
def load_ett():
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    df = pd.read_csv(url)
    return df["OT"].values.astype(float)

def load_monash_like():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    df = pd.read_csv(url)
    base = df["Passengers"].values.astype(float)

    datasets = []
    for i in range(5):
        datasets.append(base[i*50:(i+1)*50])
    return datasets

def load_darts_like():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv"
    df = pd.read_csv(url)
    return [df["Sunspots"].values.astype(float)]

# ================================
# AGGREGATION
# ================================
def aggregate_results(results_list):
    # combine results across multiple series using geometric mean
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
    # creates bar plots
    models = list(results.keys())
    values = list(results.values())

    plt.figure()
    plt.bar(models, values)

    plt.axhline(1.0, linestyle='--')
    plt.ylim(0, 1.2)

    plt.ylabel("Scaled MAE (lower is better)")
    plt.title(title)

    plt.xticks(rotation=20)
    plt.show()

# ================================
# RUN EXPERIMENTS
# ================================
def run_all():

    # ---- MONASH ----
    print("Running Monash...")
    monash_data = load_monash_like()

    res_model_monash = train_residual_model(monash_data, 24, 12)

    monash_results = []
    for s in monash_data:
        if len(s) < 40:
            continue
        monash_results.append(evaluate_series(s, 24, 12, res_model_monash))

    plot_results(aggregate_results(monash_results), "Monash (Figure 2a Approx)")

    # ---- DARTS ----
    print("Running Darts...")
    darts_data = load_darts_like()

    res_model_darts = train_residual_model(darts_data, 100, 12)

    darts_results = []
    for s in darts_data:
        darts_results.append(evaluate_series(s, 100, 12, res_model_darts))

    plot_results(aggregate_results(darts_results), "Darts (Figure 2b Approx)")

    # ---- ETT ----
    print("Running ETT...")
    ett_series = load_ett()

    res_model_ett = train_residual_model([ett_series], 512, 96)

    ett_results = [evaluate_series(ett_series, 512, 96, res_model_ett)]

    plot_results(aggregate_results(ett_results), "ETT (Figure 2c Approx)")

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    run_all()