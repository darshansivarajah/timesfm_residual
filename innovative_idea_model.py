#uv run innovative_idea_model.py
import numpy as np
import torch
import timesfm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Load pretrained TimesFM model
# -------------------------------
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

model.compile(
    timesfm.ForecastConfig(
        max_context=1024,       # "look back" at up to 1024 historical points
        max_horizon=256,        # predict up to 256 points into the future
        normalize_inputs=True,  # normalize time series internally
    )
)

# -------------------------------
# 2. Helper: Generate time features
# -------------------------------
def create_time_features(horizon):
    horizon = int(horizon)
    t = np.arange(horizon)

    # cyclical encoding transforms time into sine/cosine waves

    return np.stack([
        t,                                   # linear time index
        np.sin(2 * np.pi * t / 24),          # daily seasonality (Sine)
        np.cos(2 * np.pi * t / 24),          # daily seasonality (Cosine)
        np.sin(2 * np.pi * t / 7),           # weekly seasonality (Sine)
        np.cos(2 * np.pi * t / 7),           # weekly seasonality (Cosine)
    ], axis=1)


# -------------------------------
# 3. Run TimesFM prediction
# -------------------------------
def timesfm_predict(context, horizon):
    context = np.asarray(context, dtype=float)
    context = np.nan_to_num(context)

    out = model.forecast(inputs=[context], horizon=horizon) # generate the zero-shot forecast

    if isinstance(out, dict):
        pred = out["mean"]
    else:
        pred = out[0]

    pred = np.array(pred)

    # flatten the dimensions (TimesFM often returns 3D arrays: batch, horizon, sample)
    if pred.ndim == 3:        # (1, H, S)
        pred = pred[0].mean(axis=-1)
    elif pred.ndim == 2:      # (1, H)
        pred = pred[0]

    return pred.reshape(-1)   # return a clean 1D array of length 'horizon'



# -------------------------------
# 4. Train Residual Model
# -------------------------------
class ResidualForecaster:
    def __init__(self):
        self.model = GradientBoostingRegressor()    # Gradient Boosting finds non-linear patterns in small datasets
        self.scaler = StandardScaler()

    def fit(self, contexts, y_trues, horizon):
        X_all = []
        y_all = []

        for context, y_true in zip(contexts, y_trues):
            context = np.asarray(context)
            y_true = np.asarray(y_true)

            base_pred = timesfm_predict(context, horizon)
            residual = y_true - base_pred

            time_feats = create_time_features(horizon)  # get time features for this specific window

            if len(residual) != horizon:
                raise ValueError(f"Residual shape mismatch: {residual.shape}")
            
            for t in range(horizon):
                X_all.append(time_feats[t])
                y_all.append(residual[t])

        # normalize features and train the Booster to predict the "error"
        X_all = np.array(X_all)
        y_all = np.array(y_all)

        X_all = self.scaler.fit_transform(X_all)
        self.model.fit(X_all, y_all)

    def predict(self, context, horizon):
        context = np.asarray(context)
        base_pred = timesfm_predict(context, horizon)

        # Gradient Boosting to predict the 'correction factor' for this time
        time_feats = create_time_features(horizon)
        time_feats = self.scaler.transform(time_feats)
        residual_pred = self.model.predict(time_feats)

        final_pred = base_pred + residual_pred      # combine base prediction + correction = final prediction
        return final_pred


# -------------------------------
# 5. Example Usage
# -------------------------------
if __name__ == "__main__":
    # fake data (replace with real dataset)
    np.random.seed(0)

    contexts = []
    y_trues = []

    for _ in range(50):
        series = np.sin(np.linspace(0, 20, 200)) + np.random.normal(0, 0.1, 200)

        context = series[:150].astype(np.float32)
        future = series[150:170].astype(np.float32)

        contexts.append(context)
        y_trues.append(future)

        horizon = 20

    # train residual model
    residual_model = ResidualForecaster()
    residual_model.fit(contexts, y_trues, horizon)

    # test prediction
    test_context = contexts[0]
    pred = residual_model.predict(test_context, horizon)

    print("Final Prediction:", pred)