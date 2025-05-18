from data.preprocess import build_model_input
from data.model_input import generate_model_input
from models import xgboost_model, lstm_model, patchtst_model, nbeats_model
from sklearn.metrics import mean_squared_error
from ensemble import train_and_predict_ensemble
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# ========== PARAMETERS ==========
ticker = "MSFT"
timeframes = ["1d"]
start = "2022-01-01"
end = "2023-06-01"
target_col = f"{timeframes[0]}_close"
future_steps = 5

# ========== LOAD DATA ==========
df = build_model_input(ticker, timeframes, start, end)

# ========== GENERATE INPUT ==========
X_train, y_train, X_test, y_test = generate_model_input(df, target_col)

print(f"\n✅ Model Input Shapes:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"y_test : {y_test.shape}")

# ========== XGBOOST ==========
print("\n🌲 Training XGBoost...")
model_xgb = xgboost_model.build_model()
X_train_xgb, y_train_xgb, X_test_xgb = xgboost_model.prepare_data(X_train, y_train, X_test)
y_pred_xgb = xgboost_model.train_and_predict(X_train_xgb, y_train_xgb, X_test_xgb, model_xgb)
y_pred_xgb = np.array(y_pred_xgb).flatten()[:len(y_test)]
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f"📊 XGBoost RMSE: {rmse_xgb:.4f}")
future_xgb = xgboost_model.predict_future_sequence(model_xgb, X_test_xgb.iloc[-1], steps=future_steps)
print(f"🔮 XGBoost Future {future_steps} Predictions: {future_xgb}")

# ========== LSTM ==========
print("\n🧠 Training LSTM...")
X_train_lstm, y_train_lstm, X_test_lstm = lstm_model.prepare_data(X_train, y_train, X_test)
model_lstm = lstm_model.build_model(X_train_lstm.shape[1:])
y_pred_lstm = lstm_model.train_and_predict(X_train_lstm, y_train_lstm, X_test_lstm, model_lstm)
y_pred_lstm = np.array(y_pred_lstm).flatten()[:len(y_test)]
rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
print(f"📊 LSTM RMSE: {rmse_lstm:.4f}")
future_lstm = lstm_model.predict_future_sequence(model_lstm, X_test_lstm[-1], steps=future_steps)
print(f"🔮 LSTM Future {future_steps} Predictions: {future_lstm}")

# ========== PATCHTST ==========
print("\n📈 Training PatchTST...")
X_train_patchtst, y_train_patchtst, X_test_patchtst = patchtst_model.prepare_data(X_train, y_train, X_test)
model_patchtst = patchtst_model.build_model()
y_pred_patchtst = patchtst_model.train_and_predict(X_train_patchtst, y_train_patchtst, X_test_patchtst, model_patchtst)
y_pred_patchtst = np.array(y_pred_patchtst).flatten()[:len(y_test)]
rmse_patchtst = np.sqrt(mean_squared_error(y_test, y_pred_patchtst))
print(f"📊 PatchTST RMSE: {rmse_patchtst:.4f}")
future_patchtst = patchtst_model.predict_future_sequence(model_patchtst, X_test_patchtst[-1], steps=future_steps)
print(f"🔮 PatchTST Future {future_steps} Predictions: {future_patchtst}")

# ========== N-BEATS ==========
print("\n⏳ Training N-BEATS...")
X_train_nbeats, y_train_nbeats, X_test_nbeats = nbeats_model.prepare_data(X_train, y_train, X_test)
model_nbeats = nbeats_model.build_model()
y_pred_nbeats = nbeats_model.train_and_predict(X_train_nbeats, y_train_nbeats, X_test_nbeats, model_nbeats)
y_pred_nbeats = np.array(y_pred_nbeats).flatten()[:len(y_test)]
rmse_nbeats = np.sqrt(mean_squared_error(y_test, y_pred_nbeats))
print(f"📊 N-BEATS RMSE: {rmse_nbeats:.4f}")
future_nbeats = nbeats_model.predict_future_sequence(model_nbeats, X_test_nbeats[-1], steps=future_steps)
print(f"🔮 N-BEATS Future {future_steps} Predictions: {future_nbeats}")

# ========== ENSEMBLE ==========
print("\n🧩 Running Ensemble Model...")

data_per_model = {
    "XGBoost": {
        "X_train": X_train_xgb, "y_train": y_train_xgb, "X_test": X_test_xgb
    },
    "LSTM": {
        "X_train": X_train_lstm, "y_train": y_train_lstm, "X_test": X_test_lstm
    },
    "PatchTST": {
        "X_train": X_train_patchtst, "y_train": y_train_patchtst, "X_test": X_test_patchtst
    },
    "NBEATS": {
        "X_train": X_train_nbeats, "y_train": y_train_nbeats, "X_test": X_test_nbeats
    }
}

models_dict = {
    "XGBoost": xgboost_model,
    "LSTM": lstm_model,
    "PatchTST": patchtst_model,
    "NBEATS": nbeats_model
}

ensemble_preds, individual_preds = train_and_predict_ensemble(models_dict, data_per_model, y_true=y_test)
ensemble_preds = np.array(ensemble_preds).flatten()[:len(y_test)]
rmse_ens = np.sqrt(mean_squared_error(y_test, ensemble_preds))
print(f"📊 Ensemble RMSE: {rmse_ens:.4f}")
print(f"🔮 Ensemble Future Predictions (last {future_steps}): {ensemble_preds[-future_steps:]}")


def plot_model_predictions(name, y_true, y_pred):
    plt.figure(figsize=(10, 4))
    plt.plot(y_true.values, label="Actual", linewidth=2)
    plt.plot(y_pred, label="Predicted", linestyle='--')
    plt.title(f"{name} – Prediction vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot each model
plot_model_predictions("XGBoost", y_test, y_pred_xgb)
plot_model_predictions("LSTM", y_test, y_pred_lstm)
plot_model_predictions("PatchTST", y_test, y_pred_patchtst)
plot_model_predictions("N-BEATS", y_test, y_pred_nbeats)
plot_model_predictions("Ensemble", y_test, ensemble_preds)

# Combine predictions into one DataFrame
results_df = pd.DataFrame({
    "Actual": y_test.values,
    "XGBoost": y_pred_xgb,
    "LSTM": y_pred_lstm,
    "PatchTST": y_pred_patchtst,
    "NBEATS": y_pred_nbeats,
    "Ensemble": ensemble_preds
})

# Save to CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_df.to_csv(f"prediction_results_{ticker}_{timestamp}.csv", index=False)
print(f"\n📁 Saved predictions to prediction_results_{ticker}_{timestamp}.csv")