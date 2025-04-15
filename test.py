from data.preprocess import build_model_input
from data.model_input import generate_model_input
from models import xgboost_model, lstm_model, patchtst_model
from sklearn.metrics import mean_squared_error
import numpy as np

# ========== PARAMETERS ==========
ticker = "AAPL"
timeframes = ["1d"]
start = "2023-01-01"
end = "2023-06-01"
target_col = f"close_{ticker}_{timeframes[0]}"
future_steps = 5  # Number of future steps to predict

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

# ✅ Flatten and align
y_pred_xgb = np.array(y_pred_xgb).flatten()[:len(y_test)]
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f"📊 XGBoost RMSE: {rmse_xgb:.4f}")

# Predict next N future steps
future_xgb = xgboost_model.predict_future_sequence(model_xgb, X_test_xgb.iloc[-1], steps=future_steps)
print(f"🔮 XGBoost Future {future_steps} Predictions: {future_xgb}")

# ========== LSTM ==========
print("\n🧠 Training LSTM...")
X_train_lstm, y_train_lstm, X_test_lstm = lstm_model.prepare_data(X_train, y_train, X_test, window_size=10)
model_lstm = lstm_model.build_model(input_shape=X_train_lstm.shape[1:])
y_pred_lstm, rmse_lstm = lstm_model.train_and_predict(X_train_lstm, y_train_lstm, X_test_lstm, model_lstm)
print(f"📐 LSTM RMSE: {rmse_lstm:.4f}")

# Future prediction
recent_window_lstm = X_test.tail(10)
future_lstm = lstm_model.predict_future_sequence(model_lstm, recent_window_lstm, steps=future_steps, window_size=10)
print(f"🔮 LSTM Future {future_steps} Predictions: {future_lstm}")

# ========== PatchTST ==========
print("\n🚀 Training PatchTST...")
X_train_patch, y_train_patch, X_test_patch = patchtst_model.prepare_data(X_train, y_train, X_test, patch_len=8)
model_patch = patchtst_model.build_model(X_train_patch.shape[1:])
y_pred_patch, rmse_patch = patchtst_model.train_and_predict(X_train_patch, y_train_patch, X_test_patch, model_patch)
print(f"📐 PatchTST RMSE: {rmse_patch:.4f}")

# ========== SUMMARY ==========
print("\n📊 Model RMSE Summary:")
print(f"XGBoost   RMSE: {rmse_xgb:.4f}")
print(f"LSTM      RMSE: {rmse_lstm:.4f}")
print(f"PatchTST  RMSE: {rmse_patch:.4f}")
