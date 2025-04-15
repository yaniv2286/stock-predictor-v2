import pandas as pd
from data.preprocess import build_model_input
from data.model_input import generate_model_input
from models import xgboost_model, lstm_model
from sklearn.metrics import root_mean_squared_error

# ---- PARAMETERS ----
ticker = "AAPL"
timeframes = ["1d", "1h"]
start = "2023-01-01"
end = "2023-06-01"
target_col = "close_AAPL_1d"
n_steps_ahead = 1
test_size = 0.2
timesteps = 10  # For LSTM

# ---- BUILD FEATURES ----
df = build_model_input(ticker, timeframes, start, end)

print(f"\n✅ Data shape: {df.shape}\n")
print("📈 Sample Columns:\n", df.columns[:15].tolist(), "...\n")
print("📊 Tail of DataFrame:\n", df.tail(5), "\n")

# ---- MODEL INPUT ----
X_train, y_train, X_test, y_test = generate_model_input(
    df,
    target_col=target_col,
    n_steps_ahead=n_steps_ahead,
    test_size=test_size
)

print("✅ Model Input Shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

# ==========================
# 🚀 XGBOOST MODEL
# ==========================
X_train_xgb, y_train_xgb, X_test_xgb = xgboost_model.prepare_data(X_train, y_train, X_test)
xgb = xgboost_model.build_model()
xgb_preds = xgboost_model.train_and_predict(X_train_xgb, y_train_xgb, X_test_xgb, xgb)
xgb_rmse = root_mean_squared_error(y_test, xgb_preds)
print(f"\n📊 XGBoost RMSE: {xgb_rmse:.4f}")

# ==========================
# 🧠 LSTM MODEL
# ==========================
X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm = lstm_model.prepare_data(X_train, y_train, X_test, timesteps)
lstm = lstm_model.build_model(input_shape=X_train_lstm.shape[1:])
lstm_preds = lstm_model.train_and_predict(X_train_lstm, y_train_lstm, X_test_lstm, lstm)
lstm_rmse = root_mean_squared_error(y_test_lstm, lstm_preds)
print(f"\n🧠 LSTM RMSE: {lstm_rmse:.4f}")
