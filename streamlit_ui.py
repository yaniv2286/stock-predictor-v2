import streamlit as st
import pandas as pd
from data.preprocess import build_model_input
from data.model_input import generate_model_input
from ensemble import train_and_predict_ensemble
from models import xgboost_model, lstm_model, patchtst_model, nbeats_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Model registry
MODELS = {
    "XGBoost": xgboost_model,
    "LSTM": lstm_model,
    "PatchTST": patchtst_model,
    "N-BEATS": nbeats_model,
    "Ensemble": None  # Placeholder for ensemble logic
}

st.set_page_config(page_title="📈 Stock Prediction Dashboard", layout="wide")
st.title("📊 Stock Forecast with Multiple Models")

# Sidebar inputs
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
timeframes = st.sidebar.multiselect("Timeframes", ["1d", "1wk", "1mo"], default=["1d"])
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-06-01"))
model_choice = st.sidebar.selectbox("Model", list(MODELS.keys()))
forecast_steps = st.sidebar.slider("Forecast Horizon (steps)", 1, 30, 5)

if st.sidebar.button("Run Forecast"):
    try:
        df = build_model_input(ticker, timeframes, str(start_date), str(end_date))
        target_col = f"{timeframes[0]}_close"
        X_train, y_train, X_test, y_test = generate_model_input(df, target_col)

        if model_choice == "Ensemble":
            data_per_model = {}
            model_preds = {}
            for name, model in MODELS.items():
                if name != "Ensemble":
                    Xtr, ytr, Xte = model.prepare_data(X_train, y_train, X_test)
                    data_per_model[name] = {
                        "X_train": Xtr,
                        "y_train": ytr,
                        "X_test": Xte
                    }
            models_to_use = {name: MODELS[name] for name in data_per_model}
            ensemble_preds, individual_preds = train_and_predict_ensemble(models_to_use, data_per_model, y_true=y_test)
            ensemble_preds = pd.Series(ensemble_preds, index=y_test.index[:len(ensemble_preds)])
            st.subheader("📊 Ensemble Model Results")
            st.line_chart(pd.DataFrame({"Actual": y_test, "Ensemble": ensemble_preds}))
            rmse = mean_squared_error(y_test, ensemble_preds, squared=False)
            st.success(f"Ensemble RMSE: {rmse:.4f}")
        else:
            model = MODELS[model_choice]
            Xtr, ytr, Xte = model.prepare_data(X_train, y_train, X_test)
            model_instance = model.build_model()
            y_pred = model.train_and_predict(Xtr, ytr, Xte, model_instance)
            y_pred = pd.Series(y_pred, index=y_test.index[:len(y_pred)])
            st.subheader(f"📊 {model_choice} Model Results")
            st.line_chart(pd.DataFrame({"Actual": y_test, model_choice: y_pred}))
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            st.success(f"{model_choice} RMSE: {rmse:.4f}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
