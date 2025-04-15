import xgboost as xgb
import pandas as pd
import numpy as np

def prepare_data(X_train, y_train, X_test):
    """
    XGBoost uses standard tabular input, so no transformation needed.
    """
    return X_train, y_train, X_test

def build_model(params=None):
    """
    Builds and returns an XGBoost regressor.
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror'
        }
    model = xgb.XGBRegressor(**params)
    return model

def train_and_predict(X_train, y_train, X_test, model):
    """
    Trains XGBoost on X_train/y_train and returns predictions for X_test.
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds

def predict_future_sequence(model, recent_X, steps=1):
    """
    Autoregressively predicts 'steps' into the future using XGBoost.
    This is optional and mainly for live forecasting.
    """
    preds = []
    current_input = recent_X.copy()

    for _ in range(steps):
        pred = model.predict(current_input.values.reshape(1, -1))[0]
        preds.append(pred)

        # You can optionally update current_input here to include pred if needed
        # For now, we keep it static

    return np.array(preds)
