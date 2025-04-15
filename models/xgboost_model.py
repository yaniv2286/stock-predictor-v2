import xgboost as xgb
import numpy as np

def prepare_data(X_train, y_train, X_test):
    return X_train, y_train, X_test

def build_model(params=None):
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror'
        }
    return xgb.XGBRegressor(**params)

def train_and_predict(X_train, y_train, X_test, model):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds

def predict_future_sequence(model, recent_X, steps=1):
    preds = []
    current_input = recent_X.copy()
    for _ in range(steps):
        pred = model.predict(current_input.values.reshape(1, -1))[0]
        preds.append(pred)
    return np.array(preds)
