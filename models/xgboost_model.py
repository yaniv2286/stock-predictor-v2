import xgboost as xgb
import numpy as np

def prepare_data(X_train, y_train, X_test):
    return X_train, y_train, X_test

def build_model(params=None):
    if params is None:
        params = {
            'n_estimators': 300,         # More trees for better performance
            'max_depth': 4,              # Shallower trees for regularization
            'learning_rate': 0.03,       # Smaller learning rate
            'subsample': 0.8,            # Subsample for each boosting round
            'colsample_bytree': 0.7,     # Feature subsampling
            'reg_alpha': 0.1,            # L1 regularization
            'reg_lambda': 1.0,           # L2 regularization
            'objective': 'reg:squarederror',
            'random_state': 42
        }
    model = xgb.XGBRegressor(**params)
    return model

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
