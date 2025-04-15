import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def prepare_data(X_train, y_train, X_test, timesteps=10):
    """
    Reshapes flat features into 3D [samples, timesteps, features] for LSTM.
    Uses sliding window to create sequences.
    """
    def create_sequences(X, y, timesteps):
        Xs, ys = [], []
        for i in range(len(X) - timesteps):
            Xs.append(X.iloc[i:i+timesteps].values)
            ys.append(y.iloc[i+timesteps])
        return np.array(Xs), np.array(ys)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, timesteps)
    X_test_seq, y_test_seq = create_sequences(X_test, y_train[-len(X_test):], timesteps)

    return X_train_seq, y_train_seq, X_test_seq, y_test_seq

def build_model(input_shape):
    """
    Builds and returns an LSTM model.
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_predict(X_train, y_train, X_test, model):
    """
    Trains the LSTM model and predicts on X_test.
    """
    es = EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[es], verbose=0)
    preds = model.predict(X_test).flatten()
    return preds

def predict_future_sequence(model, recent_X, steps=1):
    """
    Autoregressive LSTM forecasting (optional).
    """
    predictions = []
    current_input = recent_X.copy()

    for _ in range(steps):
        pred = model.predict(current_input.reshape(1, *current_input.shape))[0][0]
        predictions.append(pred)

        # Shift window
        new_input = np.append(current_input[1:], [[*current_input[-1]]], axis=0)
        new_input[-1][-1] = pred
        current_input = new_input

    return np.array(predictions)
