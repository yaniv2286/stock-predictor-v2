import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error

def reshape_for_patchtst(X, patch_len=8):
    num_samples, num_features = X.shape
    new_len = (num_features // patch_len) * patch_len
    X = X.iloc[:, :new_len]
    return X.values.reshape((num_samples, new_len // patch_len, patch_len))

def prepare_data(X_train, y_train, X_test, patch_len=8):
    X_train_reshaped = reshape_for_patchtst(X_train, patch_len)
    X_test_reshaped = reshape_for_patchtst(X_test, patch_len)
    return X_train_reshaped, y_train.values, X_test_reshaped

def build_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.LayerNormalization()(inputs)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=8)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model

def train_and_predict(X_train, y_train, X_test, model, epochs=20):
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    preds = model.predict(X_test).flatten()
    rmse = np.sqrt(mean_squared_error(y_train[-len(preds):], preds))
    return preds, rmse
