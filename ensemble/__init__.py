import numpy as np
from sklearn.metrics import mean_absolute_error

def train_and_predict_ensemble(models: dict, data_per_model: dict, y_true=None):
    """
    Trains and predicts using multiple models and returns an ensemble forecast.

    Args:
        models (dict): {model_name: model_module}
        data_per_model (dict): {
            model_name: {
                'X_train': ..., 'y_train': ..., 'X_test': ...
            }
        }
        y_true (array-like, optional): True values to compute model weights.

    Returns:
        tuple: (ensemble_predictions, individual_predictions)
    """
    predictions = {}
    errors = {}

    for model_name, model_module in models.items():
        data = data_per_model[model_name]
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']

        model_instance = model_module.build_model()
        preds = model_module.train_and_predict(X_train, y_train, X_test, model_instance)
        predictions[model_name] = preds

        if y_true is not None:
            mae = mean_absolute_error(y_true, preds[:len(y_true)])
            errors[model_name] = mae
        else:
            errors[model_name] = 1.0  # fallback

    # Inverse MAE weighting
    weights = {name: 1 / (err + 1e-6) for name, err in errors.items()}
    total_weight = sum(weights.values())
    weights = {name: w / total_weight for name, w in weights.items()}

    # Weighted ensemble
    ensemble_preds = sum(predictions[name] * weights[name] for name in predictions)
    return ensemble_preds, predictions
