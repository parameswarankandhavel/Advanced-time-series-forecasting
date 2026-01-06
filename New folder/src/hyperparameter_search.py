import torch
import numpy as np
from src.model_lstm import LSTMModel
from src.train import train_model
from src.utils import create_dataloader

def hyperparameter_grid_search(X_train, y_train, input_size):
    """
    Simple grid search for LSTM hyperparameters.
    """
    hidden_sizes = [32, 64]
    learning_rates = [0.001, 0.0005]

    best_loss = float("inf")
    best_model = None

    for hidden in hidden_sizes:
        for lr in learning_rates:
            model = LSTMModel(input_size=input_size, hidden_size=hidden)
            dataloader = create_dataloader(X_train, y_train)

            train_model(model, dataloader, epochs=20, lr=lr)

            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_train, dtype=torch.float32)).squeeze()
                loss = torch.mean((preds - torch.tensor(y_train, dtype=torch.float32)) ** 2)

            if loss < best_loss:
                best_loss = loss
                best_model = model

    return best_model
