import torch
from src.data_loader import load_data
from src.preprocessing import preprocess_data, create_sequences
from src.model_lstm import LSTMModel
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import create_dataloader

def main():
    df = load_data()
    scaled_data, scaler = preprocess_data(df)

    X, y = create_sequences(scaled_data, seq_length=24)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_loader = create_dataloader(X_train, y_train)

    model = LSTMModel(input_size=X.shape[2])
    train_model(model, train_loader, epochs=10)

    model.eval()
    with torch.no_grad():
        predictions = model(
            torch.tensor(X_test, dtype=torch.float32)
        ).squeeze().numpy()

    rmse, mae, mape = evaluate_model(y_test, predictions)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("MAPE:", mape)

if __name__ == "__main__":
    main()
