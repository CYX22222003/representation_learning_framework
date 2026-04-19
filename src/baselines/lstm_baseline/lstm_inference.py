from baselines.lstm_baseline.lstm_model import LSTMModel
import torch
import numpy as np

model_path = "./baselines/lstm_baseline/event_stacked_lstm.pth"

SEQ_LEN = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(42)
recent_prices = np.random.rand(SEQ_LEN).astype(np.float32)  # [0.0, 1.0]

model_path = "./baselines/lstm_baseline/event_stacked_lstm.pth"
model = LSTMModel()
try:
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print("Model loaded successfully.")
except FileNotFoundError:
    print("No saved model found. Using randomly initialized model.")

model.to(device)
model.eval()

def predict_next_price(price_sequence):
    if len(price_sequence) != SEQ_LEN:
        raise ValueError(f"Input sequence length must be {SEQ_LEN}")
    x = torch.tensor(price_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        pred = model(x)
    return pred.item()

# ----------------------------
# Test pseudo prediction
# ----------------------------
if __name__ == "__main__":
    next_price = predict_next_price(recent_prices)
    print("Predicted next price:", next_price)