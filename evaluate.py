from models.model import CNNLSTM
import torch
import numpy as np

# Load trained model
model = CNNLSTM(c_in=3, cnn_out=128, lstm_hidden=64, forecast_steps=3)
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# Dummy input for testing
sample_input = np.random.randn(5, 3, 128, 128).astype(np.float32)  # Use float32

# Convert to torch tensor and add batch dimension
sample_tensor = torch.tensor(sample_input).unsqueeze(0)  # Shape: (1, 5, 3, 128, 128)

with torch.no_grad():
    prediction = model(sample_tensor)

print("🔮 Forecast:", prediction.numpy())
