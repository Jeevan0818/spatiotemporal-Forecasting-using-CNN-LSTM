import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.model import CNNLSTM

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# Example dummy data - replace with real satellite sequences
X = torch.randn(100, 5, 3, 128, 128)  # (samples, timesteps, channels, H, W)
y = torch.randn(100, 3)               # (samples, forecast steps)

dataset = TimeSeriesDataset(X, y)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = CNNLSTM(c_in=3, cnn_out=128, lstm_hidden=64, forecast_steps=3)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

for epoch in range(10):
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
    # Save the trained model
torch.save(model.state_dict(), "model_weights.pth")
print("✅ Model saved as model_weights.pth")

