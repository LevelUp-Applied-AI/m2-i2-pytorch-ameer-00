"""
Integration 2 — PyTorch: Housing Price Prediction
Module 2 — Programming for AI & Data Science

Complete each section below. Remove the TODO: comments and pass statements
as you implement each section. Do not change the overall structure.

Before running this script, install PyTorch:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



# ─── Model Definition ─────────────────────────────────────────────────────────

class HousingModel(nn.Module):
    """Neural network for predicting housing prices from property features.

    Architecture: Linear(5, 32) -> ReLU -> Linear(32, 1)
    """

    def __init__(self):
        """Define the model layers."""
        super().__init__()
        self.layer1 = nn.Linear(5, 32)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(32, 1)

    def forward(self, x):
        """Define the forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 5).

        Returns:
            torch.Tensor: Predictions of shape (N, 1).
        """
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# ─── Main Training Script ─────────────────────────────────────────────────────

def main():
    """Load data, train HousingModel, and save predictions."""

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    df = pd.read_csv('data/housing.csv')
    print(f"Data shape: {df.shape}")
    # ── 2. Separate Features and Target ──────────────────────────────────────
    feature_cols = ['area_sqm', 'bedrooms', 'floor', 'age_years', 'distance_to_center_km']
    X = df[feature_cols]
    y = df[['price_jod']]
    # ── 3. Standardize Features ───────────────────────────────────────────────
    X_mean = X.mean()
    X_std  = X.std()
    X_scaled = (X - X_mean) / X_std
    # Why: features have very different scales; standardization ensures
    #      gradient updates are balanced across all input dimensions.

    # ── 4. Convert to Tensors ─────────────────────────────────────────────────
    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values,        dtype=torch.float32)
    print(f"X shape: {X_tensor.shape}")
    print(f"y shape: {y_tensor.shape}")
    # ── 5. Instantiate Model, Loss, and Optimizer ─────────────────────────────
    torch.manual_seed(42)
    indices    = torch.randperm(len(X_tensor))
    X_shuffled = X_tensor[indices]
    y_shuffled = y_tensor[indices]
    split   = int(0.8 * len(X_tensor))
    X_train, X_test = X_shuffled[:split], X_shuffled[split:]
    y_train, y_test = y_shuffled[:split], y_shuffled[split:]
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # ── 6. Training Loop ──────────────────────────────────────────────────────
    model     = HousingModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # ── 7. Save Predictions ───────────────────────────────────────────────────
    num_epochs   = 100
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        predictions = model(X_train)
        loss        = criterion(predictions, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")
    # 8. Evaluation
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train).numpy().flatten()
        test_preds  = model(X_test).numpy().flatten()

    train_actual = y_train.numpy().flatten()
    test_actual  = y_test.numpy().flatten()

    train_mae = np.mean(np.abs(train_actual - train_preds))
    test_mae  = np.mean(np.abs(test_actual  - test_preds))

    def r_squared(actual, preds):
        ss_res = np.sum((actual - preds) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        return 1 - (ss_res / ss_tot)

    train_r2 = r_squared(train_actual, train_preds)
    test_r2  = r_squared(test_actual,  test_preds)

    print(f"\n=== Evaluation ===")
    print(f"Train MAE: {train_mae:.2f} JOD  |  Train R²: {train_r2:.4f}")
    print(f"Test  MAE: {test_mae:.2f} JOD  |  Test  R²: {test_r2:.4f}")

    # 9. Save Predictions CSV
    with torch.no_grad():
        predictions_tensor = model(X_tensor)
    results_df = pd.DataFrame({
        'actual':    y_tensor.numpy().flatten(),
        'predicted': predictions_tensor.numpy().flatten()
    })
    results_df.to_csv('predictions.csv', index=False)
    print("Saved predictions.csv")

    # 10. Actual vs Predicted Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(test_actual, test_preds, alpha=0.6)
    min_val = min(test_actual.min(), test_preds.min())
    max_val = max(test_actual.max(), test_preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    ax.set_xlabel('Actual Price (JOD)')
    ax.set_ylabel('Predicted Price (JOD)')
    ax.set_title('Actual vs Predicted Prices (Test Set)')
    ax.legend()
    fig.savefig('predictions_plot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved predictions_plot.png")

    # 11. Loss Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(num_epochs), loss_history)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curve')
    fig.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved loss_curve.png")


if __name__ == "__main__":
    main()