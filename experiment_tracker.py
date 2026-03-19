"""
Stretch — Experiment Tracker
Module 2 — Applied AI & ML Systems

Runs multiple training configurations, logs results, and prints a leaderboard.
"""

import json
import time
import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ─── Model ────────────────────────────────────────────────────────────────────

class HousingModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(5, hidden_size)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# ─── Train One Config ─────────────────────────────────────────────────────────

def run_experiment(X_train, y_train, X_test, y_test, config):
    model     = HousingModel(config['hidden_size'])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    start = time.time()
    for epoch in range(config['num_epochs']):
        model.train()
        preds = model(X_train)
        loss  = criterion(preds, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    elapsed = time.time() - start

    model.eval()
    with torch.no_grad():
        train_preds = model(X_train).numpy().flatten()
        test_preds  = model(X_test).numpy().flatten()

    train_actual = y_train.numpy().flatten()
    test_actual  = y_test.numpy().flatten()

    def mae(actual, preds):
        return float(np.mean(np.abs(actual - preds)))

    def r2(actual, preds):
        ss_res = np.sum((actual - preds) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        return float(1 - ss_res / ss_tot)

    return {
        'config':      config,
        'train_mae':   mae(train_actual, train_preds),
        'test_mae':    mae(test_actual,  test_preds),
        'train_r2':    r2(train_actual,  train_preds),
        'test_r2':     r2(test_actual,   test_preds),
        'time_seconds': round(elapsed, 3)
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load and prepare data
    df = pd.read_csv('data/housing.csv')
    feature_cols = ['area_sqm', 'bedrooms', 'floor', 'age_years', 'distance_to_center_km']
    X = df[feature_cols]
    y = df[['price_jod']]

    X_mean   = X.mean()
    X_std    = X.std()
    X_scaled = (X - X_mean) / X_std

    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values,        dtype=torch.float32)

    # Train/Test Split
    torch.manual_seed(42)
    indices    = torch.randperm(len(X_tensor))
    X_shuffled = X_tensor[indices]
    y_shuffled = y_tensor[indices]
    split      = int(0.8 * len(X_tensor))
    X_train, X_test = X_shuffled[:split], X_shuffled[split:]
    y_train, y_test = y_shuffled[:split], y_shuffled[split:]

    # Hyperparameter Grid
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.0001]
    hidden_sizes   = [16, 32, 64, 128]
    num_epochs_list = [50, 100, 200]

    grid = list(itertools.product(learning_rates, hidden_sizes, num_epochs_list))
    print(f"Total experiments: {len(grid)}")

    # Run Experiments
    results = []
    for i, (lr, hs, epochs) in enumerate(grid):
        config = {'learning_rate': lr, 'hidden_size': hs, 'num_epochs': epochs}
        print(f"[{i+1}/{len(grid)}] LR={lr}, Hidden={hs}, Epochs={epochs}", end=' ... ')
        result = run_experiment(X_train, y_train, X_test, y_test, config)
        results.append(result)
        print(f"Test MAE={result['test_mae']:.0f}")

    # Save experiments.json
    with open('experiments.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved experiments.json")

    # Sort by test MAE
    results.sort(key=lambda r: r['test_mae'])

    # Print Leaderboard
    print("\n=== Top 10 Configurations ===")
    print(f"{'Rank':>4} | {'LR':>8} | {'Hidden':>6} | {'Epochs':>6} | {'Test MAE':>12} | {'Test R²':>8} | {'Time (s)':>8}")
    print("-" * 70)
    for rank, r in enumerate(results[:10], 1):
        c = r['config']
        print(f"{rank:>4} | {c['learning_rate']:>8} | {c['hidden_size']:>6} | {c['num_epochs']:>6} | {r['test_mae']:>12.2f} | {r['test_r2']:>8.4f} | {r['time_seconds']:>8.3f}")

    # Summary Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    for hs in hidden_sizes:
        subset = [r for r in results if r['config']['hidden_size'] == hs]
        lrs    = [r['config']['learning_rate'] for r in subset]
        maes   = [r['test_mae'] for r in subset]
        ax.scatter(lrs, maes, label=f"hidden={hs}", alpha=0.7)

    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate (log scale)')
    ax.set_ylabel('Test MAE (JOD)')
    ax.set_title('Test MAE vs Learning Rate by Hidden Size')
    ax.legend()
    fig.savefig('experiment_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved experiment_summary.png")

    best = results[0]
    print(f"\nBest config: LR={best['config']['learning_rate']}, Hidden={best['config']['hidden_size']}, Epochs={best['config']['num_epochs']}")
    print(f"Best Test MAE: {best['test_mae']:.2f} JOD")


if __name__ == "__main__":
    main()