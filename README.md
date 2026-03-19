[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YUvA8hIt)
# Integration 2 — PyTorch: Housing Price Prediction

## What the Model Predicts
Predicts apartment prices in Jordanian Dinars (JOD) based on 5 features:
- `area_sqm` — Apartment area in square meters
- `bedrooms` — Number of bedrooms
- `floor` — Floor number
- `age_years` — Building age in years
- `distance_to_center_km` — Distance to city center

## Training Configuration
- Epochs: 100
- Learning rate: 0.01
- Optimizer: Adam
- Loss function: MSELoss
- Architecture: Linear(5, 32) → ReLU → Linear(32, 1)

## Training Outcome
Loss decreased from 1,950,603,008 at epoch 0 to 1,944,461,696 at epoch 90.
The loss decreased steadily across all epochs.

## Behavioral Observation
Loss decreased gradually and consistently across all 100 epochs,
with no sudden jumps or instability, indicating stable training.



**Module 2 — Programming for AI & Data Science**

See the [Module 2 Integration Task Guide](https://levelup-applied-ai.github.io/aispire-14005-pages/modules/module-2/learner/integration-guide) for full instructions.

---

## Quick Reference

**File to complete:** `train.py`

**Install PyTorch before running:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Branch:** `integration-2/pytorch`

**Submit:** PR URL → TalentLMS Unit 8 text field

## Challenge Tier 1 — Evaluation

### Train/Test Split
- Training set: 160 records (80%)
- Test set: 40 records (20%)

### Metrics
| Metric | Train | Test |
|--------|-------|------|
| MAE    | 42,270 JOD | 40,027 JOD |
| R²     | -8.70 | -10.73 |

### Overfitting Analysis
The model shows similar performance on train and test sets,
suggesting no overfitting. However, the negative R² indicates
the model needs more epochs or a larger architecture to learn
meaningful patterns from the data.

### Output Files
- `predictions_plot.png` — Actual vs Predicted scatter plot
- `loss_curve.png` — Training loss over 100 epochs