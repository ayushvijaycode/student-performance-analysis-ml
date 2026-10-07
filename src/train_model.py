"""
Student Performance Analysis & Score Prediction
Run:
    python src/train_model.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student_performance.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "study_hours_per_day",
    "attendance_percent",
    "sleep_hours",
    "assignment_completion_percent",
    "previous_exam_score",
]
TARGET = "final_exam_score"

# 1. Load data
df = pd.read_csv(DATA_PATH)

# 2. Basic cleaning
df[FEATURES + [TARGET]] = df[FEATURES + [TARGET]].apply(
    lambda col: col.fillna(col.median())
)

print("\nDataset shape:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 3. Prepare features and target
X = df[FEATURES]
y = df[TARGET]

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 5. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-" * 30)
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.3f}")

# 8. Feature coefficients
coefficients = pd.Series(model.coef_, index=FEATURES).sort_values(
    key=abs, ascending=False
)
print("\nFeature coefficients:")
print(coefficients)

# 9. Actual vs predicted plot
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.75)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, linestyle="--")
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted Final Exam Score")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "actual_vs_predicted.png", dpi=160)
plt.show()
