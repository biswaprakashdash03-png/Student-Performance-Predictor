from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "student_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "student_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

FEATURES = ["study_hours", "attendance", "previous_score", "assignments_completed"]
TARGET = "exam_score"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 3),
        "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 3),
        "r2": round(float(r2_score(y_test, predictions)), 3),
        "test_rows": int(len(y_test)),
        "training_rows": int(len(y_train))
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Model training complete!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"MAE:  {metrics['mae']}")
    print(f"RMSE: {metrics['rmse']}")
    print(f"R²:   {metrics['r2']}")

if __name__ == "__main__":
    main()
