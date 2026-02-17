from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MODEL_PATH = Path("artifacts/sales_model.joblib")


def generate_dataset(n_samples: int = 6000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    week = rng.integers(1, 53, size=n_samples)
    store_id = rng.integers(1, 51, size=n_samples)
    promo = rng.binomial(1, 0.35, size=n_samples)
    holiday = rng.binomial(1, 0.1, size=n_samples)
    fuel_price = rng.normal(3.15, 0.45, size=n_samples).clip(1.5, 5.5)
    cpi = rng.normal(215, 7, size=n_samples).clip(190, 240)
    unemployment = rng.normal(7.1, 1.4, size=n_samples).clip(3.0, 12.5)

    seasonal = 1700 * np.sin((2 * np.pi * week) / 52)
    store_effect = store_id * 130

    weekly_sales = (
        18000
        + seasonal
        + store_effect
        + promo * 6000
        + holiday * 8500
        - fuel_price * 1200
        - (cpi - 200) * 90
        - unemployment * 520
        + rng.normal(0, 850, size=n_samples)
    )

    return pd.DataFrame(
        {
            "week": week,
            "store_id": store_id,
            "promo": promo,
            "holiday": holiday,
            "fuel_price": fuel_price,
            "cpi": cpi,
            "unemployment": unemployment,
            "weekly_sales": weekly_sales,
        }
    )


def build_model() -> Pipeline:
    numeric_features = ["week", "store_id", "promo", "holiday", "fuel_price", "cpi", "unemployment"]
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_features)],
        remainder="drop",
    )

    regressor = RandomForestRegressor(
        n_estimators=350,
        max_depth=16,
        min_samples_split=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def train_and_save_model() -> float:
    df = generate_dataset()
    X = df.drop(columns=["weekly_sales"])
    y = df["weekly_sales"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    model = build_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return r2


if __name__ == "__main__":
    score = train_and_save_model()
    print(f"Model trained successfully with R² score: {score:.4f}")
