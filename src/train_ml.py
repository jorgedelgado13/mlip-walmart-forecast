from pathlib import Path
from datetime import datetime
import pandas as pd
import joblib
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean

DATA_PATH = Path("data/processed/walmart_processed.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 8
FREQ = "W-FRI"

EXOG_COLS = [
    "IsHoliday",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
]

MODEL_ARTIFACT_PATH = MODELS_DIR / "mlforecast_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "ml_metadata.joblib"
TRAINING_INFO_PATH = MODELS_DIR / "ml_training_info.json"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["ds"])
    return df


def validate_data(df: pd.DataFrame) -> None:
    required_cols = ["unique_id", "ds", "y"] + EXOG_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    if df.empty:
        raise ValueError("El dataset está vacío.")

    if df["ds"].isna().any():
        raise ValueError("La columna ds tiene nulos.")

    if df["y"].isna().any():
        raise ValueError("La columna y tiene nulos.")


def temporal_split(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    train_parts = []
    test_parts = []

    for _, grp in df.groupby("unique_id"):
        grp = grp.sort_values("ds")
        train_parts.append(grp.iloc[:-horizon])
        test_parts.append(grp.iloc[-horizon:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    return train_df, test_df


def build_models():
    models = {
        "q10": lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.10,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
        ),
        "q50": lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.50,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
        ),
        "q90": lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.90,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
        ),
    }
    return models


def fit_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame):
    models = build_models()

    fcst = MLForecast(
        models=models,
        freq=FREQ,
        lags=[1, 2, 4, 8, 12],
        lag_transforms={
            4: [RollingMean(window_size=4)],
            8: [RollingMean(window_size=8)],
        },
        date_features=[],
    )

    fcst.fit(
        train_df[["unique_id", "ds", "y"] + EXOG_COLS],
        static_features=[],
    )

    future_exog = test_df[["unique_id", "ds"] + EXOG_COLS].copy()
    preds = fcst.predict(h=HORIZON, X_df=future_exog)

    return fcst, preds


def save_outputs(train_df: pd.DataFrame, test_df: pd.DataFrame, preds: pd.DataFrame, fcst) -> None:
    train_df.to_csv(MODELS_DIR / "train_ml_split.csv", index=False)
    test_df.to_csv(MODELS_DIR / "test_ml_split.csv", index=False)
    preds.to_csv(MODELS_DIR / "ml_quantile_forecasts.csv", index=False)

    metadata = {
        "horizon": HORIZON,
        "freq": FREQ,
        "exogenous_cols": EXOG_COLS,
        "models": ["q10", "q50", "q90"],
        "lags": [1, 2, 4, 8, 12],
        "model_artifact": str(MODEL_ARTIFACT_PATH.name),
    }
    joblib.dump(metadata, MODEL_METADATA_PATH)

    # Guardar el modelo entrenado
    joblib.dump(fcst, MODEL_ARTIFACT_PATH)

    # Guardar info simple del entrenamiento
    training_info = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "n_series_train": int(train_df["unique_id"].nunique()),
        "n_series_test": int(test_df["unique_id"].nunique()),
        "artifact_path": str(MODEL_ARTIFACT_PATH),
        "metadata_path": str(MODEL_METADATA_PATH),
    }
    pd.Series(training_info).to_json(TRAINING_INFO_PATH, indent=2)


def print_summary(train_df: pd.DataFrame, test_df: pd.DataFrame, preds: pd.DataFrame) -> None:
    print("\n=== TRAIN ML ===")
    print(train_df.shape)

    print("\n=== TEST ML ===")
    print(test_df.shape)

    print("\n=== QUANTILE FORECASTS ===")
    print(preds.shape)
    print(preds.head())

    print("\n=== ARTIFACTS ===")
    print(MODEL_ARTIFACT_PATH)
    print(MODEL_METADATA_PATH)
    print(TRAINING_INFO_PATH)


def main():
    df = load_data()
    validate_data(df)

    train_df, test_df = temporal_split(df, HORIZON)
    fcst, preds = fit_and_predict(train_df, test_df)

    save_outputs(train_df, test_df, preds, fcst)
    print_summary(train_df, test_df, preds)

    print("\n[OK] train_ml finalizado.")
    print("[OK] Archivos generados:")
    print(f" - {MODELS_DIR / 'train_ml_split.csv'}")
    print(f" - {MODELS_DIR / 'test_ml_split.csv'}")
    print(f" - {MODELS_DIR / 'ml_quantile_forecasts.csv'}")
    print(f" - {MODEL_METADATA_PATH}")
    print(f" - {MODEL_ARTIFACT_PATH}")
    print(f" - {TRAINING_INFO_PATH}")


if __name__ == "__main__":
    main()