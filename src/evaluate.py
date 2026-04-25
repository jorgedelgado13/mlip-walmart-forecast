from pathlib import Path
import pandas as pd
import numpy as np

MODELS_DIR = Path("models")
TEST_PATH = MODELS_DIR / "test_stats_split.csv"
FORECASTS_PATH = MODELS_DIR / "stats_forecasts.csv"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"No existe: {TEST_PATH}")
    if not FORECASTS_PATH.exists():
        raise FileNotFoundError(f"No existe: {FORECASTS_PATH}")

    test_df = pd.read_csv(TEST_PATH, parse_dates=["ds"])
    forecasts_df = pd.read_csv(FORECASTS_PATH, parse_dates=["ds"])
    return test_df, forecasts_df


def validate_inputs(test_df: pd.DataFrame, forecasts_df: pd.DataFrame) -> None:
    required_test = ["unique_id", "ds", "y"]
    required_fcst = ["unique_id", "ds"]

    missing_test = [c for c in required_test if c not in test_df.columns]
    missing_fcst = [c for c in required_fcst if c not in forecasts_df.columns]

    if missing_test:
        raise ValueError(f"Faltan columnas en test: {missing_test}")
    if missing_fcst:
        raise ValueError(f"Faltan columnas en forecasts: {missing_fcst}")

    if test_df.empty:
        raise ValueError("test_df está vacío.")
    if forecasts_df.empty:
        raise ValueError("forecasts_df está vacío.")

    model_cols = [c for c in forecasts_df.columns if c not in ["unique_id", "ds"]]
    if not model_cols:
        raise ValueError("No se encontraron columnas de modelos en stats_forecasts.csv.")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def evaluate_models(test_df: pd.DataFrame, forecasts_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = test_df.merge(
        forecasts_df,
        on=["unique_id", "ds"],
        how="inner"
    )

    model_cols = [c for c in forecasts_df.columns if c not in ["unique_id", "ds"]]
    results = []

    for model_name in model_cols:
        y_true = merged["y"].values
        y_pred = merged[model_name].values

        results.append({
            "model": model_name,
            "MAE": mae(y_true, y_pred),
            "RMSE": rmse(y_true, y_pred),
            "WAPE": wape(y_true, y_pred),
        })

    results_df = pd.DataFrame(results).sort_values("WAPE").reset_index(drop=True)
    return merged, results_df


def save_outputs(merged: pd.DataFrame, results_df: pd.DataFrame) -> None:
    merged.to_csv(MODELS_DIR / "evaluation_metrics_merged.csv", index=False)
    results_df.to_csv(MODELS_DIR / "evaluation_metrics.csv", index=False)


def print_summary(results_df: pd.DataFrame) -> None:
    print("\n=== MÉTRICAS MODELOS ESTADÍSTICOS ===")
    print(results_df)

    best_model = results_df.iloc[0]["model"]
    print(f"\n[OK] Mejor modelo según WAPE: {best_model}")


def main():
    test_df, forecasts_df = load_inputs()
    validate_inputs(test_df, forecasts_df)

    merged, results_df = evaluate_models(test_df, forecasts_df)
    save_outputs(merged, results_df)
    print_summary(results_df)

    print("\n[OK] evaluate finalizado.")
    print("[OK] Archivos generados:")
    print(f" - {MODELS_DIR / 'evaluation_metrics_merged.csv'}")
    print(f" - {MODELS_DIR / 'evaluation_metrics.csv'}")


if __name__ == "__main__":
    main()