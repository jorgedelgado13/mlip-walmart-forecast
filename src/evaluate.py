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
    required_fcst = ["unique_id", "ds", "q10", "q50", "q90"]

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


def fix_quantile_crossing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reordena q10, q50, q90 por fila para asegurar:
    q10 <= q50 <= q90
    """
    arr = df[["q10", "q50", "q90"]].to_numpy()
    arr_sorted = np.sort(arr, axis=1)

    out = df.copy()
    out["q10"] = arr_sorted[:, 0]
    out["q50"] = arr_sorted[:, 1]
    out["q90"] = arr_sorted[:, 2]
    return out


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def interval_coverage(y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray) -> float:
    inside = ((y_true >= q_low) & (y_true <= q_high)).mean()
    return float(inside)


def avg_interval_width(q_low: np.ndarray, q_high: np.ndarray) -> float:
    return float(np.mean(q_high - q_low))


def evaluate(test_df: pd.DataFrame, forecasts_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = test_df.merge(
        forecasts_df,
        on=["unique_id", "ds"],
        how="inner"
    )

    merged = fix_quantile_crossing(merged)

    y_true = merged["y"].values
    q10 = merged["q10"].values
    q50 = merged["q50"].values
    q90 = merged["q90"].values

    metrics = pd.DataFrame([{
        "model": "LightGBM_quantile",
        "MAE_q50": mae(y_true, q50),
        "RMSE_q50": rmse(y_true, q50),
        "WAPE_q50": wape(y_true, q50),
        "Pinball_q10": pinball_loss(y_true, q10, 0.10),
        "Pinball_q50": pinball_loss(y_true, q50, 0.50),
        "Pinball_q90": pinball_loss(y_true, q90, 0.90),
        "Coverage_q10_q90": interval_coverage(y_true, q10, q90),
        "Avg_Interval_Width": avg_interval_width(q10, q90),
    }])

    return merged, metrics


def save_outputs(merged: pd.DataFrame, metrics: pd.DataFrame) -> None:
    merged.to_csv(MODELS_DIR / "evaluation_ml_merged.csv", index=False)
    metrics.to_csv(MODELS_DIR / "evaluation_ml_metrics.csv", index=False)


def print_summary(metrics: pd.DataFrame, merged: pd.DataFrame) -> None:
    print("\n=== MÉTRICAS ML PROBABILÍSTICO ===")
    print(metrics)

    bad_order = ((merged["q10"] > merged["q50"]) | (merged["q50"] > merged["q90"])).sum()
    print(f"\nFilas con cuantiles mal ordenados después de corrección: {bad_order}")


def main():
    test_df, forecasts_df = load_inputs()
    validate_inputs(test_df, forecasts_df)

    merged, metrics = evaluate(test_df, forecasts_df)
    save_outputs(merged, metrics)
    print_summary(metrics, merged)

    print("\n[OK] evaluate_ml finalizado.")
    print("[OK] Archivos generados:")
    print(f" - {MODELS_DIR / 'evaluation_ml_merged.csv'}")
    print(f" - {MODELS_DIR / 'evaluation_ml_metrics.csv'}")


if __name__ == "__main__":
    main()