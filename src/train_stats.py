from pathlib import Path
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoETS, AutoARIMA
import joblib

DATA_PATH = Path("data/processed/walmart_processed.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 8
FREQ = "W-FRI"
SEASON_LENGTH = 52


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe el archivo: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["ds"])
    return df


def validate_data(df: pd.DataFrame) -> None:
    required_cols = ["unique_id", "ds", "y"]
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
    """
    Split temporal por serie:
    - train: todo menos las últimas h observaciones de cada serie
    - test: últimas h observaciones de cada serie
    """
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    train_parts = []
    test_parts = []

    for uid, grp in df.groupby("unique_id"):
        grp = grp.sort_values("ds")
        train_parts.append(grp.iloc[:-horizon])
        test_parts.append(grp.iloc[-horizon:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    return train_df, test_df


def train_stats_models(train_df: pd.DataFrame) -> tuple[StatsForecast, pd.DataFrame]:
    models = [
        SeasonalNaive(season_length=SEASON_LENGTH),
        AutoETS(season_length=SEASON_LENGTH),
        AutoARIMA(season_length=SEASON_LENGTH),
    ]

    sf = StatsForecast(
        models=models,
        freq=FREQ,
        n_jobs=1,
    )

    forecasts = sf.forecast(df=train_df[["unique_id", "ds", "y"]], h=HORIZON)
    return sf, forecasts


def save_outputs(sf: StatsForecast, train_df: pd.DataFrame, test_df: pd.DataFrame, forecasts: pd.DataFrame) -> None:
    train_df.to_csv(MODELS_DIR / "train_stats_split.csv", index=False)
    test_df.to_csv(MODELS_DIR / "test_stats_split.csv", index=False)
    forecasts.to_csv(MODELS_DIR / "stats_forecasts.csv", index=False)

    # Guardar metadata simple del entrenamiento
    metadata = {
        "horizon": HORIZON,
        "freq": FREQ,
        "season_length": SEASON_LENGTH,
        "models": ["SeasonalNaive", "AutoETS", "AutoARIMA"],
    }
    joblib.dump(metadata, MODELS_DIR / "stats_metadata.joblib")


def print_summary(train_df: pd.DataFrame, test_df: pd.DataFrame, forecasts: pd.DataFrame) -> None:
    print("\n=== TRAIN ===")
    print(train_df.shape)
    print(train_df.groupby("unique_id").size())

    print("\n=== TEST ===")
    print(test_df.shape)
    print(test_df.groupby("unique_id").size())

    print("\n=== FORECASTS ===")
    print(forecasts.shape)
    print(forecasts.head())


def main():
    df = load_data()
    validate_data(df)

    train_df, test_df = temporal_split(df, HORIZON)
    sf, forecasts = train_stats_models(train_df)

    save_outputs(sf, train_df, test_df, forecasts)
    print_summary(train_df, test_df, forecasts)

    print("\n[OK] train_stats finalizado.")
    print("[OK] Archivos generados:")
    print(f" - {MODELS_DIR / 'train_stats_split.csv'}")
    print(f" - {MODELS_DIR / 'test_stats_split.csv'}")
    print(f" - {MODELS_DIR / 'stats_forecasts.csv'}")
    print(f" - {MODELS_DIR / 'stats_metadata.joblib'}")


if __name__ == "__main__":
    main()