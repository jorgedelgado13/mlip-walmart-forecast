from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

TRAIN_PATH = RAW_DIR / "train.csv"
FEATURES_PATH = RAW_DIR / "features.csv"
STORES_PATH = RAW_DIR / "stores.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga train, features y stores."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"No existe: {TRAIN_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"No existe: {FEATURES_PATH}")
    if not STORES_PATH.exists():
        raise FileNotFoundError(f"No existe: {STORES_PATH}")

    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    features = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    stores = pd.read_csv(STORES_PATH)

    return train, features, stores


def merge_data(
    train: pd.DataFrame, features: pd.DataFrame, stores: pd.DataFrame
) -> pd.DataFrame:
    """Hace merge de train + features + stores."""
    df = train.merge(
        features,
        on=["Store", "Date", "IsHoliday"],
        how="left"
    ).merge(
        stores,
        on="Store",
        how="left"
    )
    return df


def validate_merged(df: pd.DataFrame) -> None:
    required_cols = [
        "Store", "Dept", "Date", "Weekly_Sales", "IsHoliday",
        "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "Type", "Size"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas tras merge: {missing}")

    if df.empty:
        raise ValueError("El dataframe merged está vacío.")

    if df["Date"].isna().any():
        raise ValueError("Date tiene nulos.")

    if df["Weekly_Sales"].isna().any():
        raise ValueError("Weekly_Sales tiene nulos.")


def build_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un sample pequeño y estable:
    - Store == 1
    - 7 departamentos
    """
    df = df.copy()

    # Filtrar tienda 1
    df = df[df["Store"] == 1].copy()

    # Escoger 7 departamentos con más observaciones
    top_depts = (
        df.groupby("Dept")
        .size()
        .sort_values(ascending=False)
        .head(7)
        .index
        .tolist()
    )

    df = df[df["Dept"].isin(top_depts)].copy()

    # Crear id tipo "1_Dept"
    df["id"] = df["Store"].astype(str) + "_" + df["Dept"].astype(str)

    # Imputación simple para markdowns
    markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    for col in markdown_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Holiday a entero
    df["IsHoliday"] = df["IsHoliday"].astype(int)

    # Ordenar
    df = df.sort_values(["Dept", "Date"]).reset_index(drop=True)

    return df


def create_processed(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas al formato estándar de forecasting."""
    out = df.copy()

    out = out.rename(
        columns={
            "id": "item_id",
            "Dept": "unique_id",
            "Date": "ds",
            "Weekly_Sales": "y",
        }
    )

    keep_cols = [
        "item_id",
        "Store",
        "unique_id",
        "ds",
        "y",
        "IsHoliday",
        "Type",
        "Size",
        "Temperature",
        "Fuel_Price",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        "CPI",
        "Unemployment",
    ]

    existing_cols = [c for c in keep_cols if c in out.columns]
    out = out[existing_cols].copy()

    out = out.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return out


def save_outputs(raw_sample: pd.DataFrame, processed: pd.DataFrame) -> None:
    raw_sample.to_csv(RAW_DIR / "walmart_sample_raw.csv", index=False)
    processed.to_csv(PROCESSED_DIR / "walmart_processed.csv", index=False)
    processed.to_parquet(PROCESSED_DIR / "walmart_processed.parquet", index=False)


def print_summary(train, features, stores, merged, raw_sample, processed) -> None:
    print("\n=== INPUTS ===")
    print("train:", train.shape)
    print("features:", features.shape)
    print("stores:", stores.shape)

    print("\n=== MERGED ===")
    print(merged.shape)

    print("\n=== RAW SAMPLE ===")
    print(raw_sample.shape)
    print("Store únicos:", raw_sample["Store"].nunique())
    print("Dept únicos:", raw_sample["Dept"].nunique())
    print("Fechas:", raw_sample["Date"].min(), "->", raw_sample["Date"].max())

    print("\n=== PROCESSED ===")
    print(processed.shape)
    print("Series únicas:", processed["unique_id"].nunique())
    print(processed.head())


def main():
    train, features, stores = load_inputs()
    merged = merge_data(train, features, stores)
    validate_merged(merged)

    raw_sample = build_sample(merged)
    processed = create_processed(raw_sample)

    save_outputs(raw_sample, processed)
    print_summary(train, features, stores, merged, raw_sample, processed)

    print("\n[OK] data_prep finalizado.")
    print("[OK] Archivos generados:")
    print(f" - {RAW_DIR / 'walmart_sample_raw.csv'}")
    print(f" - {PROCESSED_DIR / 'walmart_processed.csv'}")
    print(f" - {PROCESSED_DIR / 'walmart_processed.parquet'}")


if __name__ == "__main__":
    main()