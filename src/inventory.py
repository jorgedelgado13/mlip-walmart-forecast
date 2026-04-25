from pathlib import Path
import pandas as pd
import numpy as np

MODELS_DIR = Path("models")
INPUT_PATH = MODELS_DIR / "evaluation_ml_merged.csv"

OUTPUT_DETAIL = MODELS_DIR / "inventory_simulation_detail.csv"
OUTPUT_SUMMARY = MODELS_DIR / "inventory_simulation_summary.csv"


LEAD_TIME = 1  # semanas
SERVICE_BUFFER_FACTOR = 0.25  # qué tan agresivo usar q90 sobre q50
INITIAL_INVENTORY_FACTOR = 1.20  # inventario inicial respecto a q50


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"No existe: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, parse_dates=["ds"])
    return df


def validate_data(df: pd.DataFrame) -> None:
    required_cols = ["unique_id", "ds", "y", "q10", "q50", "q90"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    if df.empty:
        raise ValueError("El dataframe está vacío.")


def simulate_inventory_for_series(grp: pd.DataFrame) -> pd.DataFrame:
    grp = grp.sort_values("ds").reset_index(drop=True).copy()

    inventory = max(0.0, grp.loc[0, "q50"] * INITIAL_INVENTORY_FACTOR)
    pending_orders = []  # lista de dicts: {"arrival_idx": i, "qty": x}

    rows = []

    for i, row in grp.iterrows():
        # 1) recibir órdenes pendientes
        arriving_qty = 0.0
        still_pending = []
        for po in pending_orders:
            if po["arrival_idx"] == i:
                arriving_qty += po["qty"]
            else:
                still_pending.append(po)
        pending_orders = still_pending

        inventory += arriving_qty

        inventory_start = inventory

        demand = float(row["y"])
        q50 = float(row["q50"])
        q90 = float(row["q90"])

        # 2) atender demanda
        sales_served = min(inventory, demand)
        lost_sales = max(0.0, demand - inventory)
        inventory -= sales_served

        # 3) política dinámica basada en forecast
        reorder_point = q50
        safety_buffer = max(0.0, q90 - q50) * SERVICE_BUFFER_FACTOR
        order_up_to = q90 + safety_buffer

        order_qty = 0.0
        if inventory < reorder_point:
            order_qty = max(0.0, order_up_to - inventory)
            arrival_idx = i + LEAD_TIME
            if arrival_idx < len(grp):
                pending_orders.append({
                    "arrival_idx": arrival_idx,
                    "qty": order_qty
                })

        rows.append({
            "unique_id": row["unique_id"],
            "ds": row["ds"],
            "y": demand,
            "q50": q50,
            "q90": q90,
            "inventory_start": inventory_start,
            "sales_served": sales_served,
            "lost_sales": lost_sales,
            "inventory_end": inventory,
            "reorder_point": reorder_point,
            "order_up_to": order_up_to,
            "order_qty": order_qty,
            "arriving_qty": arriving_qty,
            "stockout_flag": int(lost_sales > 0),
        })

    return pd.DataFrame(rows)


def build_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        detail_df.groupby("unique_id")
        .agg(
            demand_total=("y", "sum"),
            sales_served_total=("sales_served", "sum"),
            lost_sales_total=("lost_sales", "sum"),
            avg_inventory=("inventory_end", "mean"),
            total_order_qty=("order_qty", "sum"),
            stockout_weeks=("stockout_flag", "sum"),
            weeks=("stockout_flag", "count"),
        )
        .reset_index()
    )

    summary["fill_rate"] = np.where(
        summary["demand_total"] > 0,
        summary["sales_served_total"] / summary["demand_total"],
        np.nan,
    )

    summary["stockout_rate"] = np.where(
        summary["weeks"] > 0,
        summary["stockout_weeks"] / summary["weeks"],
        np.nan,
    )

    overall = pd.DataFrame([{
        "unique_id": "OVERALL",
        "demand_total": detail_df["y"].sum(),
        "sales_served_total": detail_df["sales_served"].sum(),
        "lost_sales_total": detail_df["lost_sales"].sum(),
        "avg_inventory": detail_df["inventory_end"].mean(),
        "total_order_qty": detail_df["order_qty"].sum(),
        "stockout_weeks": detail_df["stockout_flag"].sum(),
        "weeks": len(detail_df),
        "fill_rate": detail_df["sales_served"].sum() / detail_df["y"].sum() if detail_df["y"].sum() > 0 else np.nan,
        "stockout_rate": detail_df["stockout_flag"].mean(),
    }])

    summary = pd.concat([summary, overall], ignore_index=True)
    return summary


def save_outputs(detail_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    detail_df.to_csv(OUTPUT_DETAIL, index=False)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)


def print_summary(summary_df: pd.DataFrame) -> None:
    print("\n=== INVENTORY SIMULATION SUMMARY ===")
    print(summary_df)


def main():
    df = load_data()
    validate_data(df)

    detail_parts = []
    for uid, grp in df.groupby("unique_id"):
        detail_parts.append(simulate_inventory_for_series(grp))

    detail_df = pd.concat(detail_parts).reset_index(drop=True)
    summary_df = build_summary(detail_df)

    save_outputs(detail_df, summary_df)
    print_summary(summary_df)

    print("\n[OK] inventory finalizado.")
    print("[OK] Archivos generados:")
    print(f" - {OUTPUT_DETAIL}")
    print(f" - {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()