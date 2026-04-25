from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

EVAL_STATS_PATH = MODELS_DIR / "evaluation_metrics.csv"
EVAL_ML_PATH = MODELS_DIR / "evaluation_ml_metrics.csv"
INV_PATH = MODELS_DIR / "inventory_simulation_summary.csv"

OUTPUT_SUMMARY_PATH = MODELS_DIR / "final_evaluation_summary.csv"
OUTPUT_MODEL_COMPARISON_PATH = MODELS_DIR / "final_model_comparison.csv"
OUTPUT_PLOT_WAPE = MODELS_DIR / "plot_wape_comparison.png"
OUTPUT_PLOT_INVENTORY = MODELS_DIR / "plot_inventory_kpis.png"


def load_inputs():
    eval_stats = pd.read_csv(EVAL_STATS_PATH)
    eval_ml = pd.read_csv(EVAL_ML_PATH)
    inv = pd.read_csv(INV_PATH)
    return eval_stats, eval_ml, inv


def build_model_comparison(eval_stats: pd.DataFrame, eval_ml: pd.DataFrame) -> pd.DataFrame:
    stats_models = eval_stats.copy()
    stats_models["model_type"] = "statistical"
    stats_models["WAPE_main"] = stats_models["WAPE"]
    stats_models["MAE_main"] = stats_models["MAE"]
    stats_models["RMSE_main"] = stats_models["RMSE"]

    ml_row = eval_ml.iloc[0]
    ml_models = pd.DataFrame([{
        "model": "LightGBM_q50",
        "model_type": "probabilistic",
        "WAPE_main": ml_row["WAPE_q50"],
        "MAE_main": ml_row["MAE_q50"],
        "RMSE_main": ml_row["RMSE_q50"],
        "Pinball_q10": ml_row["Pinball_q10"],
        "Pinball_q50": ml_row["Pinball_q50"],
        "Pinball_q90": ml_row["Pinball_q90"],
        "Coverage_q10_q90": ml_row["Coverage_q10_q90"],
        "Avg_Interval_Width": ml_row["Avg_Interval_Width"],
    }])

    keep_cols = [
        "model", "model_type", "WAPE_main", "MAE_main", "RMSE_main",
        "Pinball_q10", "Pinball_q50", "Pinball_q90",
        "Coverage_q10_q90", "Avg_Interval_Width"
    ]

    for col in keep_cols:
        if col not in stats_models.columns:
            stats_models[col] = None
        if col not in ml_models.columns:
            ml_models[col] = None

    comparison = pd.concat(
        [stats_models[keep_cols], ml_models[keep_cols]],
        ignore_index=True
    )

    comparison = comparison.sort_values("WAPE_main").reset_index(drop=True)
    return comparison


def build_final_summary(model_comparison: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    best_model = model_comparison.iloc[0]

    overall = inv[inv["unique_id"].astype(str) == "OVERALL"]
    if overall.empty:
        raise ValueError("No se encontró fila OVERALL en inventory_simulation_summary.csv")
    overall = overall.iloc[0]

    summary = pd.DataFrame([{
        "best_model": best_model["model"],
        "best_model_type": best_model["model_type"],
        "best_WAPE": best_model["WAPE_main"],
        "best_MAE": best_model["MAE_main"],
        "best_RMSE": best_model["RMSE_main"],
        "coverage_q10_q90": best_model["Coverage_q10_q90"],
        "avg_interval_width": best_model["Avg_Interval_Width"],
        "inventory_fill_rate": overall["fill_rate"],
        "inventory_stockout_rate": overall["stockout_rate"],
        "inventory_lost_sales_total": overall["lost_sales_total"],
        "inventory_avg_inventory": overall["avg_inventory"],
        "inventory_total_order_qty": overall["total_order_qty"],
    }])

    return summary


def plot_wape_comparison(model_comparison: pd.DataFrame):
    plot_df = model_comparison[["model", "WAPE_main"]].copy()

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["model"], plot_df["WAPE_main"])
    plt.title("Comparación de WAPE por modelo")
    plt.ylabel("WAPE")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_WAPE, dpi=150)
    plt.close()


def plot_inventory_kpis(inv: pd.DataFrame):
    overall = inv[inv["unique_id"].astype(str) == "OVERALL"]
    if overall.empty:
        return
    overall = overall.iloc[0]

    labels = ["Fill Rate", "Stockout Rate"]
    values = [overall["fill_rate"], overall["stockout_rate"]]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title("KPIs operativos de inventario")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_INVENTORY, dpi=150)
    plt.close()


def main():
    eval_stats, eval_ml, inv = load_inputs()

    model_comparison = build_model_comparison(eval_stats, eval_ml)
    final_summary = build_final_summary(model_comparison, inv)

    model_comparison.to_csv(OUTPUT_MODEL_COMPARISON_PATH, index=False)
    final_summary.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    plot_wape_comparison(model_comparison)
    plot_inventory_kpis(inv)

    print("\n=== MODEL COMPARISON ===")
    print(model_comparison)

    print("\n=== FINAL SUMMARY ===")
    print(final_summary)

    print("\n[OK] evaluate_all finalizado.")
    print("[OK] Archivos generados:")
    print(f" - {OUTPUT_MODEL_COMPARISON_PATH}")
    print(f" - {OUTPUT_SUMMARY_PATH}")
    print(f" - {OUTPUT_PLOT_WAPE}")
    print(f" - {OUTPUT_PLOT_INVENTORY}")


if __name__ == "__main__":
    main()