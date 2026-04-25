from pathlib import Path
import os
import pandas as pd
import wandb

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

EVAL_STATS_PATH = MODELS_DIR / "evaluation_metrics.csv"
EVAL_ML_PATH = MODELS_DIR / "evaluation_ml_metrics.csv"
INV_PATH = MODELS_DIR / "inventory_simulation_summary.csv"
MODEL_ARTIFACT_PATH = MODELS_DIR / "mlforecast_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "ml_metadata.joblib"
TRAINING_INFO_PATH = MODELS_DIR / "ml_training_info.json"
DATASET_PATH = DATA_DIR / "walmart_processed.csv"

PROJECT = os.getenv("WANDB_PROJECT", "mlip-walmart-forecast")
ENTITY = os.getenv("WANDB_ENTITY")  # opcional
MODE = os.getenv("WANDB_MODE", "online")


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def main():
    eval_stats = safe_read_csv(EVAL_STATS_PATH)
    eval_ml = safe_read_csv(EVAL_ML_PATH)
    inv = safe_read_csv(INV_PATH)

    config = {
        "project_type": "forecasting_retail_mlops",
        "model_family": "LightGBM_quantile + statistical baselines",
        "baselines": ["SeasonalNaive", "AutoETS", "AutoARIMA"],
        "probabilistic_model": "LightGBM_quantile",
        "inventory_policy": "dynamic reorder point / order-up-to",
        "dataset": "Walmart reduced sample (Store 1, 7 departments)",
        "mode": MODE,
    }

    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        config=config,
        job_type="training-evaluation",
        tags=["forecasting", "retail", "mlops", "probabilistic", "inventory"],
        mode=MODE,
    )

    # -----------------------------
    # Log metrics: classical models
    # -----------------------------
    if eval_stats is not None and not eval_stats.empty:
        for _, row in eval_stats.iterrows():
            model_name = str(row["model"])
            wandb.log({
                f"stats/{model_name}/MAE": float(row["MAE"]),
                f"stats/{model_name}/RMSE": float(row["RMSE"]),
                f"stats/{model_name}/WAPE": float(row["WAPE"]),
            })

    # --------------------------------
    # Log metrics: probabilistic model
    # --------------------------------
    if eval_ml is not None and not eval_ml.empty:
        row = eval_ml.iloc[0]
        wandb.log({
            "ml/MAE_q50": float(row["MAE_q50"]),
            "ml/RMSE_q50": float(row["RMSE_q50"]),
            "ml/WAPE_q50": float(row["WAPE_q50"]),
            "ml/Pinball_q10": float(row["Pinball_q10"]),
            "ml/Pinball_q50": float(row["Pinball_q50"]),
            "ml/Pinball_q90": float(row["Pinball_q90"]),
            "ml/Coverage_q10_q90": float(row["Coverage_q10_q90"]),
            "ml/Avg_Interval_Width": float(row["Avg_Interval_Width"]),
        })

        run.summary["best_probabilistic_model"] = "LightGBM_quantile"

    # -----------------------------
    # Log inventory / business KPIs
    # -----------------------------
    if inv is not None and not inv.empty:
        overall = inv[inv["unique_id"].astype(str) == "OVERALL"]
        if not overall.empty:
            row = overall.iloc[0]
            wandb.log({
                "inventory/demand_total": float(row["demand_total"]),
                "inventory/sales_served_total": float(row["sales_served_total"]),
                "inventory/lost_sales_total": float(row["lost_sales_total"]),
                "inventory/avg_inventory": float(row["avg_inventory"]),
                "inventory/total_order_qty": float(row["total_order_qty"]),
                "inventory/fill_rate": float(row["fill_rate"]),
                "inventory/stockout_rate": float(row["stockout_rate"]),
            })

    # -----------------------------
    # Log artifacts
    # -----------------------------
    if DATASET_PATH.exists():
        dataset_artifact = wandb.Artifact(
            name="walmart_processed_dataset",
            type="dataset",
            description="Processed reduced Walmart dataset for forecasting"
        )
        dataset_artifact.add_file(str(DATASET_PATH))
        run.log_artifact(dataset_artifact)

    model_artifact = wandb.Artifact(
        name="lightgbm_quantile_forecaster",
        type="model",
        description="Trained MLForecast LightGBM quantile model"
    )

    if MODEL_ARTIFACT_PATH.exists():
        model_artifact.add_file(str(MODEL_ARTIFACT_PATH))
    if MODEL_METADATA_PATH.exists():
        model_artifact.add_file(str(MODEL_METADATA_PATH))
    if TRAINING_INFO_PATH.exists():
        model_artifact.add_file(str(TRAINING_INFO_PATH))
    if EVAL_STATS_PATH.exists():
        model_artifact.add_file(str(EVAL_STATS_PATH))
    if EVAL_ML_PATH.exists():
        model_artifact.add_file(str(EVAL_ML_PATH))
    if INV_PATH.exists():
        model_artifact.add_file(str(INV_PATH))

    run.log_artifact(model_artifact)

    run.summary["artifact_logged"] = True
    run.summary["pipeline_status"] = "completed"

    run.finish()
    print("[OK] W&B logging completed.")


if __name__ == "__main__":
    main()