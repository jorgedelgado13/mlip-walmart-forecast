import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from flask import Flask, render_template_string, request
import pandas as pd
import json
import joblib
import base64
from prometheus_flask_exporter import PrometheusMetrics
from kafka_utils import get_kafka_producer, publish_event, build_inference_event

DATA_PATH = BASE_DIR / "data" / "processed" / "walmart_processed.csv"
TEST_FUTURE_PATH = BASE_DIR / "models" / "test_ml_split.csv"
INVENTORY_PATH = BASE_DIR / "models" / "inventory_simulation_summary.csv"

MODEL_ARTIFACT_PATH = BASE_DIR / "models" / "mlforecast_model.joblib"
MODEL_METADATA_PATH = BASE_DIR / "models" / "ml_metadata.joblib"
MODEL_TRAINING_INFO_PATH = BASE_DIR / "models" / "ml_training_info.json"

FINAL_MODEL_COMPARISON_PATH = BASE_DIR / "models" / "final_model_comparison.csv"
FINAL_EVAL_SUMMARY_PATH = BASE_DIR / "models" / "final_evaluation_summary.csv"
PLOT_WAPE_PATH = BASE_DIR / "models" / "plot_wape_comparison.png"
PLOT_INVENTORY_PATH = BASE_DIR / "models" / "plot_inventory_kpis.png"

app = Flask(__name__)
metrics = PrometheusMetrics(app)
metrics.info("app_info", "Application info", version="1.0.0", app_name="mlip_walmart_forecast")

kafka_producer = None

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

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>MLIP Walmart Forecast</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 30px;
            background: #f7f7f7;
        }
        h1, h2 {
            color: #222;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        label, select, button {
            font-size: 16px;
        }
        select, button {
            padding: 8px 12px;
            margin-right: 10px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            background: white;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
        }
        th {
            background: #f0f0f0;
        }
        .kpi {
            display: inline-block;
            min-width: 180px;
            margin-right: 16px;
            margin-bottom: 12px;
            padding: 12px;
            border-radius: 10px;
            background: #fafafa;
            border: 1px solid #e5e5e5;
            vertical-align: top;
        }
        .small {
            font-size: 14px;
            color: #555;
        }
        .img-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .img-grid img {
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 10px;
            background: white;
        }
        @media (max-width: 900px) {
            .img-grid {
                grid-template-columns: 1fr;
            }
        }
        code {
            background: #f2f2f2;
            padding: 2px 6px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <h1>MLIP Walmart Forecast Dashboard</h1>

    <div class="card">
        <h2>Estado del modelo</h2>
        {% if model_status %}
            <div class="kpi"><strong>Artefacto</strong><br>{{ model_status["artifact_exists"] }}</div>
            <div class="kpi"><strong>Metadata</strong><br>{{ model_status["metadata_exists"] }}</div>
            <div class="kpi"><strong>Training info</strong><br>{{ model_status["training_info_exists"] }}</div>
            <div class="kpi"><strong>Entrenado en</strong><br>{{ model_status["trained_at"] }}</div>
            <div class="kpi"><strong>Series train</strong><br>{{ model_status["n_series_train"] }}</div>
            <div class="kpi"><strong>Rows train</strong><br>{{ model_status["train_rows"] }}</div>
        {% else %}
            <p>No se pudo cargar el estado del modelo.</p>
        {% endif %}
        <p class="small">
            Esta app carga el artefacto entrenado <code>mlforecast_model.joblib</code> y genera predicciones en inferencia.
        </p>
    </div>

    <div class="card">
        <h2>Resumen ejecutivo de evaluación</h2>
        {% if eval_summary %}
            <div class="kpi"><strong>Mejor modelo WAPE</strong><br>{{ eval_summary["best_model"] }}</div>
            <div class="kpi"><strong>Tipo</strong><br>{{ eval_summary["best_model_type"] }}</div>
            <div class="kpi"><strong>Best WAPE</strong><br>{{ eval_summary["best_WAPE"] }}</div>
            <div class="kpi"><strong>Best MAE</strong><br>{{ eval_summary["best_MAE"] }}</div>
            <div class="kpi"><strong>Best RMSE</strong><br>{{ eval_summary["best_RMSE"] }}</div>
            <div class="kpi"><strong>Coverage q10-q90</strong><br>{{ eval_summary["coverage_q10_q90"] }}</div>
            <div class="kpi"><strong>Avg Interval Width</strong><br>{{ eval_summary["avg_interval_width"] }}</div>
            <div class="kpi"><strong>Fill Rate</strong><br>{{ eval_summary["inventory_fill_rate"] }}</div>
            <div class="kpi"><strong>Stockout Rate</strong><br>{{ eval_summary["inventory_stockout_rate"] }}</div>
            <div class="kpi"><strong>Lost Sales</strong><br>{{ eval_summary["inventory_lost_sales_total"] }}</div>
        {% else %}
            <p>No se pudo cargar el resumen consolidado.</p>
        {% endif %}
        <p class="small">
            Nota: el mejor benchmark por WAPE puede no coincidir con el modelo desplegado si el objetivo del sistema incluye forecasting probabilístico e inventario.
        </p>
    </div>

    <div class="card">
        <form method="get" action="/">
            <label for="unique_id">Selecciona departamento:</label>
            <select name="unique_id" id="unique_id">
                {% for uid in unique_ids %}
                    <option value="{{ uid }}" {% if uid == selected_id %}selected{% endif %}>{{ uid }}</option>
                {% endfor %}
            </select>
            <button type="submit">Ver forecast</button>
        </form>
    </div>

    <div class="card">
        <h2>KPIs de inventario por serie</h2>
        {% if inventory_row %}
            <div class="kpi"><strong>Demand Total</strong><br>{{ inventory_row["demand_total"] }}</div>
            <div class="kpi"><strong>Sales Served</strong><br>{{ inventory_row["sales_served_total"] }}</div>
            <div class="kpi"><strong>Lost Sales</strong><br>{{ inventory_row["lost_sales_total"] }}</div>
            <div class="kpi"><strong>Fill Rate</strong><br>{{ inventory_row["fill_rate"] }}</div>
            <div class="kpi"><strong>Stockout Rate</strong><br>{{ inventory_row["stockout_rate"] }}</div>
            <div class="kpi"><strong>Avg Inventory</strong><br>{{ inventory_row["avg_inventory"] }}</div>
        {% else %}
            <p>No hay datos de inventario para esta serie.</p>
        {% endif %}
    </div>

    <div class="card">
        <h2>Forecast probabilístico</h2>
        <canvas id="forecastChart" width="1200" height="450"></canvas>
    </div>

    <div class="card">
        <h2>Comparación consolidada de modelos</h2>
        {{ comparison_table|safe }}
    </div>

    <div class="card">
        <h2>Gráficos de evaluación</h2>
        <div class="img-grid">
            {% if wape_plot %}
                <div>
                    <h3>WAPE por modelo</h3>
                    <img src="data:image/png;base64,{{ wape_plot }}" alt="WAPE comparison">
                </div>
            {% endif %}
            {% if inventory_plot %}
                <div>
                    <h3>KPIs operativos</h3>
                    <img src="data:image/png;base64,{{ inventory_plot }}" alt="Inventory KPIs">
                </div>
            {% endif %}
        </div>
    </div>

    <div class="card">
        <h2>Detalle forecast</h2>
        {{ forecast_table|safe }}
    </div>

    <script>
        const labels = {{ labels | safe }};
        const historical = {{ historical | safe }};
        const actual = {{ actual | safe }};
        const q10 = {{ q10 | safe }};
        const q50 = {{ q50 | safe }};
        const q90 = {{ q90 | safe }};

        new Chart(document.getElementById('forecastChart'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Histórico',
                        data: historical,
                        borderWidth: 2,
                        spanGaps: true
                    },
                    {
                        label: 'Real test',
                        data: actual,
                        borderWidth: 2,
                        spanGaps: true
                    },
                    {
                        label: 'q10',
                        data: q10,
                        borderWidth: 1,
                        borderDash: [5, 5],
                        spanGaps: true
                    },
                    {
                        label: 'q50',
                        data: q50,
                        borderWidth: 2,
                        spanGaps: true
                    },
                    {
                        label: 'q90',
                        data: q90,
                        borderWidth: 1,
                        borderDash: [5, 5],
                        spanGaps: true
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: true }
                },
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
    </script>
</body>
</html>
"""


def encode_image_base64(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_base_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe {DATA_PATH}")
    if not TEST_FUTURE_PATH.exists():
        raise FileNotFoundError(f"No existe {TEST_FUTURE_PATH}")
    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(f"No existe {INVENTORY_PATH}")

    df_hist = pd.read_csv(DATA_PATH, parse_dates=["ds"])
    df_test_future = pd.read_csv(TEST_FUTURE_PATH, parse_dates=["ds"])
    df_inv = pd.read_csv(INVENTORY_PATH)

    return df_hist, df_test_future, df_inv


def load_model_objects():
    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"No existe {MODEL_ARTIFACT_PATH}")
    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(f"No existe {MODEL_METADATA_PATH}")
    if not MODEL_TRAINING_INFO_PATH.exists():
        raise FileNotFoundError(f"No existe {MODEL_TRAINING_INFO_PATH}")

    model = joblib.load(MODEL_ARTIFACT_PATH)
    metadata = joblib.load(MODEL_METADATA_PATH)
    training_info = pd.read_json(MODEL_TRAINING_INFO_PATH, typ="series").to_dict()

    model_status = {
        "artifact_exists": MODEL_ARTIFACT_PATH.exists(),
        "metadata_exists": MODEL_METADATA_PATH.exists(),
        "training_info_exists": MODEL_TRAINING_INFO_PATH.exists(),
        "trained_at": training_info.get("trained_at", "N/A"),
        "train_rows": training_info.get("train_rows", "N/A"),
        "n_series_train": training_info.get("n_series_train", "N/A"),
        "models": metadata.get("models", []),
        "horizon": metadata.get("horizon", "N/A"),
    }

    return model, metadata, training_info, model_status


def load_evaluation_outputs():
    comparison_df = pd.read_csv(FINAL_MODEL_COMPARISON_PATH) if FINAL_MODEL_COMPARISON_PATH.exists() else pd.DataFrame()
    summary_df = pd.read_csv(FINAL_EVAL_SUMMARY_PATH) if FINAL_EVAL_SUMMARY_PATH.exists() else pd.DataFrame()

    comparison_table = comparison_df.round(4).to_html(index=False) if not comparison_df.empty else "<p>No disponible.</p>"
    eval_summary = summary_df.iloc[0].to_dict() if not summary_df.empty else None

    if eval_summary:
        for k, v in eval_summary.items():
            if isinstance(v, float):
                eval_summary[k] = round(v, 4)

    wape_plot = encode_image_base64(PLOT_WAPE_PATH)
    inventory_plot = encode_image_base64(PLOT_INVENTORY_PATH)

    return comparison_table, eval_summary, wape_plot, inventory_plot


def generate_live_forecasts(model, df_test_future: pd.DataFrame) -> pd.DataFrame:
    future_exog = df_test_future[["unique_id", "ds"] + EXOG_COLS].copy()
    preds = model.predict(h=8, X_df=future_exog)

    merged = df_test_future.merge(
        preds,
        on=["unique_id", "ds"],
        how="inner"
    )

    arr = merged[["q10", "q50", "q90"]].to_numpy()
    arr.sort(axis=1)

    merged["q10"] = arr[:, 0]
    merged["q50"] = arr[:, 1]
    merged["q90"] = arr[:, 2]

    return merged


def build_chart_data(df_hist, df_fcst_live, selected_id):
    hist = df_hist[df_hist["unique_id"] == selected_id].sort_values("ds").copy()
    fcst = df_fcst_live[df_fcst_live["unique_id"] == selected_id].sort_values("ds").copy()

    if hist.empty or fcst.empty:
        return [], [], [], [], [], []

    forecast_dates = set(fcst["ds"])
    hist_train = hist[~hist["ds"].isin(forecast_dates)].copy()
    all_dates = sorted(hist["ds"].tolist())

    hist_map = dict(zip(hist_train["ds"], hist_train["y"]))
    actual_map = dict(zip(fcst["ds"], fcst["y"]))
    q10_map = dict(zip(fcst["ds"], fcst["q10"]))
    q50_map = dict(zip(fcst["ds"], fcst["q50"]))
    q90_map = dict(zip(fcst["ds"], fcst["q90"]))

    labels = [d.strftime("%Y-%m-%d") for d in all_dates]
    historical = [round(hist_map[d], 2) if d in hist_map else None for d in all_dates]
    actual = [round(actual_map[d], 2) if d in actual_map else None for d in all_dates]
    q10 = [round(q10_map[d], 2) if d in q10_map else None for d in all_dates]
    q50 = [round(q50_map[d], 2) if d in q50_map else None for d in all_dates]
    q90 = [round(q90_map[d], 2) if d in q90_map else None for d in all_dates]

    return labels, historical, actual, q10, q50, q90

@app.route("/health")
def health():
    return {
        "status": "ok",
        "service": "mlip-walmart-forecast",
        "model_artifact": MODEL_ARTIFACT_PATH.exists(),
        "metadata": MODEL_METADATA_PATH.exists(),
        "training_info": MODEL_TRAINING_INFO_PATH.exists(),
    }, 200


@app.route("/")
def home():
    df_hist, df_test_future, df_inv = load_base_data()
    model, metadata, training_info, model_status = load_model_objects()

    df_fcst_live = generate_live_forecasts(model, df_test_future)

    unique_ids = sorted(df_hist["unique_id"].unique().tolist())
    selected_id = request.args.get("unique_id", unique_ids[0])
    try:
        selected_id = int(selected_id)
    except ValueError:
        pass

    labels, historical, actual, q10, q50, q90 = build_chart_data(df_hist, df_fcst_live, selected_id)

    forecast_table = (
        df_fcst_live[df_fcst_live["unique_id"] == selected_id]
        .sort_values("ds")[["ds", "y", "q10", "q50", "q90"]]
        .round(2)
        .to_html(index=False)
    )

    inv_match = df_inv[df_inv["unique_id"].astype(str) == str(selected_id)]
    inventory_row = inv_match.iloc[0].to_dict() if not inv_match.empty else None

    if inventory_row:
        for k, v in inventory_row.items():
            if isinstance(v, float):
                inventory_row[k] = round(v, 4)

    comparison_table, eval_summary, wape_plot, inventory_plot = load_evaluation_outputs()

    global kafka_producer
    if kafka_producer is None:
        kafka_producer = get_kafka_producer()

    event = build_inference_event(
        selected_id=selected_id,
        endpoint="/",
        status="success",
        model_name="LightGBM_quantile",
    )
    publish_event(kafka_producer, event)

    return render_template_string(
        HTML_TEMPLATE,
        unique_ids=unique_ids,
        selected_id=selected_id,
        inventory_row=inventory_row,
        forecast_table=forecast_table,
        model_status=model_status,
        comparison_table=comparison_table,
        eval_summary=eval_summary,
        wape_plot=wape_plot,
        inventory_plot=inventory_plot,
        labels=json.dumps(labels),
        historical=json.dumps(historical),
        actual=json.dumps(actual),
        q10=json.dumps(q10),
        q50=json.dumps(q50),
        q90=json.dumps(q90),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)