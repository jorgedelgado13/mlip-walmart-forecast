from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from kafka import KafkaProducer

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "forecast-events")


def get_kafka_producer() -> KafkaProducer | None:
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda v: v.encode("utf-8") if v is not None else None,
            request_timeout_ms=3000,
            retries=1,
        )
        return producer
    except Exception as e:
        print(f"[WARN] Kafka producer no disponible: {e}")
        return None


def publish_event(producer: KafkaProducer | None, event: dict[str, Any]) -> None:
    if producer is None:
        return

    try:
        producer.send(
            KAFKA_TOPIC,
            key=str(event.get("selected_id", "unknown")),
            value=event,
        )
        producer.flush(timeout=3)
    except Exception as e:
        print(f"[WARN] No se pudo publicar evento Kafka: {e}")


def build_inference_event(
    selected_id: Any,
    endpoint: str,
    status: str = "success",
    model_name: str = "LightGBM_quantile",
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "selected_id": selected_id,
        "model": model_name,
        "status": status,
    }