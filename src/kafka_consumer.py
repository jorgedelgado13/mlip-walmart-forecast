from __future__ import annotations

import json
import os
import time

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "forecast-events")


def create_consumer_with_retry(max_attempts: int = 20, sleep_seconds: int = 5) -> KafkaConsumer:
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[INFO] Intentando conectar a Kafka ({attempt}/{max_attempts})...")
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                group_id="forecast-consumer-group",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            )
            print(f"[OK] Kafka consumer escuchando topic: {KAFKA_TOPIC}")
            return consumer
        except NoBrokersAvailable as e:
            last_error = e
            print(f"[WARN] Kafka aún no disponible. Reintentando en {sleep_seconds}s...")
            time.sleep(sleep_seconds)

    raise last_error if last_error else RuntimeError("No se pudo crear el consumer.")


def main():
    consumer = create_consumer_with_retry()

    for message in consumer:
        print("[KAFKA EVENT]", message.value)


if __name__ == "__main__":
    main()