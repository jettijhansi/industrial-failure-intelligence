
import numpy as np
import pandas as pd

from config import RAW_DIR, PROCESSED_DIR, GROUND_TRUTH_DIR


TELEMETRY_FILE = RAW_DIR / "telemetry.csv"
FAILURE_FILE = GROUND_TRUTH_DIR / "failure_plan.csv"
OUTPUT_FILE = PROCESSED_DIR / "training_dataset.csv"

PREDICTION_HORIZON_HOURS = 24
ROLLING_WINDOW_HOURS = 6


def load_data():
    if not TELEMETRY_FILE.exists():
        raise FileNotFoundError(f"Missing telemetry file: {TELEMETRY_FILE}")

    if not FAILURE_FILE.exists():
        raise FileNotFoundError(f"Missing failure plan: {FAILURE_FILE}")

    telemetry = pd.read_csv(TELEMETRY_FILE)
    failures = pd.read_csv(FAILURE_FILE)

    required_telemetry = {
        "machine_id", "sensor_id", "timestamp", "value"
    }
    required_failures = {
        "machine_id", "failure_time"
    }

    if not required_telemetry.issubset(telemetry.columns):
        raise ValueError("Telemetry CSV is missing required columns.")

    if not required_failures.issubset(failures.columns):
        raise ValueError("Failure plan is missing required columns.")

    telemetry["timestamp"] = pd.to_datetime(
        telemetry["timestamp"], errors="coerce"
    )
    failures["failure_time"] = pd.to_datetime(
        failures["failure_time"], errors="coerce"
    )

    telemetry["value"] = pd.to_numeric(
        telemetry["value"], errors="coerce"
    )

    telemetry = telemetry.dropna(
        subset=["machine_id", "sensor_id", "timestamp", "value"]
    )

    failures = failures.dropna(
        subset=["machine_id", "failure_time"]
    )

    return telemetry, failures


def create_hourly_sensor_features(telemetry):
    # Sensor IDs end with the sensor type and sensor number:
    # ...-TEMP-01, ...-VIB-01, ...-PRESS-01, etc.
    telemetry = telemetry.copy()

    telemetry["sensor_type"] = (
        telemetry["sensor_id"]
        .astype(str)
        .str.split("-")
        .str[-2]
        .str.lower()
    )

    valid_types = {
        "temp", "vib", "press", "current", "humidity"
    }

    telemetry = telemetry[
        telemetry["sensor_type"].isin(valid_types)
    ]

    if telemetry.empty:
        raise ValueError("No recognized sensor types were found.")

    # Aggregate readings by machine, hour and sensor type.
    hourly = (
        telemetry.groupby(
            ["machine_id", "timestamp", "sensor_type"],
            as_index=False
        )["value"]
        .mean()
    )

    wide = hourly.pivot(
        index=["machine_id", "timestamp"],
        columns="sensor_type",
        values="value"
    )

    wide.columns = [
        f"{column}_value" for column in wide.columns
    ]

    wide = wide.reset_index()

    # Create a complete hourly timeline per machine.
    # Missing sensor readings remain NaN, never zero.
    all_frames = []

    for machine_id, machine_data in wide.groupby("machine_id"):
        machine_data = machine_data.sort_values(
            "timestamp"
        ).set_index("timestamp")

        full_index = pd.date_range(
            start=machine_data.index.min(),
            end=machine_data.index.max(),
            freq="h"
        )

        machine_data = machine_data.reindex(full_index)
        machine_data["machine_id"] = machine_id
        machine_data.index.name = "timestamp"

        all_frames.append(machine_data.reset_index())

    return pd.concat(all_frames, ignore_index=True)


def add_rolling_features(data):
    sensor_columns = [
        column for column in data.columns
        if column.endswith("_value")
    ]

    result_frames = []

    for machine_id, machine_data in data.groupby("machine_id"):
        machine_data = machine_data.sort_values(
            "timestamp"
        ).copy()

        generated = {}

        for column in sensor_columns:
            series = machine_data[column]

            generated[f"{column}_mean_6h"] = (
                series.rolling(
                    ROLLING_WINDOW_HOURS,
                    min_periods=1
                ).mean()
            )

            generated[f"{column}_std_6h"] = (
                series.rolling(
                    ROLLING_WINDOW_HOURS,
                    min_periods=2
                ).std()
            )

            generated[f"{column}_change_1h"] = series.diff()

        feature_frame = pd.DataFrame(
            generated,
            index=machine_data.index
        )

        machine_data = pd.concat(
            [machine_data, feature_frame],
            axis=1
        )

        result_frames.append(machine_data)

    return pd.concat(result_frames, ignore_index=True)


def add_failure_labels(features, failures):
    features = features.copy()
    features["failure_within_24h"] = 0

    failures_by_machine = {
        machine_id: group["failure_time"].sort_values().tolist()
        for machine_id, group in failures.groupby("machine_id")
    }

    for machine_id, machine_data in features.groupby("machine_id"):
        failure_times = failures_by_machine.get(machine_id, [])

        if not failure_times:
            continue

        timestamps = machine_data["timestamp"]

        positive_mask = np.zeros(len(machine_data), dtype=bool)

        for failure_time in failure_times:
            hours_until_failure = (
                failure_time - timestamps
            ).dt.total_seconds() / 3600

            positive_mask |= (
                (hours_until_failure > 0)
                & (hours_until_failure <= PREDICTION_HORIZON_HOURS)
            ).to_numpy()

        features.loc[
            machine_data.index[positive_mask],
            "failure_within_24h"
        ] = 1

    return features


def main():
    print("\nIMPROVED FAILURE PREDICTION FEATURES")
    print("=" * 45)

    telemetry, failures = load_data()

    print(f"Telemetry readings loaded: {len(telemetry):,}")
    print(f"Failure records loaded: {len(failures):,}")

    features = create_hourly_sensor_features(telemetry)

    print(f"Hourly machine rows: {len(features):,}")

    features = add_rolling_features(features)
    features = add_failure_labels(features, failures)

    features = features.replace(
        [np.inf, -np.inf], np.nan
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    positive = int(features["failure_within_24h"].sum())
    negative = len(features) - positive

    feature_columns = [
        column for column in features.columns
        if column not in {
            "machine_id", "timestamp", "failure_within_24h"
        }
    ]

    print("\nFEATURE GENERATION COMPLETE")
    print("=" * 45)
    print(f"Dataset rows: {len(features):,}")
    print(f"Positive labels: {positive:,}")
    print(f"Negative labels: {negative:,}")
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
