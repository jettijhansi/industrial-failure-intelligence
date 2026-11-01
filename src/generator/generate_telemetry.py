
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

from src.database.connection import SessionLocal
from src.database.models import Machine, Sensor


# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

SEED = 20261022
rng = random.Random(SEED)

START_DATE = datetime(2026, 8, 3, 0, 0, 0)

# Start with 2 days for testing.
# Later change to 90 for the full challenge.
DAYS = 90

READINGS_PER_HOUR = 1
FAILURE_COUNT = 120

OUTPUT_DIR = Path("data/raw")
GROUND_TRUTH_DIR = Path("data/ground_truth")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

TELEMETRY_FILE = OUTPUT_DIR / "telemetry.csv"
FAILURE_PLAN_FILE = GROUND_TRUTH_DIR / "failure_plan.csv"


# -----------------------------------------
# FAILURE SCHEDULE
# -----------------------------------------

def generate_failure_schedule(machines):

    schedule = []

    machine_ids = sorted(
        machine.machine_id for machine in machines
    )

    failure_types = [
        "OVERHEATING",
        "BEARING_WEAR",
        "MOTOR_DEGRADATION",
        "PRESSURE_FAILURE"
    ]

    for i in range(FAILURE_COUNT):

        machine_id = machine_ids[i % len(machine_ids)]

        failure_day = rng.randint(5, 89)

        failure_time = (
            START_DATE
            + timedelta(days=failure_day)
            + timedelta(hours=rng.randint(0, 23))
        )

        schedule.append({
            "failure_id": f"FAIL-{i + 1:06d}",
            "machine_id": machine_id,
            "failure_time": failure_time,
            "failure_type": rng.choice(failure_types),
            "severity": rng.choice(["MEDIUM", "HIGH", "CRITICAL"]),
            "downtime_hours": round(rng.uniform(2, 24), 2),
            "ground_truth_label": 1
        })

    return sorted(
        schedule,
        key=lambda x: x["failure_time"]
    )


def save_failure_schedule(schedule):

    with open(
        FAILURE_PLAN_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "failure_id",
            "machine_id",
            "failure_time",
            "failure_type",
            "severity",
            "downtime_hours",
            "ground_truth_label"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for event in schedule:
            row = event.copy()
            row["failure_time"] = row["failure_time"].isoformat()
            writer.writerow(row)


# -----------------------------------------
# MACHINE OPERATING STATE
# -----------------------------------------

def get_operating_state(timestamp):

    # Sunday-like scheduled shutdown every seventh day
    day_number = (timestamp - START_DATE).days

    if day_number % 7 == 6:
        return "SHUTDOWN"

    # Daily operating schedule
    if 7 <= timestamp.hour < 19:
        return "RUNNING"

    return "IDLE"


# -----------------------------------------
# SENSOR BASELINE VALUES
# -----------------------------------------

def generate_normal_value(sensor_type, state):

    if state == "SHUTDOWN":
        return {
            "TEMP": 25,
            "VIB": 0.0,
            "PRESS": 0.0,
            "CURRENT": 0.0,
            "HUMIDITY": 45
        }[sensor_type] + rng.uniform(-1, 1)

    if state == "IDLE":
        return {
            "TEMP": 35,
            "VIB": 0.5,
            "PRESS": 1.0,
            "CURRENT": 2.0,
            "HUMIDITY": 45
        }[sensor_type] + rng.uniform(-2, 2)

    return {
        "TEMP": rng.gauss(65, 4),
        "VIB": rng.gauss(2.5, 0.5),
        "PRESS": rng.gauss(6.0, 0.6),
        "CURRENT": rng.gauss(20, 3),
        "HUMIDITY": rng.gauss(45, 5)
    }[sensor_type]


# -----------------------------------------
# PRE-FAILURE DETERIORATION
# -----------------------------------------

def apply_failure_pattern(
    value,
    sensor_type,
    timestamp,
    machine_failures
):

    for failure in machine_failures:

        hours_remaining = (
            failure["failure_time"] - timestamp
        ).total_seconds() / 3600

        # Inject deterioration during the final 72 hours
        if 0 < hours_remaining <= 72:

            deterioration = (72 - hours_remaining) / 72

            if sensor_type == "TEMP":
                value += 25 * deterioration

            elif sensor_type == "VIB":
                value += 5 * deterioration

            elif sensor_type == "PRESS":
                value += 2 * deterioration

    return value


# -----------------------------------------
# GENERATE TELEMETRY
# -----------------------------------------

def generate_telemetry():

    db = SessionLocal()

    try:

        machines = (
            db.query(Machine)
            .order_by(Machine.machine_id)
            .all()
        )

        sensors = (
            db.query(Sensor)
            .order_by(Sensor.sensor_id)
            .all()
        )

        if len(machines) != 100 or len(sensors) != 500:
            raise ValueError(
                "Expected 100 machines and 500 sensors. "
                "Run master data generation first."
            )

        # Create a deterministic failure schedule
        failure_schedule = generate_failure_schedule(machines)

        save_failure_schedule(failure_schedule)

        # Group planned failures by machine
        failures_by_machine = {}

        for failure in failure_schedule:

            failures_by_machine.setdefault(
                failure["machine_id"], []
            ).append(failure)

        total_readings = 0
        sensor_fault_count = 0

        with open(
            TELEMETRY_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            fieldnames = [
                "telemetry_id",
                "sensor_id",
                "machine_id",
                "timestamp",
                "value",
                "unit",
                "operating_state",
                "issue_type"
            ]

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            end_date = START_DATE + timedelta(days=DAYS)

            timestamp = START_DATE

            while timestamp < end_date:

                for sensor in sensors:

                    machine_id = sensor.machine_id
                    sensor_type = sensor.sensor_type

                    state = get_operating_state(timestamp)

                    value = generate_normal_value(
                        sensor_type,
                        state
                    )

                    # Add gradual pre-failure deterioration
                    value = apply_failure_pattern(
                        value,
                        sensor_type,
                        timestamp,
                        failures_by_machine.get(
                            machine_id, []
                        )
                    )

                    # Sensor faults are explicitly identified
                    issue_type = None

                    if rng.random() < 0.002:

                        issue_type = "SENSOR_FAULT"
                        sensor_fault_count += 1

                    # Clamp baseline values to valid ranges.
                    # Deliberate invalid readings will be
                    # introduced separately in the anomaly stage.
                    value = max(sensor.min_value, value)
                    value = min(sensor.max_value, value)

                    telemetry_id = (
                        f"TEL-{sensor.sensor_id}-"
                        f"{timestamp.strftime('%Y%m%dT%H%M%S')}"
                    )

                    writer.writerow({
                        "telemetry_id": telemetry_id,
                        "sensor_id": sensor.sensor_id,
                        "machine_id": machine_id,
                        "timestamp": timestamp.isoformat(),
                        "value": round(value, 3),
                        "unit": sensor.unit,
                        "operating_state": state,
                        "issue_type": issue_type
                    })

                    total_readings += 1

                timestamp += timedelta(
                    hours=1 / READINGS_PER_HOUR
                )

        print("\nTELEMETRY GENERATION COMPLETE")
        print("--------------------------------")
        print("Machines:", len(machines))
        print("Sensors:", len(sensors))
        print("Days generated:", DAYS)
        print("Telemetry readings:", total_readings)
        print("Sensor fault readings:", sensor_fault_count)
        print("Planned failures:", len(failure_schedule))
        print("Telemetry file:", TELEMETRY_FILE)
        print("Failure plan:", FAILURE_PLAN_FILE)

    finally:
        db.close()


if __name__ == "__main__":
    generate_telemetry()
