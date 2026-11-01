
import random
from datetime import datetime, timedelta

from src.database.connection import SessionLocal, create_database
from src.database.models import (
    Plant,
    ProductionLine,
    Machine,
    Sensor,
    Part,
    Technician
)

# Fixed seed for reproducibility
SEED = 20261022
rng = random.Random(SEED)

# Master data configuration
PLANT_COUNT = 3
LINE_COUNT = 12
MACHINE_COUNT = 100
SENSORS_PER_MACHINE = 5

# Sensor definitions: type, unit, minimum, maximum
SENSOR_TYPES = {
    "TEMP": ("Celsius", 10.0, 120.0),
    "VIB": ("mm/s", 0.0, 20.0),
    "PRESS": ("bar", 0.0, 15.0),
    "CURRENT": ("Ampere", 0.0, 50.0),
    "HUMIDITY": ("Percent", 0.0, 100.0),
}


def generate_plants():
    plants = []

    for i in range(1, PLANT_COUNT + 1):
        plant_id = f"PLT-{i:03d}"

        plants.append(
            Plant(
                plant_id=plant_id,
                plant_name=f"Manufacturing Plant {i}",
                location=f"Industrial Zone {i}",
                status="ACTIVE"
            )
        )

    return plants


def generate_lines(plants):
    lines = []

    for plant_index, plant in enumerate(plants, start=1):
        for line_number in range(1, 5):

            line_id = f"LINE-{plant.plant_id}-{line_number:03d}"

            lines.append(
                ProductionLine(
                    line_id=line_id,
                    plant_id=plant.plant_id,
                    line_name=f"Production Line {line_number}",
                    status="ACTIVE"
                )
            )

    return lines


def generate_machines(lines):
    machines = []
    machine_counter = 0

    # 100 machines distributed across 12 lines:
    # First 4 lines receive 9 machines each.
    # Remaining 8 lines receive 8 machines each.

    for line_index, line in enumerate(lines):

        machines_in_line = 9 if line_index < 4 else 8

        for machine_number in range(1, machines_in_line + 1):

            machine_counter += 1

            machine_id = (
                f"MCH-{line.line_id}-{machine_number:03d}"
            )

            installation_date = datetime(2024, 1, 1) + timedelta(
                days=rng.randint(0, 365)
            )

            machines.append(
                Machine(
                    machine_id=machine_id,
                    line_id=line.line_id,
                    machine_name=f"Industrial Machine {machine_counter:03d}",
                    machine_type=rng.choice([
                        "MOTOR",
                        "PUMP",
                        "COMPRESSOR",
                        "CONVEYOR",
                        "PRESS"
                    ]),
                    installation_date=installation_date,
                    operating_status="RUNNING"
                )
            )

    assert machine_counter == MACHINE_COUNT

    return machines


def generate_sensors(machines):
    sensors = []

    for machine in machines:

        for sensor_type, details in SENSOR_TYPES.items():

            unit, min_value, max_value = details

            sensor_id = (
                f"SNS-{machine.machine_id}-{sensor_type}-01"
            )

            sensors.append(
                Sensor(
                    sensor_id=sensor_id,
                    machine_id=machine.machine_id,
                    sensor_type=sensor_type,
                    unit=unit,
                    min_value=min_value,
                    max_value=max_value,
                    status="ACTIVE"
                )
            )

    return sensors


def generate_parts():
    parts = []

    for i in range(1, 21):

        parts.append(
            Part(
                part_id=f"PRT-{i:03d}",
                part_name=f"Replacement Part {i:03d}",
                category=["BEARING", "FILTER", "MOTOR", "BELT"][
                    (i - 1) % 4
                ],
                unit_cost=round(rng.uniform(100, 5000), 2),
                stock_quantity=rng.randint(5, 100)
            )
        )

    return parts


def generate_technicians():
    technicians = []

    specializations = [
        "MECHANICAL",
        "ELECTRICAL",
        "INSTRUMENTATION",
        "HYDRAULICS",
        "GENERAL"
    ]

    for i in range(1, 31):

        technicians.append(
            Technician(
                technician_id=f"TECH-{i:03d}",
                technician_name=f"Technician {i:03d}",
                specialization=specializations[(i - 1) % 5],
                availability_status="AVAILABLE"
            )
        )

    return technicians


def seed_master_data():

    create_database()

    db = SessionLocal()

    try:
        expected_counts = {
            Plant: PLANT_COUNT,
            ProductionLine: LINE_COUNT,
            Machine: MACHINE_COUNT,
            Sensor: MACHINE_COUNT * SENSORS_PER_MACHINE,
            Part: 20,
            Technician: 30
        }

        existing_counts = {
            model: db.query(model).count()
            for model in expected_counts
        }

        if any(existing_counts.values()):

            if existing_counts == expected_counts:
                print("Master data already exists. No changes made.")
                return

            raise RuntimeError(
                "Database contains partial master data. "
                "Do not seed again until it is inspected."
            )

        # Generate parent records first
        plants = generate_plants()
        db.add_all(plants)
        db.flush()

        # Generate dependent records
        lines = generate_lines(plants)
        db.add_all(lines)
        db.flush()

        machines = generate_machines(lines)
        db.add_all(machines)
        db.flush()

        sensors = generate_sensors(machines)
        db.add_all(sensors)

        parts = generate_parts()
        technicians = generate_technicians()

        db.add_all(parts)
        db.add_all(technicians)

        db.commit()

        print("\nMASTER DATA GENERATED SUCCESSFULLY")
        print("----------------------------------")
        print("Plants:", len(plants))
        print("Production Lines:", len(lines))
        print("Machines:", len(machines))
        print("Sensors:", len(sensors))
        print("Parts:", len(parts))
        print("Technicians:", len(technicians))

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    seed_master_data()
