
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from config import GROUND_TRUTH_DIR
from src.database.connection import SessionLocal
from src.database.models import (
    Machine,
    OperatingShift,
    MaintenanceWorkOrder,
    FailureEvent,
    Part,
    Technician,
    work_order_parts,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SEED = 20261022
START_DATE = datetime(2026, 8, 3, 0, 0, 0)
DAYS = 90

FAILURE_FILE = GROUND_TRUTH_DIR / "failure_plan.csv"

SHIFTS_PER_DAY = 3


# --------------------------------------------------
# LOAD FAILURE PLAN
# --------------------------------------------------

def load_failure_plan():
    if not FAILURE_FILE.exists():
        raise FileNotFoundError(
            f"Failure plan not found: {FAILURE_FILE}"
        )

    failures = []

    with open(FAILURE_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            failure_time = datetime.fromisoformat(
                row["failure_time"]
            )

            downtime_hours = float(
                row.get("downtime_hours")
                or row.get("downtime")
                or 8
            )

            failures.append({
                "machine_id": row["machine_id"],
                "failure_time": failure_time,
                "failure_type": row["failure_type"],
                "severity": row["severity"],
                "downtime_hours": downtime_hours,
                "ground_truth_label": (
                    row.get("ground_truth_label") or "FAILURE"
                ),
            })

    if not failures:
        raise ValueError("The failure plan contains no records.")

    return failures


# --------------------------------------------------
# GENERATE OPERATING SHIFTS
# --------------------------------------------------

def generate_operating_shifts(db, machines):
    shifts = []

    shift_definitions = [
        ("NIGHT", 0, 7),
        ("DAY", 7, 19),
        ("EVENING", 19, 24),
    ]

    for machine in machines:

        for day_number in range(DAYS):
            current_date = START_DATE + timedelta(days=day_number)

            # Weekly shutdown day
            shutdown_day = day_number % 7 == 6

            for shift_name, start_hour, end_hour in shift_definitions:

                start_time = current_date + timedelta(
                    hours=start_hour
                )

                end_time = current_date + timedelta(
                    hours=end_hour
                )

                if shutdown_day:
                    operating_state = "IDLE"
                elif shift_name == "DAY":
                    operating_state = "RUNNING"
                else:
                    operating_state = "IDLE"

                shift_id = (
                    f"SHF-{machine.machine_id}-"
                    f"{current_date.strftime('%Y%m%d')}-"
                    f"{shift_name}"
                )

                shifts.append(
                    OperatingShift(
                        shift_id=shift_id,
                        machine_id=machine.machine_id,
                        shift_name=shift_name,
                        start_time=start_time,
                        end_time=end_time,
                        operating_state=operating_state,
                    )
                )

    db.add_all(shifts)
    db.flush()

    return len(shifts)


# --------------------------------------------------
# GENERATE FAILURE EVENTS AND MAINTENANCE WORK ORDERS
# --------------------------------------------------

def generate_failure_and_work_orders(
    db,
    failure_plan,
    machines,
    parts,
    technicians,
):
    rng = random.Random(SEED)

    machine_ids = {
        machine.machine_id for machine in machines
    }

    available_parts = [
        part for part in parts
        if part.stock_quantity > 0
    ]

    if not available_parts:
        raise ValueError("No parts have available stock.")

    if not technicians:
        raise ValueError("No technicians found in the database.")

    failure_records = []
    work_orders = []
    part_assignments = []

    end_of_period = START_DATE + timedelta(days=DAYS)

    for index, failure in enumerate(failure_plan, start=1):

        machine_id = failure["machine_id"]
        failure_time = failure["failure_time"]
        downtime_hours = failure["downtime_hours"]

        if machine_id not in machine_ids:
            raise ValueError(
                f"Unknown machine in failure plan: {machine_id}"
            )

        if not START_DATE <= failure_time < end_of_period:
            raise ValueError(
                f"Failure time outside the 90-day period: {failure_time}"
            )

        if downtime_hours <= 0:
            raise ValueError(
                f"Invalid downtime for failure {index}"
            )

        failure_id = f"FAIL-2026-{index:04d}"
        work_order_id = f"WO-2026-{index:06d}"

        repair_end_time = failure_time + timedelta(
            hours=downtime_hours
        )

        # Create failure record
        failure_record = FailureEvent(
            failure_id=failure_id,
            machine_id=machine_id,
            failure_time=failure_time,
            failure_type=failure["failure_type"],
            severity=failure["severity"],
            downtime_hours=downtime_hours,
            ground_truth_label=failure["ground_truth_label"],
        )

        failure_records.append(failure_record)

        # Create maintenance work order
        work_order = MaintenanceWorkOrder(
            work_order_id=work_order_id,
            machine_id=machine_id,
            created_at=failure_time - timedelta(hours=24),
            start_time=failure_time,
            end_time=repair_end_time,
            priority=failure["severity"],
            status="COMPLETED",
            description=(
                f"Repair for {failure['failure_type']} failure "
                f"on machine {machine_id}"
            ),
        )

        # Assign a technician through the ORM relationship
        technician = rng.choice(technicians)
        work_order.technicians.append(technician)

        work_orders.append(work_order)

        # Select one part for this maintenance job
        part = rng.choice(available_parts)

        part_assignments.append({
            "work_order_id": work_order_id,
            "part_id": part.part_id,
            "quantity": 1,
        })

        # Update available stock in this in-memory run
        part.stock_quantity -= 1

    # Insert failure events and work orders
    db.add_all(failure_records)
    db.add_all(work_orders)

    # Ensure work-order rows exist before inserting associations
    db.flush()

    # Insert parts with an explicit quantity
    if part_assignments:
        db.execute(
            work_order_parts.insert(),
            part_assignments,
        )

    return len(failure_records), len(work_orders)


# --------------------------------------------------
# MAIN GENERATOR
# --------------------------------------------------

def generate_maintenance_data():

    db = SessionLocal()

    try:
        print("\nGENERATING MAINTENANCE DATA")
        print("--------------------------------")

        # Prevent duplicate generation
        existing_shifts = db.scalar(
            select(OperatingShift.shift_id).limit(1)
        )

        existing_failures = db.scalar(
            select(FailureEvent.failure_id).limit(1)
        )

        existing_work_orders = db.scalar(
            select(MaintenanceWorkOrder.work_order_id).limit(1)
        )

        if (
            existing_shifts is not None
            or existing_failures is not None
            or existing_work_orders is not None
        ):
            raise RuntimeError(
                "Maintenance data already exists. "
                "Not generating duplicate records."
            )

        machines = db.scalars(
            select(Machine).order_by(Machine.machine_id)
        ).all()

        parts = db.scalars(
            select(Part).order_by(Part.part_id)
        ).all()

        technicians = db.scalars(
            select(Technician).order_by(Technician.technician_id)
        ).all()

        if not machines:
            raise ValueError("No machines found. Generate master data first.")

        if len(machines) != 100:
            raise ValueError(
                f"Expected 100 machines, found {len(machines)}."
            )

        failure_plan = load_failure_plan()

        print(f"Machines found: {len(machines)}")
        print(f"Failure plan records: {len(failure_plan)}")
        print("Generating operating shifts...")

        shift_count = generate_operating_shifts(
            db,
            machines,
        )

        print("Generating failure and maintenance records...")

        failure_count, work_order_count = (
            generate_failure_and_work_orders(
                db,
                failure_plan,
                machines,
                parts,
                technicians,
            )
        )

        db.commit()

        print("\nMAINTENANCE GENERATION COMPLETE")
        print("--------------------------------")
        print(f"Operating shifts: {shift_count}")
        print(f"Failure events: {failure_count}")
        print(f"Maintenance work orders: {work_order_count}")
        print("Database changes committed successfully.")

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    generate_maintenance_data()
