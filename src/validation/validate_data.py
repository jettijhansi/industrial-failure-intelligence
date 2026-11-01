
from sqlalchemy import func, select

from src.database.connection import SessionLocal
from src.database.models import (
    Plant,
    ProductionLine,
    Machine,
    Sensor,
    OperatingShift,
    FailureEvent,
    MaintenanceWorkOrder,
    Part,
    Technician,
)


EXPECTED_COUNTS = {
    "plants": (Plant, 3),
    "production_lines": (ProductionLine, 12),
    "machines": (Machine, 100),
    "sensors": (Sensor, 500),
    "operating_shifts": (OperatingShift, 27000),
    "failure_events": (FailureEvent, 120),
    "maintenance_work_orders": (MaintenanceWorkOrder, 120),
}


def validate_data():
    db = SessionLocal()
    errors = []

    try:
        print("\nINDUSTRIAL DATA VALIDATION")
        print("=" * 40)

        for name, (model, expected) in EXPECTED_COUNTS.items():
            actual = db.scalar(
                select(func.count()).select_from(model)
            )

            status = "PASS" if actual == expected else "FAIL"

            print(
                f"{name:25} {actual:8} "
                f"(expected {expected}) [{status}]"
            )

            if actual != expected:
                errors.append(
                    f"{name}: expected {expected}, found {actual}"
                )

        # Verify every sensor belongs to an existing machine
        orphan_sensors = db.query(Sensor).filter(
            ~Sensor.machine_id.in_(
                select(Machine.machine_id)
            )
        ).count()

        print(f"\nOrphan sensors: {orphan_sensors}")

        if orphan_sensors:
            errors.append("Some sensors reference missing machines.")

        # Verify every machine belongs to an existing production line
        orphan_machines = db.query(Machine).filter(
            ~Machine.line_id.in_(
                select(ProductionLine.line_id)
            )
        ).count()

        print(f"Orphan machines: {orphan_machines}")

        if orphan_machines:
            errors.append("Some machines reference missing production lines.")

        # Verify valid shift time ranges
        invalid_shifts = db.query(OperatingShift).filter(
            OperatingShift.end_time <= OperatingShift.start_time
        ).count()

        print(f"Invalid shifts: {invalid_shifts}")

        if invalid_shifts:
            errors.append("Some operating shifts have invalid time ranges.")

        # Verify maintenance time ranges
        invalid_work_orders = db.query(MaintenanceWorkOrder).filter(
            MaintenanceWorkOrder.end_time <= MaintenanceWorkOrder.start_time
        ).count()

        print(f"Invalid maintenance work orders: {invalid_work_orders}")

        if invalid_work_orders:
            errors.append("Some maintenance work orders have invalid time ranges.")

        print("\n" + "=" * 40)

        if errors:
            print("VALIDATION FAILED")
            for error in errors:
                print("-", error)
        else:
            print("ALL VALIDATION CHECKS PASSED")

    finally:
        db.close()


if __name__ == "__main__":
    validate_data()
