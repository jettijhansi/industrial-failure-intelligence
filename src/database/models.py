
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    ForeignKey,
    Boolean,
    Table,
    UniqueConstraint,
    CheckConstraint,
    Index
)

from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


# Many-to-many bridge: Work orders and replacement parts
work_order_parts = Table(
    "work_order_parts",
    Base.metadata,
    Column(
        "work_order_id",
        String,
        ForeignKey("maintenance_work_orders.work_order_id"),
        primary_key=True
    ),
    Column(
        "part_id",
        String,
        ForeignKey("parts.part_id"),
        primary_key=True
    ),
    Column("quantity", Integer, nullable=False)
)


# Many-to-many bridge: Work orders and technicians
work_order_technicians = Table(
    "work_order_technicians",
    Base.metadata,
    Column(
        "work_order_id",
        String,
        ForeignKey("maintenance_work_orders.work_order_id"),
        primary_key=True
    ),
    Column(
        "technician_id",
        String,
        ForeignKey("technicians.technician_id"),
        primary_key=True
    )
)


class Plant(Base):
    __tablename__ = "plants"

    plant_id = Column(String, primary_key=True)
    plant_name = Column(String, nullable=False, unique=True)
    location = Column(String, nullable=False)
    status = Column(String, nullable=False, default="ACTIVE")

    production_lines = relationship(
        "ProductionLine",
        back_populates="plant"
    )


class ProductionLine(Base):
    __tablename__ = "production_lines"

    line_id = Column(String, primary_key=True)
    plant_id = Column(
        String,
        ForeignKey("plants.plant_id"),
        nullable=False
    )
    line_name = Column(String, nullable=False)
    status = Column(String, nullable=False, default="ACTIVE")

    plant = relationship(
        "Plant",
        back_populates="production_lines"
    )

    machines = relationship(
        "Machine",
        back_populates="production_line"
    )


class Machine(Base):
    __tablename__ = "machines"

    machine_id = Column(String, primary_key=True)
    line_id = Column(
        String,
        ForeignKey("production_lines.line_id"),
        nullable=False
    )

    machine_name = Column(String, nullable=False)
    machine_type = Column(String, nullable=False)
    installation_date = Column(DateTime, nullable=False)
    operating_status = Column(
        String,
        nullable=False,
        default="RUNNING"
    )

    production_line = relationship(
        "ProductionLine",
        back_populates="machines"
    )

    sensors = relationship("Sensor", back_populates="machine")

    work_orders = relationship(
        "MaintenanceWorkOrder",
        back_populates="machine"
    )

    failures = relationship("FailureEvent", back_populates="machine")


class Sensor(Base):
    __tablename__ = "sensors"

    sensor_id = Column(String, primary_key=True)
    machine_id = Column(
        String,
        ForeignKey("machines.machine_id"),
        nullable=False
    )

    sensor_type = Column(String, nullable=False)
    unit = Column(String, nullable=False)
    min_value = Column(Float, nullable=False)
    max_value = Column(Float, nullable=False)
    status = Column(String, nullable=False, default="ACTIVE")

    machine = relationship("Machine", back_populates="sensors")

    telemetry_events = relationship(
        "TelemetryEvent",
        back_populates="sensor"
    )

    __table_args__ = (
        CheckConstraint(
            "min_value < max_value",
            name="check_sensor_range"
        ),
    )


class TelemetryEvent(Base):
    __tablename__ = "telemetry_events"

    telemetry_id = Column(String, primary_key=True)
    sensor_id = Column(
        String,
        ForeignKey("sensors.sensor_id"),
        nullable=False
    )

    timestamp = Column(DateTime, nullable=False, index=True)
    value = Column(Float, nullable=False)
    unit = Column(String, nullable=False)
    operating_state = Column(String, nullable=False)

    issue_type = Column(String, nullable=True)

    sensor = relationship(
        "Sensor",
        back_populates="telemetry_events"
    )

    __table_args__ = (
        UniqueConstraint(
            "sensor_id",
            "timestamp",
            name="unique_sensor_timestamp"
        ),
        Index("idx_telemetry_sensor_time", "sensor_id", "timestamp"),
    )


class OperatingShift(Base):
    __tablename__ = "operating_shifts"

    shift_id = Column(String, primary_key=True)
    machine_id = Column(
        String,
        ForeignKey("machines.machine_id"),
        nullable=False
    )

    shift_name = Column(String, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    operating_state = Column(String, nullable=False)

    __table_args__ = (
        CheckConstraint(
            "end_time > start_time",
            name="check_shift_dates"
        ),
    )


class MaintenanceWorkOrder(Base):
    __tablename__ = "maintenance_work_orders"

    work_order_id = Column(String, primary_key=True)
    machine_id = Column(
        String,
        ForeignKey("machines.machine_id"),
        nullable=False
    )

    created_at = Column(DateTime, nullable=False)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)

    priority = Column(String, nullable=False)
    status = Column(String, nullable=False)

    description = Column(String, nullable=False)

    machine = relationship(
        "Machine",
        back_populates="work_orders"
    )

    parts = relationship(
        "Part",
        secondary=work_order_parts
    )

    technicians = relationship(
        "Technician",
        secondary=work_order_technicians
    )


class FailureEvent(Base):
    __tablename__ = "failure_events"

    failure_id = Column(String, primary_key=True)
    machine_id = Column(
        String,
        ForeignKey("machines.machine_id"),
        nullable=False
    )

    failure_time = Column(DateTime, nullable=False)
    failure_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)

    downtime_hours = Column(Float, nullable=False)
    ground_truth_label = Column(Integer, nullable=False)

    machine = relationship(
        "Machine",
        back_populates="failures"
    )


class Part(Base):
    __tablename__ = "parts"

    part_id = Column(String, primary_key=True)
    part_name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    unit_cost = Column(Float, nullable=False)
    stock_quantity = Column(Integer, nullable=False)


class Technician(Base):
    __tablename__ = "technicians"

    technician_id = Column(String, primary_key=True)
    technician_name = Column(String, nullable=False)
    specialization = Column(String, nullable=False)
    availability_status = Column(
        String,
        nullable=False,
        default="AVAILABLE"
    )
