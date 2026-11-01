# Industrial Equipment Failure Intelligence

An end-to-end industrial equipment failure prediction system built using synthetic industrial telemetry, maintenance history, feature engineering, and machine learning.

## Project Objective

Predict the risk of industrial equipment failure within the next 24 hours using sensor telemetry and historical operating information.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- SQLAlchemy and SQLite
- FastAPI
- Streamlit
- Git and GitHub

## Project Structure

```text
industrial-failure-intelligence/
├── data/
│   ├── raw/
│   ├── processed/
│   └── ground_truth/
├── src/
│   ├── generator/
│   ├── database/
│   ├── analytics/
│   ├── api/
│   └── validation/
├── tests/
├── dashboard/
├── docs/
├── config.py
├── init_db.py
├── requirements.txt
└── README.md
```

## Dataset

The project generates synthetic industrial data for:

- 3 plants
- 12 production lines
- 100 machines
- 500 sensors
- 90 days of telemetry
- 120 simulated failure events

The telemetry dataset contains approximately 1.08 million readings.

Large generated CSV files are excluded from GitHub because of repository file-size limits. Generate them locally using the scripts in `src/generator/`.

## Reproducibility

The random seed is fixed at `20261022`.

Run the generators in the following order:

```powershell
py init_db.py
py -m src.generator.generate_data
py -m src.generator.generate_telemetry
py -m src.generator.generate_maintenance
```

Run validation:

```powershell
py -m src.validation.validate_data
```

## Feature Engineering and Model Training

```powershell
py -m src.analytics.build_features
py -m src.analytics.train_model
```

The model predicts whether a machine may experience a failure within the next 24 hours.

## API

Start the FastAPI server:

```powershell
py -m uvicorn src.api.main:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

## Limitations

This project uses synthetic data. Its model performance should not be interpreted as evidence of real-world industrial failure prediction accuracy. Real deployment would require validated plant