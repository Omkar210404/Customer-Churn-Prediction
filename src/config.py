"""Project-wide constants: paths, dataset source, column schema."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_CSV = RAW_DIR / "Telco-Customer-Churn.csv"
MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
# SHA-256 of the file as downloaded on 2026-10-09.
DATASET_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

ID_COL = "customerID"
TARGET = "Churn"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Allowed values for each categorical input column (after cleaning).
CATEGORICAL_VALUES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}
CATEGORICAL_COLS = list(CATEGORICAL_VALUES)

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
