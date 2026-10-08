import joblib
from pathlib import Path


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "energy_forecasting_model.joblib"
)

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["features"]