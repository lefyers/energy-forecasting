import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app

from app.features import create_features
from app.model import model, features
from app.schemas import PredictionRequest, PredictionResponse


app = FastAPI(
    title="Energy Forecasting API",
    version="1.0.0",
)


REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": type(model).__name__,
        "features": len(features),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    with PREDICTION_LATENCY.time():
        REQUEST_COUNT.inc()

        try:
            forecast_timestamp = pd.Timestamp(request.timestamp)

            history_timestamps = pd.date_range(
                end=forecast_timestamp - pd.Timedelta(hours=1),
                periods=len(request.values),
                freq="h",
            )

            history = pd.DataFrame(
                {"AEP_MW": request.values},
                index=history_timestamps,
            )

            target_row = pd.DataFrame(
                {"AEP_MW": [float("nan")]},
                index=[forecast_timestamp],
            )

            df = pd.concat([history, target_row])

            X = create_features(df)

            prediction_features = X.loc[[forecast_timestamp]]
            prediction_features = prediction_features[features]

            if prediction_features.isna().any().any():
                raise ValueError("Not enough valid history to create features")

            prediction = model.predict(prediction_features)[0]

            return PredictionResponse(
                timestamp=request.timestamp,
                forecast=float(prediction),
            )

        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc


metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)