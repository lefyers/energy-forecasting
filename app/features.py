import numpy as np
import pandas as pd


FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    y = df["AEP_MW"]

    X = pd.DataFrame(index=df.index)

    X["hour"] = df.index.hour
    X["day_of_week"] = df.index.dayofweek
    X["month"] = df.index.month
    X["day_of_year"] = df.index.dayofyear

    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)

    X["dow_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
    X["dow_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)

    X["lag_1"] = y.shift(1)
    X["lag_24"] = y.shift(24)
    X["lag_168"] = y.shift(168)

    X["rolling_mean_24"] = y.shift(1).rolling(24).mean()
    X["rolling_mean_168"] = y.shift(1).rolling(168).mean()

    X["lag_2"] = y.shift(2)
    X["lag_3"] = y.shift(3)
    X["lag_6"] = y.shift(6)
    X["lag_12"] = y.shift(12)

    return X[FEATURES]