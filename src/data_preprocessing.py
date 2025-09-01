#data_preprocessing.py
import pandas as pd
import numpy as np
import feature_enginnering as fe


def remove_outliers(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_columns:
        Q1 = df[col].quantile(0.05)
        Q3 = df[col].quantile(0.95)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        df = df[(df['haversine_distance'] > 0) & (df['haversine_distance'] < 50)]
        df = df[df['speed'] <= 65]
        df = df[df['trip_duration'] <= 7200]

    return df


def drop_columns(df):
    drop_cols = ["id", "pickup_datetime"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


def dropna(df):
    return df.dropna()


def preprocess(df):
    df = dropna(df)
    df = fe.time_extraction(df)
    df = fe.add_geo_features(df)
    df = fe.add_speed_features(df)
    df = fe.add_estimated_time(df)
    df = drop_columns(df)
    return df