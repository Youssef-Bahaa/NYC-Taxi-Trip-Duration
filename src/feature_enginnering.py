#feature_enginnering.py

import pandas as pd
import numpy as np


def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r


def manhattan_distance_vectorized(lat1, lon1, lat2, lon2):
    lat_dist = haversine_distance_vectorized(lat1, lon1, lat2, lon1)
    lon_dist = haversine_distance_vectorized(lat1, lon1, lat1, lon2)
    return lat_dist + lon_dist


def bearing_vectorized(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y, x)
    bearing = np.degrees(bearing)
    return (bearing + 360) % 360


def add_geo_features(df):
    df["haversine_distance"] = haversine_distance_vectorized(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )
    df["manhattan_distance"] = manhattan_distance_vectorized(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )
    df["bearing"] = bearing_vectorized(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )
    return df



def day_period(hour):
    if 0 <= hour < 6:
        return 'Night'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'


def month_to_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


def time_extraction(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['month'] = df['pickup_datetime'].dt.month
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day'] = df['pickup_datetime'].dt.day
    df['quarter'] = df['pickup_datetime'].dt.quarter
    df['day_name'] = df['pickup_datetime'].dt.day_name()
    df['day_period'] = df['hour'].apply(day_period)
    df['season'] = df['month'].apply(month_to_season)
    df['weekend'] = np.where(df['pickup_datetime'].dt.weekday >= 5, 1, 0)

    # Rush hour detection
    hour_counts = df['pickup_datetime'].dt.hour.value_counts()
    mean = hour_counts.mean()
    rush_hours = hour_counts[hour_counts >= mean].index.tolist()
    df['is_rush_hour'] = np.where(df['hour'].isin(rush_hours), 1, 0)

    # log trip duration
    if 'trip_duration' in df.columns:
        df['log_trip_duration'] = np.log1p(df['trip_duration'])

    return df


def add_speed_features(df):
    df["speed"] = df["haversine_distance"] / (df["trip_duration"] / 3600)
    return df



def add_estimated_time(df):
    if "haversine_distance" in df.columns:
        df["estimated_time"] = np.log1p((df["haversine_distance"] / df['speed'].mean()) * 3600)
    return df
