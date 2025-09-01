# encode.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def encode(df):
    numeric_features = ['estimated_time','hour','month','day','quarter', 'passenger_count','bearing']
    categorical_features = ['day_name','day_period','vendor_id','store_and_fwd_flag','season']
    binary = ['is_rush_hour','weekend',]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
            ('bin', 'passthrough', binary)
        ]
    )

    return preprocessor