#models.py
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

def get_model():
    models = {
        'LinearRegression':LinearRegression(),
        'Ridge':Ridge(),
        'Lasso':Lasso(),
        'XGBoost':xgb.XGBRegressor(    n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42)

    }

    return models


def evaluate_model(model,X_train,X_val,X_test,y_train,y_val,y_test):

    model.fit(X_train,y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    r2 = {
    'train_r2_score' : r2_score(y_train,y_pred_train),
    'val_r2_score' : r2_score(y_val,y_pred_val),
    'test_r2_score' : r2_score(y_test,y_pred_test)
    }

    return r2



