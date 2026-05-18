import os
import time
import joblib
import pandas as pd
import numpy as np
import warnings

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

def get_regression_model(name, **kwargs):
    if name == "Random Forest Regressor": return RandomForestRegressor(random_state=42, n_jobs=-1, **kwargs)
    elif name == "Gradient Boosting Regressor": return GradientBoostingRegressor(random_state=42, **kwargs)
    elif name == "Ridge Regression": return Ridge(**kwargs)
    elif name == "Lasso Regression": return Lasso(random_state=42, **kwargs)
    elif name == "Support Vector Regressor": return SVR(**kwargs)

def prep_regression_data():
    X_train = pd.read_csv('X_train_regression.csv')
    X_test = pd.read_csv('X_test_regression.csv')
    y_train_log = pd.read_csv('y_train_regression.csv').squeeze("columns")
    y_test_log = pd.read_csv('y_test_regression.csv').squeeze("columns")

    leaky_cols = ['SteamSpyOwners', 'SteamSpyOwnersVariance', 'PriceInitial', 'ReleaseYear', 'date_extraction_failure']
    X_train = X_train.drop(columns=leaky_cols, errors='ignore')
    X_test = X_test.drop(columns=leaky_cols, errors='ignore')

    return X_train, y_train_log, X_test, y_test_log

def train_and_eval_regression(model, X_train, y_train_log, X_test, y_test_log):
    start_train = time.time()
    model.fit(X_train, y_train_log)
    train_time = time.time() - start_train

    start_test = time.time()
    y_pred_test_log = model.predict(X_test)
    test_time = time.time() - start_test

    y_test_actual = np.expm1(y_test_log)
    y_pred_test_actual = np.expm1(y_pred_test_log)

    mse_test = mean_squared_error(y_test_actual, y_pred_test_actual)
    r2_test = r2_score(y_test_actual, y_pred_test_actual)
    mae_test = mean_absolute_error(y_test_actual, y_pred_test_actual)

    return model, train_time, test_time, mse_test, r2_test, mae_test, y_test_actual, y_pred_test_actual