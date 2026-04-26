import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from analysis import feature_importance, show_top_correlations
import warnings

warnings.filterwarnings("ignore")

def main():
    X_train = pd.read_csv('X_train_final.csv')
    X_test = pd.read_csv('X_test_final.csv')
    y_train_log = pd.read_csv('y_train_final.csv').squeeze("columns")
    y_test_log = pd.read_csv('y_test_final.csv').squeeze("columns")

    leaky_cols = [
        'SteamSpyOwners',
        'SteamSpyOwnersVariance',
        'PriceInitial',
        'ReleaseYear',
        'date_extraction_failure'
    ]
    X_train = X_train.drop(columns=leaky_cols, errors='ignore')
    X_test = X_test.drop(columns=leaky_cols, errors='ignore')

    show_top_correlations(X_train,0.9)

    print(f"Number of training samples: {X_train.shape[0]}")
    print(f"Number of testing samples:  {X_test.shape[0]}")

    models = {
        "Ridge Regression": RidgeCV(alphas=np.logspace(-2, 4, 50), scoring='neg_mean_absolute_error'),
        "Lasso Regression": Lasso(random_state=42),
        "Support Vector Regressor": SVR(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    param_grids = {
        "Lasso Regression": {
            'alpha': [0.01, 0.1, 1.0]
        },
        "Support Vector Regressor": {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1.0, 10.0]
        },
        "Random Forest": {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_leaf': [1, 2, 4]
        },
        "Gradient Boosting": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
    }

    trained_models = perform_grid_search(models=models, param_grids=param_grids, X_train=X_train, y_train=y_train_log, X_test=X_test, y_test=y_test_log)
    feature_names = X_train.columns

    if "Ridge Regression" in trained_models:
        print("Plotting Ridge Regression Coefficients...")
        feature_importance(trained_models["Ridge Regression"], feature_names, "Ridge Regression")


def perform_grid_search(models, param_grids, X_train, y_train, X_test, y_test):

    print('Begin Model Training')

    best_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}")

        if name in param_grids:
            grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"Selected Parameters: {grid_search.best_params_}")
        else:
            # Exception for the Ridge Model as it uses RidgeCV instead of GridSearchCV
            print("(Using built-in Cross Validation)")
            model.fit(X_train, y_train)
            best_model = model

            if hasattr(best_model, 'alpha_'):
                print(f"Alpha: {best_model.alpha_}")

        best_models[name] = best_model
        print(f"-------------------- {name} Results --------------------")

        calculate_scores(X_test, y_test, best_model, name)

    return best_models


def calculate_scores(X_test, y_test_log, model, name):
    y_pred_log = model.predict(X_test)
    y_pred_actual = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test_log)

    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    mse = mean_squared_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_actual, y_pred_actual)

    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    limit_val = np.percentile(y_test_actual, 80)

    mask = (y_test_actual <= limit_val) & (y_pred_actual <= limit_val)
    y_test_filtered = y_test_actual[mask]
    y_pred_filtered = y_pred_actual[mask]

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_filtered, y_pred_filtered, alpha=0.5, color='blue')

    if len(y_test_filtered) > 1:
        m, b = np.polyfit(y_test_filtered, y_pred_filtered, 1)
        plt.plot(y_test_filtered, m*y_test_filtered + b, color='green', lw=2, label='Regression Line')


    plt.xlim(0, limit_val)
    plt.ylim(0, limit_val)
    plt.title(f'{name}')
    plt.xlabel('Y True')
    plt.ylabel('Y Predict')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return mae, mse, rmse, r2


if __name__ == "__main__":
    main()