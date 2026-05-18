import os
import time
import joblib
import pandas as pd
import warnings

from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

warnings.filterwarnings("ignore")

def get_classification_model(name, **kwargs):
    if name == "Random Forest Classifier": return RandomForestClassifier(random_state=42, n_jobs=-1, **kwargs)
    elif name == "Extra Trees Classifier": return ExtraTreesClassifier(random_state=42, n_jobs=-1, **kwargs)
    elif name == "Gradient Boosting Classifier": return GradientBoostingClassifier(random_state=42, **kwargs)
    elif name == "AdaBoost Classifier": return AdaBoostClassifier(random_state=42, **kwargs)
    elif name == "Logistic Regression": return LogisticRegression(random_state=42, **kwargs)
    elif name == "Naive Bayes": return GaussianNB(**kwargs)
    elif name == "XGBoost Classifier" and XGBClassifier is not None:
        return XGBClassifier(
            random_state=42,
            n_jobs=-1,
            objective='multi:softmax',
            num_class=3,
            eval_metric='mlogloss',
            **kwargs
        )
    elif name == "LightGBM Classifier" and LGBMClassifier is not None:
        return LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            class_weight={0: 1, 1: 25, 2: 5},
            **kwargs
        )
    elif name == "Support Vector Classifier":
        return SVC(random_state=42, class_weight='balanced', **kwargs)
    elif name == "Decision Tree Classifier":
        return DecisionTreeClassifier(random_state=42, class_weight='balanced', **kwargs)
    else:
        raise ValueError(f"{name} is not installed or not supported.")


def prep_classification_data(use_smote=True):
    X_train = pd.read_csv('X_train_classification.csv')
    X_test = pd.read_csv('X_test_classification.csv')
    y_train = pd.read_csv('y_train_classification.csv').squeeze("columns")
    y_test = pd.read_csv('y_test_classification.csv').squeeze("columns")

    leaky_cols = ['SteamSpyOwners', 'SteamSpyOwnersVariance', 'PriceInitial', 'ReleaseYear', 'date_extraction_failure']
    X_train = X_train.drop(columns=leaky_cols, errors='ignore')
    X_test = X_test.drop(columns=leaky_cols, errors='ignore')

    weights = None
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    else:
        weights = compute_sample_weight(class_weight='balanced', y=y_train)

    return X_train, y_train, X_test, y_test, weights

def train_and_eval_classification(model, X_train, y_train, X_test, y_test, weights=None):
    start_train = time.time()
    if weights is not None:
        try: model.fit(X_train, y_train, sample_weight=weights)
        except (TypeError, ValueError): model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_test = time.time()
    y_pred_test = model.predict(X_test)
    test_time = time.time() - start_test

    acc_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average='weighted')

    return model, train_time, test_time, acc_test, f1_test, y_pred_test