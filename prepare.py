import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from helper import preprocess, drop_unnecessary_rows

def prepare_data(task):
    print(f"prepare {task} data")

    DATA_PATHS = {
        'classification': 'train_data_classification.csv',
        'regression': 'train_data_regression.csv'
    }

    data_path = DATA_PATHS[task]

    try:
        print(f"Loading raw dataset")
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"File Not found")
        return

    if task == 'regression':
        try:
            df = drop_unnecessary_rows(df)
        except NameError:
            print("'drop_unnecessary_rows' function not found.")

        X = df.drop(columns=['RecommendationCount'])
        y = df['RecommendationCount']
        y_transformed = np.log1p(y)

    elif task == 'classification':
        X = df.drop(columns=['GamePopularity'])
        encoding = {'Low': 0, 'Medium': 1, 'High': 2}
        y_transformed = df['GamePopularity'].map(encoding)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_transformed, test_size=0.2, random_state=42, shuffle=True
    )

    X_train_prep, scaler, text_pcas, train_cols = preprocess(X_train, is_train=True)

    X_test_prep = preprocess(
        X_test,
        is_train=False,
        scaler=scaler,
        text_pcas=text_pcas,
        train_cols=train_cols
    )

    X_train_prep.to_csv(f'X_train_{task}.csv', index=False)
    X_test_prep.to_csv(f'X_test_{task}.csv', index=False)
    y_train.to_csv(f'y_train_{task}.csv', index=False)
    y_test.to_csv(f'y_test_{task}.csv', index=False)

    joblib.dump({
        'scaler': scaler,
        'text_pcas': text_pcas,
        'columns': train_cols
    }, f'preprocessing_tools_{task}.pkl')

    print(f"\nDone")

def main():
    parser = argparse.ArgumentParser(description="Unified Data Preparation Pipeline")
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], required=True)
    args = parser.parse_args()

    prepare_data(task=args.task)

if __name__ == "__main__":
    main()