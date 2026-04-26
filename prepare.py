import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from helper import preprocess, drop_unnecessary_rows

def main():
    df = pd.read_csv('train_data.csv')
    df = drop_unnecessary_rows(df)

    X = df.drop(columns=['RecommendationCount'])
    y = df['RecommendationCount']
    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42, shuffle=True)

    print("Preprocessing Training Data")
    X_train_prep, scaler, text_pcas = preprocess(X_train, is_train=True)
    print("Preprocessing Testing Data")
    X_test_prep = preprocess(X_test, is_train=False, scaler=scaler, text_pcas=text_pcas)

    print("Saving preprocessed datasets")

    X_train_prep.to_csv('X_train_final.csv', index=False)
    X_test_prep.to_csv('X_test_final.csv', index=False)
    y_train_log.to_csv('y_train_final.csv', index=False)
    y_test_log.to_csv('y_test_final.csv', index=False)

    print("Saving fitted preprocessing tools (Scaler, PCAs)...")
    joblib.dump({'scaler': scaler, 'text_pcas': text_pcas}, 'preprocessing_tools.pkl')

    print("Fitted preprocessing tools were saved successfully")

if __name__ == "__main__":
    main()