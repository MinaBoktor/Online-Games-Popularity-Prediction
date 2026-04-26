import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def visualization(df):

    log_cols = [
        'PriceInitial',
        'PriceFinal',
        'SteamSpyOwners',
        'SteamSpyPlayersEstimate'
    ]

    valid_cols = [c for c in log_cols if c in df.columns]

    if valid_cols:
        _, axes = plt.subplots(1, len(valid_cols), figsize=(12, 4))

        if len(valid_cols) == 1:
            axes = [axes]

        for i, col in enumerate(valid_cols):
            axes[i].boxplot(df[col].dropna())
            axes[i].set_title(col)

        plt.tight_layout()
        plt.show()
    else:
        print("No valid columns for boxplots.")


    pd.set_option('display.max_columns', None)
    print("Data shape:", df.shape)
    print(df.head())


    if 'RecommendationCount' in df.columns:
        numeric_df = df.select_dtypes(include='number')

        corr = (
            numeric_df.corr()['RecommendationCount']
            .drop('RecommendationCount')
            .sort_values()
        )

        plt.figure(figsize=(8, 6))
        corr.plot(kind='barh', color='skyblue', edgecolor='black')
        plt.title('Correlation with RecommendationCount')
        plt.xlabel('Correlation')
        plt.ylabel('Feature')
        plt.grid(axis='x', alpha=0.8)
        plt.axvline(x=0, color='black', linewidth=1)
        plt.tight_layout()
        plt.show()
    else:
        print("RecommendationCount column not found.")


    plt.figure(figsize=(10, 8))
    corr_matrix = df.select_dtypes(include='number').corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt='.2f',
        annot_kws={"size": 7},
        cbar_kws={"shrink": .8}
    )
    plt.title('Feature Correlation Heatmap', fontsize=12)

    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)

    plt.tight_layout()
    plt.show()


    sample_df = df.sample(n=500, random_state=42)
    sns.pairplot(sample_df[['RecommendationCount'] + list(corr.index[1:5])])
    plt.show()

    price_cols = ['PriceInitial', 'PriceFinal', 'SteamSpyOwners', 'SteamSpyPlayersEstimate', 'SteamSpyOwnersVariance', 'SteamSpyPlayersVariance']
    price_cols = [c for c in price_cols if c in df.columns]

    if len(price_cols) >= 2:
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            df[price_cols].corr(),
            annot=True,
            cmap='coolwarm',
            center=0
        )
        plt.title('Top Correlated Features Heatmap')
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough columns found.")


def feature_importance(model, feature_names, model_name="Ridge Regression"):
    if not hasattr(model, 'coef_'):
        print(f"Skipping: {model_name} does not have a 'coef_' attribute.")
        return

    coefs = model.coef_.flatten()

    feat_imp = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})

    feat_imp = feat_imp.sort_values(by='Coefficient', ascending=True)

    top_neg = feat_imp.head(15)
    top_pos = feat_imp.tail(15)
    combined_imp = pd.concat([top_neg, top_pos])

    plt.figure(figsize=(10, 8))
    colors = ['red' if x < 0 else 'green' for x in combined_imp['Coefficient']]
    plt.barh(combined_imp['Feature'], combined_imp['Coefficient'], color=colors)
    plt.title(f'Top 15 Positive and Negative features - {model_name}')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def show_top_correlations(df, threshold=0.8):
    if 'RecommendationCount' in df.columns:
        df = df.drop(columns=['RecommendationCount'])

    df = df.select_dtypes(include='number')

    corr_matrix = df.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    corr_pairs = (
        upper.stack()
        .reset_index()
    )

    corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']

    high_corr = corr_pairs[corr_pairs['Correlation'] > threshold]

    high_corr = high_corr.sort_values(by='Correlation', ascending=False)

    print(f"\nTop correlated feature pairs (>{threshold}):\n")
    print(high_corr)

    return high_corr


if __name__ == "__main__":
    try:
        df = pd.read_csv(r"train_data.csv")
        visualization(df)
    except FileNotFoundError:
        print("Could not find 'train_data.csv'")