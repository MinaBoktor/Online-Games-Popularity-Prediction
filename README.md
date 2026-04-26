# Online Games Popularity Prediction

## Analysis

### 1. Correlation Between Features

The correlation analysis shows a strong relationship among several features indicating feature redundancy.

### 2. Correlation with RecommendationCount

The most correlated features are:

- `SteamSpyPlayersEstimate`
- `SteamSpyPlayersVariance`
- `SteamSpyOwners`
- `SteamSpyOwnersVariance`

### 3. Top Correlated Features Heatmap

This graph indicates that the correlation between:

- `PriceInitial` and `PriceFinal` is **0.95**
- `SteamSpyOwners` and `SteamSpyPlayersEstimate` is **0.97**
- `SteamSpyOwnersVariance` and `SteamSpyPlayersVariance` is **0.97**

For each pair of correlated features, one must be removed to prevent multicollinearity.

### 4. Pair-Wise Relationships

This shows how one variable changes in relation to another:

- **Positive**: When one goes up, the other goes up — e.g., More owners → More Players
- **Negative**: When one goes up, the other goes down

---

## Preprocessing

### 1. Data Cleaning

- Drop Response Identification Columns: `["QueryID", "ResponseID", "QueryName", "ResponseName"]`
- Clip `PublisherCount` and `DeveloperCount` values lower than 1 (illogical)
- Drop `IsFree` and recreate it due to inaccuracy in values with respect to `PriceFinal`
- Convert `PriceCurrency` into a binary feature `PriceCurrencyUSD` (value = 1 for all records), then drop the original column
- Drop inaccurate rows where `SteamSpyOwners` = 0 and `RecommendationCount` > 0
- Drop inaccurate rows where `RecommendationCount` = 0 and `SteamSpyOwners` > 0
- Fix inconsistencies in `SupportedLanguages`
- Impute missing values using the median of games released in the same year
- Drop rows where `SteamSpyOwners` > 0 and `RecommendationCount` = 0 (likely an error)
- Drop rows where `SteamSpyOwners` = 0 and `RecommendationCount` > 0 (likely an error)

### 2. Encoding

- Binary encoding for `RequiredAge` (restricted or not)
- Binary encoding for all TRUE/FALSE columns
- Binarize presence/absence columns (1 = non-null, 0 = null): `['SupportEmail', 'SupportURL', 'Background', 'HeaderImage', 'Website']`
- Binary encoding for `LegalNotice`

### 3. Text Preprocessing (NLP)

- Lowercase all strings
- Embed: `['AboutText', 'ShortDescrip', 'DetailedDescrip', 'Reviews']`
- Textual features processed using the pre-trained transformer model **all-MiniLM-L6-v2**, which generates dense vector representations capturing semantic relationships

### 4. Feature Extraction

- Convert raw text (e.g., "8 GB RAM", "2 GHz CPU") into numerical features
- Extract game restrictions from `DRMNotice` and `ExtUserAcctNotice` using specific keywords

### 5. Feature Engineering

- Extract `ReleaseYear` and `ReleaseMonth` from `ReleaseDate`
- Divide `ReleaseMonth` into `Month_Sin` and `Month_Cos`
- Extract language count from `SupportedLanguages` → new column `LanguageCount`
- Create new features:
  - `Game_age` = 2026 − ReleaseYear
  - `New_game` = if Game_age < 3
  - `Old_game` = if Game_age > 10
  - `Players_per_Owner` = SteamSpyPlayersEstimate / SteamSpyOwners
- Extract RAM, STORAGE, PROCESSOR GHz from `process_system_requirements` into new columns based on:
  - `PCMinReqsText`, `PCRecReqsText`
  - `LinuxMinReqsText`, `LinuxRecReqsText`
  - `MacMinReqsText`, `MacRecReqsText`

### 6. Feature Transformation

- Power Transformation on all numerical columns using the **Yeo-Johnson** method to address skewness

### 7. Dimensionality Reduction

- High-dimensional embeddings reduced using **Principal Component Analysis (PCA)** to address the curse of dimensionality and improve computational efficiency

---

## Feature Selection

The following features were removed due to high correlation with other features:

- `SteamSpyOwners`
- `SteamSpyOwnersVariance`
- `PriceInitial`
- `ReleaseYear`
- `date_extraction_failure`

---

## Model

**Dataset Split:**

| Set | Samples |
|---|---|
| Training | 5,136 |
| Testing | 1,284 |
| Validation | 3-fold (3,424 training / 1,712 validation per fold) |

**GridSearchCV** was used to find the best hyperparameters for all models, except Ridge which used **RidgeCV**.

---

### First: Linear Models (Regularized)

#### Ridge (alpha: 0.01)

| Metric | Value |
|---|---|
| MAE | 534.1445 |
| MSE | 5,092,186.7803 |
| RMSE | 2,256.5874 |
| R² | **0.9683** |

The strong performance of Ridge Regression over more complex non-linear ensembles (Gradient Boosting, Random Forest) indicates that the underlying relationship between features and the target variable (`RecommendationCount`) is mostly linear.

#### Lasso (alpha: 0.01)

| Metric | Value |
|---|---|
| MAE | 660.6385 |
| MSE | 12,766,373.0636 |
| RMSE | 3,573.0062 |
| R² | 0.9206 |

Lasso uses L1 regularization, which can shrink some coefficients to zero, effectively performing feature selection. However, Ridge outperformed Lasso here, suggesting that eliminating features caused a loss of useful information.

---

### Second: Kernel Model

#### Support Vector Regressor (C: 10.0, kernel: linear)

| Metric | Value |
|---|---|
| MAE | 561.9609 |
| MSE | 5,909,749.3049 |
| RMSE | 2,430.9976 |
| R² | 0.9632 |

The best performance with a linear kernel further reinforces that the data has a predominantly linear structure.

---

### Third: Tree Models

#### Random Forest (max_depth: None, min_samples_leaf: 1, n_estimators: 200)

| Metric | Value |
|---|---|
| MAE | 582.3882 |
| MSE | 9,061,782.7920 |
| RMSE | 3,010.2795 |
| R² | 0.9436 |

Ridge outperforming Random Forest suggests that the complexity of the trees may be capturing noise rather than actual patterns, even with 200 estimators.

#### Gradient Boosting (learning_rate: 0.05, max_depth: 5, n_estimators: 200)

| Metric | Value |
|---|---|
| MAE | 595.0055 |
| MSE | 10,561,229.3927 |
| RMSE | 3,249.8045 |
| R² | 0.9343 |

The higher RMSE suggests the model may be capturing noise or is sensitive to hyperparameter choices.

---

## Conclusion

At the start of this project, we hypothesized that the number of players would have a strong impact on game popularity (`RecommendationCount`). Our models reinforce this idea by demonstrating that player-related features carry the strongest predictive signal, even when combined with a broader range of variables.

We detected highly correlated features causing multicollinearity. While Ridge regression remains robust in prediction due to L2 regularization, coefficient allocation among correlated variables becomes unstable and sensitive to changes in correlation structure — which can affect interpretability and feature importance consistency under distribution shifts.

We achieved a highly predictive model for game recommendations. **Ridge Regression achieved an R² score of 0.9683**, which is very strong. The strong performance of the linear kernel in the Support Vector Regressor further confirms that the relationship between features and `RecommendationCount` is mostly linear.

Preprocessing steps — including power transformation, extracting structured numerical values from raw text, and adding NLP embeddings for game descriptions — improved both the feature space and model performance. Overall, the results support the conclusion that while many factors affect prediction accuracy, **the number of players remains the most important factor in determining game popularity**.