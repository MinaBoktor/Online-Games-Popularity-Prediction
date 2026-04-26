from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import pandas as pd
import re


def preprocess(df, is_train=True, scaler=None, text_pcas=None):

    # Lower Case
    string_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in string_cols:
        df[col] = df[col].astype(str).str.lower()

    # Drop Response Identification Columns
    df.drop(columns=["QueryID", "ResponseID", "QueryName", "ResponseName"], inplace=True)

    # Release Data
    df = release_date(df, 'ReleaseDate')
    df['Month_Sin'] = np.sin(2 * np.pi * df['ReleaseMonth'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['ReleaseMonth'] / 12)
    df.drop(columns=['ReleaseMonth'], inplace=True)

    # Binarize RequiredAge Column
    df['RequiredAge'] = (df['RequiredAge'] >= 2).astype(int)

    # Turn error in Developer Count to 1
    df['DeveloperCount'] = df['DeveloperCount'].replace({0: 1})
    # Turn error in Publisher Count to 1
    df['PublisherCount'] = df['PublisherCount'].replace({0: 1})

    # Binary Columns
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Fixing Errors at IsFree
    df['IsFree'] = df['PriceFinal'].apply(lambda x: 1 if x == 0 else 0)

    # PriceCurrency have only USD
    df['PriceCurrencyUSD'] = 1
    df.drop(columns=['PriceCurrency'], inplace=True)

    # Some columns were binarized to indicate the presence or absence of data, where 1 represents a non-null value and 0 represents a null value.
    link_cols = ['SupportEmail', 'SupportURL', 'Background', 'HeaderImage', 'Website', 'LegalNotice']

    for col in link_cols:
        df[col] = (df[col].replace(r'^\s*$', np.nan, regex=True).notna().astype(int))

    # Preprocessing Text using NLP Model (all-MiniLM-L6-v2)
    text_cols_to_embed = ['AboutText', 'ShortDescrip', 'DetailedDescrip', 'Reviews', ]
    df, text_pcas = text_embedding(df, text_cols_to_embed, is_train, text_pcas)
    df.drop(columns=text_cols_to_embed, inplace=True)

    # Extracting Languages from SupportedLanguages
    df['SupportedLanguages'] = df['SupportedLanguages'].apply(support_languages)
    df['LanguageCount'] = df['SupportedLanguages'].apply(len)
    df.drop(columns=['SupportedLanguages'], inplace=True)

    # Extracting game restrictions from DRMNotice and ExtUserAcctNotice
    df = extract_game_restrictions(df)

    # Processing Different Systems Min and Recommend Requirement
    df = process_system_requirements(df)

    # Feature Engineering
    df['Game_age'] = 2026 - df['ReleaseYear']
    df['New_game'] = (df['Game_age'] < 3).astype(int)
    df['Old_game'] = (df['Game_age'] > 10).astype(int)
    df['Players_per_Owner'] = df['SteamSpyPlayersEstimate'] / (df['SteamSpyOwners'] + 1)
    df['ContentTotal'] = df['ScreenshotCount'] + df['MovieCount'] + df['DLCCount']

    # Fill Null values using Median
    df = fill_na(df)

    # Power Transform all values to solve Skewness
    continuous_cols = df.select_dtypes(include=['number']).columns

    if is_train:
        scaler = PowerTransformer(method='yeo-johnson')
        df[continuous_cols] = pd.DataFrame(
            scaler.fit_transform(df[continuous_cols]),
            columns=continuous_cols,
            index=df.index
        )

        return df, scaler, text_pcas

    else:
        if scaler is None:
            raise ValueError("You must pass a fitted 'scaler' object when is_train=False")
        df[continuous_cols] = scaler.transform(df[continuous_cols])

        return df


def drop_unnecessary_rows(df):

    first_condition = (df['SteamSpyOwners'] > 0) & (df['RecommendationCount'] == 0)
    second_condition = (df['SteamSpyOwners'] == 0) & (df['RecommendationCount'] > 0)

    df = df[~first_condition]
    df = df[~second_condition]

    return df


device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)


def text_embedding(df, text_cols, is_train=True, text_pcas=None, n_components=5):

    existing_cols = [c for c in text_cols if c in df.columns]

    if not existing_cols:
        return df, text_pcas

    pca_dfs = []

    if is_train:
        text_pcas = {}
    elif text_pcas is None:
        raise ValueError("You must pass a fitted 'text_pcas' dictionary when is_train=False")

    for col in existing_cols:
        print(f"[{'TRAIN' if is_train else 'TEST'}] Embedding column: {col}...")

        text_data = df[col].fillna('').astype(str).tolist()

        embeddings = embedder.encode(text_data, show_progress_bar=True, batch_size=32)

        if is_train:
            pca = PCA(n_components=n_components, random_state=42)
            compressed_embeddings = pca.fit_transform(embeddings)
            text_pcas[col] = pca
        else:
            pca = text_pcas[col]
            compressed_embeddings = pca.transform(embeddings)

        feature_names = [f'{col}_pca_{i}' for i in range(n_components)]
        col_pca_df = pd.DataFrame(compressed_embeddings, columns=feature_names, index=df.index)
        pca_dfs.append(col_pca_df)

    df = pd.concat([df] + pca_dfs, axis=1)

    return df, text_pcas


def process_system_requirements(df):
    df = df.copy()

    req_cols = [
        'PCMinReqsText', 'PCRecReqsText',
        'LinuxMinReqsText', 'LinuxRecReqsText',
        'MacMinReqsText', 'MacRecReqsText'
    ]

    def extract_text(text):
        if pd.isna(text):
            return pd.Series({'RAM_GB': None, 'Storage_GB': None, 'Processor_GHz': None})

        text = str(text).lower()
        text = re.sub(r'gigabytes?', 'gb', text)
        text = re.sub(r'megabytes?', 'mb', text)
        text = re.sub(r'terabytes?', 'tb', text)
        text = re.sub(r'kilobytes?', 'kb', text)

        def to_gb(val, unit):
            if not val: return None
            v = float(val)
            if unit == 'tb': return v * 1024.0
            if unit == 'gb': return v
            if unit == 'mb': return v / 1024.0
            if unit == 'kb': return v / (1024.0 * 1024.0)
            return v

        ram_gb = None
        ram_match = re.search(r'(?:memory|ram)[\s\:\-]+(\d+(?:\.\d+)?)\s*(tb|gb|mb|kb)|(\d+(?:\.\d+)?)\s*(tb|gb|mb|kb)\s*(?:of\s*)?(?:ram|memory|system memory)', text)
        if ram_match:
            val = ram_match.group(1) or ram_match.group(3)
            unit = ram_match.group(2) or ram_match.group(4)
            ram_gb = to_gb(val, unit)

        storage_gb = None
        storage_match = re.search(r'(?:storage|disk space|space|hard drive|hdd|ssd)[\s\:\-]+(\d+(?:\.\d+)?)\s*(tb|gb|mb|kb)|(\d+(?:\.\d+)?)\s*(tb|gb|mb|kb)\s*(?:of\s*)?(?:available|free|storage|space|disk|hd|hdd|ssd)', text)
        if storage_match:
            val = storage_match.group(1) or storage_match.group(3)
            unit = storage_match.group(2) or storage_match.group(4)
            storage_gb = to_gb(val, unit)

        cpu_ghz = None
        cpu_match = re.search(r'(\d+(?:\.\d+)?)\s*(ghz|mhz)', text)
        if cpu_match:
            val, unit = float(cpu_match.group(1)), cpu_match.group(2)
            cpu_ghz = val if unit == 'ghz' else val / 1000.0

        return pd.Series({'RAM_GB': ram_gb, 'Storage_GB': storage_gb, 'Processor_GHz': cpu_ghz})

    for col in req_cols:
        if col in df.columns:
            extracted_features = df[col].apply(extract_text)

            extracted_features = extracted_features.rename(columns={
                'RAM_GB': f'{col}_RAM_GB',
                'Storage_GB': f'{col}_Storage_GB',
                'Processor_GHz': f'{col}_Processor_GHz'
            })

            df = pd.concat([df, extracted_features], axis=1)

    df.drop(columns=[c for c in req_cols if c in df.columns], inplace=True, errors='ignore')

    return df


def extract_game_restrictions(df, drm_col='DRMNotice', notice_col='ExtUserAcctNotice'):

    df = df.copy()

    for col in [drm_col, notice_col]:
        if col in df.columns:
            df[col] = df[col].replace([' ', ''], np.nan)
            df[col] = df[col].str.lower().str.strip()

    df['drm_score'] = 0

    if drm_col in df.columns:
        df['drm_score'] += df[drm_col].str.contains('denuvo', na=False).astype(int)
        df['drm_score'] += df[drm_col].str.contains('activation limit', na=False).astype(int)
        df['drm_score'] += df[drm_col].str.contains('account|login|online|internet', na=False).astype(int)

    df['notice_score'] = 0

    if notice_col in df.columns:
        df['notice_score'] += df[notice_col].str.contains('account|login|registration', na=False).astype(int)
        df['notice_score'] += df[notice_col].str.contains('steam|uplay|ubisoft|rockstar', na=False).astype(int)
        df['notice_score'] += df[notice_col].str.contains('online|cloud|multiplayer|server', na=False).astype(int)
        df['notice_score'] += df[notice_col].str.contains('facebook|twitter|google|twitch', na=False).astype(int)
        df['notice_score'] += df[notice_col].str.contains(r'http|www|https', na=False).astype(int)

    df['total_restriction_score'] = df['drm_score'] + df['notice_score']

    df.drop(columns=[c for c in [drm_col, notice_col] if c in df.columns], inplace=True)

    return df


def support_languages(x):
    if pd.isna(x):
        return []

    x_str = str(x).lower()
    results = re.split(r'\s*\*+\s*', x_str)

    final = []
    if 'with full audio support' in x_str:
        final = ['full audio support']

    for i in results:
        match = re.findall(r'[a-z]{2,}', i)
        if match:
            final.extend(match)

    ignore_words = ['languages', 'with', 'full', 'audio', 'support', 'text', 'only']
    final = list(set(final) - set(ignore_words))

    return final


def release_date(df, date_column='ReleaseDate'):
    raw_dates = df[date_column].astype(str)
    parsed_dates = pd.to_datetime(raw_dates, errors='coerce')

    years = parsed_dates.dt.year
    months = parsed_dates.dt.month

    failed = parsed_dates.isna()

    def manual_search_year(text):
        if text == 'nan' or not text: return np.nan
        match = re.search(r'(19\d{2}|20\d{2})', text)
        return int(match.group(1)) if match else np.nan

    def manual_search_month(text):
        if text == 'nan' or not text: return np.nan
        text = text.lower()
        if any(w in text for w in ['q1', 'spring', 'jan', 'feb', 'mar', 'first quarter']): return 3
        if any(w in text for w in ['q2', 'summer', 'apr', 'may', 'jun', 'mid']): return 6
        if any(w in text for w in ['q3', 'autumn', 'fall', 'jul', 'aug', 'sep']): return 9
        if any(w in text for w in ['q4', 'winter', 'holiday', 'christmas', 'halloween', 'oct', 'nov', 'dec']): return 12
        return np.nan

    years = years.fillna(raw_dates[failed].apply(manual_search_year))
    months = months.fillna(raw_dates[failed].apply(manual_search_month))

    df['ReleaseYear'] = years.fillna(-1).astype(int)
    df['ReleaseMonth'] = months.fillna(-1).astype(int)

    df['date_extraction_failure'] = (df['ReleaseYear'] == -1).astype(int)

    df.drop(columns=[date_column], inplace=True)

    return df


def fill_na(df):
    df = df.copy()

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    exclude_cols = ['ReleaseYear', 'ReleaseMonth', 'Game_age'] 
    target_cols = [c for c in numeric_cols if c not in exclude_cols]

    for col in target_cols:
        if df[col].isna().sum() > 0:

            df[f'{col}_Missing'] = df[col].isna().astype(int)

            if df[col].dropna().nunique() <= 2:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
            else:
                year_medians = df.groupby('ReleaseYear')[col].transform('median')
                df[col] = df[col].fillna(year_medians)

                global_median = df[col].median()
                df[col] = df[col].fillna(global_median if not pd.isna(global_median) else 0)

    return df


if __name__ == "__main__":
    df = pd.read_csv('train_data.csv')
    df, _, _ = preprocess(df)
    df.to_csv("temp.csv", index=False)