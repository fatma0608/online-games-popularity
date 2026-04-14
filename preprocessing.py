import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# LOAD DATA
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'train_data.csv'))
print(df.head())

missing_val = pd.DataFrame({
    "Nulls": df.isnull().sum(),
    "Percent": df.isnull().mean() * 100
}).sort_values(by="Percent", ascending=False)
print("\nMissing Values:\n", missing_val)

# Drop columns with lots of missing values
high_missing = df.isnull().mean()
drop_cols = high_missing[high_missing > 0.5].index
df.drop(columns=drop_cols, inplace=True)
print("\nDropped high missing columns:", list(drop_cols))

#Drop columns with CONSTANT & LOW VARIANCE
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
print("\nConstant columns:", constant_cols)
bool_cols = [c for c in df.columns if df[c].dtype == bool]
df[bool_cols] = df[bool_cols].astype(int)
bool_variance = df[bool_cols].var().sort_values()
low_var_bool = bool_variance[bool_variance < 0.001].index
df.drop(columns=low_var_bool, inplace=True)
print("\nDropped low variance bool columns:", list(low_var_bool))

df.drop(columns=['QueryID', 'ResponseID'], inplace=True, errors='ignore')

df['ReleaseDate'] = pd.to_datetime(df['ReleaseDate'], errors='coerce')
df['release_year'] = df['ReleaseDate'].dt.year
df['release_month'] = df['ReleaseDate'].dt.month
df['game_age_days'] = (pd.Timestamp.today() - df['ReleaseDate']).dt.days
# fill missing
df['release_year'] = df['release_year'].fillna(df['release_year'].median())
df['release_month'] = df['release_month'].fillna(6)
df['game_age_days'] = df['game_age_days'].fillna(df['game_age_days'].median())

df.drop(columns=['ReleaseDate'], inplace=True)

df['discount_ratio'] = (
    (df['PriceInitial'] - df['PriceFinal']) /
    (df['PriceInitial'] + 1e-9)
).clip(0, 1)

df['is_effectively_free'] = (
    (df['PriceInitial'] == 0) | (df['IsFree'] == 1)
).astype(int)

df['has_metacritic'] = (df['Metacritic'] > 0).astype(int)

df['num_languages'] = df['SupportedLanguages'].fillna('').apply(
    lambda x: len([w for w in x.split(' ') if len(w) > 2])
)

df['has_website']       = df['Website'].notna().astype(int)
df['has_support_email'] = df['SupportEmail'].notna().astype(int)
df['has_support_url']   = df['SupportURL'].notna().astype(int)
df['has_legal_notice']  = df['LegalNotice'].fillna('').apply(lambda x: 1 if len(x.strip()) > 1 else 0)
df['has_reviews_text']  = df['Reviews'].fillna('').apply(lambda x: 1 if len(x.strip()) > 5 else 0)
df['about_length']      = df['AboutText'].fillna('').apply(len)
df['short_length']      = df['ShortDescrip'].fillna('').apply(len)
df['detail_length']     = df['DetailedDescrip'].fillna('').apply(len)
df['has_pc_min_reqs']   = df['PCMinReqsText'].fillna('').apply(lambda x: 1 if len(x.strip()) > 5 else 0)
df['has_pc_rec_reqs']   = df['PCRecReqsText'].fillna('').apply(lambda x: 1 if len(x.strip()) > 5 else 0)
df['has_linux_min_reqs']= df['LinuxMinReqsText'].fillna('').apply(lambda x: 1 if len(x.strip()) > 5 else 0)
df['has_mac_min_reqs']  = df['MacMinReqsText'].fillna('').apply(lambda x: 1 if len(x.strip()) > 5 else 0)
df['has_drm']           = df['DRMNotice'].fillna('').apply(lambda x: 1 if len(x.strip()) > 1 else 0)
df['has_ext_account']   = df['ExtUserAcctNotice'].fillna('').apply(lambda x: 1 if len(x.strip()) > 1 else 0)

steamspy_cols = [
    'SteamSpyOwners', 'SteamSpyOwnersVariance',
    'SteamSpyPlayersEstimate', 'SteamSpyPlayersVariance'
]

for c in steamspy_cols:
    df[f'{c}_log'] = np.log1p(df[c])


df['target_log'] = np.log1p(df['RecommendationCount'])

# CLEAN RAW TEXT COLUMNS
about_text_backup = df['AboutText'].fillna('').copy()

drop_text_cols = [
    'QueryName', 'ResponseName',
    'Website', 'SupportEmail',
    'SupportURL','LegalNotice',
    'Reviews','SupportedLanguages', 
    'ShortDescrip', 'DetailedDescrip',
    'DRMNotice', 'ExtUserAcctNotice',
    'PriceCurrency',
    'Background',
    'HeaderImage',
]

df.drop(columns=[c for c in drop_text_cols if c in df.columns], inplace=True)

corr_matrix = df.corr(numeric_only=True)
target_corr = corr_matrix["target_log"].sort_values(ascending=False)
print("\nTop correlations with TARGET (log):\n")
print(target_corr)

# SPLIT
X = df.drop(columns=['RecommendationCount', 'target_log'])
y = df['target_log']

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42
)

# HANDLE MISSING VALUES (CORRECT WAY)
num_cols = X_train.select_dtypes(include=np.number).columns

X_train[num_cols] = X_train[num_cols].fillna(X_train[num_cols].median())
X_val[num_cols]   = X_val[num_cols].fillna(X_train[num_cols].median())
X_test[num_cols]  = X_test[num_cols].fillna(X_train[num_cols].median())

print("\nRemaining NaNs:")
print(X_train.isnull().sum().sum(), 
      X_val.isnull().sum().sum(), 
      X_test.isnull().sum().sum())