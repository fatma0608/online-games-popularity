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

cat_cols = X_train.select_dtypes(include=['object']).columns
print(cat_cols)

#######################encoding###########################
X_train = pd.get_dummies(X_train, columns=cat_cols)
X_val   = pd.get_dummies(X_val, columns=cat_cols)
X_test  = pd.get_dummies(X_test, columns=cat_cols)

X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
##############################################################
print("\nRemaining NaNs:")
print(X_train.isnull().sum().sum(),
      X_val.isnull().sum().sum(),
      X_test.isnull().sum().sum())

print(df.shape)

# print(df.columns)

# print(df.head())

# df.info()

# df.describe(include='all')

# for col, val in df.nunique().sort_values().items():
#     print(f"{col}: {val}")

#dominant_ratio
to_drop = []

for col in df.columns:
    dominant_ratio = df[col].value_counts(normalize=True, dropna=False).max()

    if dominant_ratio >= 0.95:
        to_drop.append(col)

df.drop(columns=to_drop, inplace=True)

print("Dropped columns:", to_drop)

#VarianceThreshold
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)

feature_names = X_train.columns

X_train_sel = selector.fit_transform(X_train)
X_val_sel   = selector.transform(X_val)
X_test_sel  = selector.transform(X_test)

selected_cols = feature_names[selector.get_support()]

X_train = pd.DataFrame(X_train_sel, columns=selected_cols)
X_val   = pd.DataFrame(X_val_sel, columns=selected_cols)
X_test  = pd.DataFrame(X_test_sel, columns=selected_cols)

print("New shape:", X_train.shape)
print("Selected features:", len(selected_cols))

# Correlation between columns
corr = X_train.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

to_drop = [c for c in upper.columns if any(upper[c] > 0.9)]
print(to_drop)
X_train.drop(columns=to_drop, inplace=True)
X_val.drop(columns=to_drop, inplace=True)
X_test.drop(columns=to_drop, inplace=True)

#corr with target
num_cols = X_train.select_dtypes(include=np.number).columns

corr_with_target = X_train[num_cols].corrwith(y_train).abs()

low_corr_features = corr_with_target[corr_with_target < 0.01].index

X_train.drop(columns=low_corr_features, inplace=True)
X_val.drop(columns=low_corr_features, inplace=True)
X_test.drop(columns=low_corr_features, inplace=True)

print("Dropped low correlation features:", list(low_corr_features))

#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(
    RandomForestRegressor(n_estimators=100, random_state=42),
    threshold="median"
)

selector.fit(X_train, y_train)

X_train_sel = selector.transform(X_train)
X_val_sel   = selector.transform(X_val)
X_test_sel  = selector.transform(X_test)
