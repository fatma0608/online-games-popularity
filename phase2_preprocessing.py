import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('./data/processed', exist_ok=True)
os.makedirs('./models', exist_ok=True)
os.makedirs('./plots', exist_ok=True)

COLORS = ['#4C8EDA', '#E8593C', '#1D9E75', '#7F77DD', '#EF9F27']

# phase 2 preprocessing
df = pd.read_csv('./data/raw/train_data.csv')
print(f"Loaded dataset: {df.shape}")

#drop useless columns
print("\n" + "="*60)
print("STEP 1: Dropping Useless Columns")
print("="*60)

#drop high missing (>50%)
high_missing = df.isnull().mean()
drop_high_missing = high_missing[high_missing > 0.5].index.tolist()
df.drop(columns=drop_high_missing, inplace=True)
print(f"Dropped high-missing cols: {drop_high_missing}")

#drop constant columns features with zero variance, not useful for learning
constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
df.drop(columns=constant_cols, inplace=True)
print(f"Dropped constant cols: {constant_cols}")

#drop id columns not informative for prediction,may cause overfitting
df.drop(columns=['QueryID', 'ResponseID'], inplace=True, errors='ignore')
print("Dropped QueryID, ResponseID")

#drop low variance boolean features constant,provide little to no predictive information
bool_cols = [c for c in df.columns if df[c].dtype == bool]
df[bool_cols] = df[bool_cols].astype(int)
low_var_bool = [c for c in bool_cols if df[c].var() < 0.001]
df.drop(columns=low_var_bool, inplace=True)
print(f"Dropped low-variance bool cols: {low_var_bool}")

# 2 feature engineering
print("\n" + "="*60)
print("STEP 2: Feature Engineering")
print("="*60)

#date features
df['ReleaseDate'] = pd.to_datetime(df['ReleaseDate'], errors='coerce')
df['release_year']   = df['ReleaseDate'].dt.year
df['release_month']  = df['ReleaseDate'].dt.month
df['game_age_days']  = (pd.Timestamp.today() - df['ReleaseDate']).dt.days
df.drop(columns=['ReleaseDate'], inplace=True)
print(" Date features created: release_year, release_month, game_age_days")

#price features
df['discount_ratio']     = ((df['PriceInitial'] - df['PriceFinal']) /
                            (df['PriceInitial'] + 1e-9)).clip(0, 1)
df['is_effectively_free'] = ((df['PriceInitial'] == 0) |
                             (df['IsFree'] == 1)).astype(int)
print(" Price features created")

#content flags
df['has_metacritic']    = (df['Metacritic'] > 0).astype(int)
df['num_languages']     = df['SupportedLanguages'].fillna('').apply(
    lambda x: len([w for w in x.split(' ') if len(w) > 2]))
df['has_website']       = df['Website'].notna().astype(int)
df['has_support_email'] = df['SupportEmail'].notna().astype(int)
df['has_support_url']   = df['SupportURL'].notna().astype(int)
df['has_legal_notice']  = df['LegalNotice'].fillna('').apply(
    lambda x: 1 if len(x.strip()) > 1 else 0)
df['has_reviews_text']  = df['Reviews'].fillna('').apply(
    lambda x: 1 if len(x.strip()) > 5 else 0)
df['about_length']      = df['AboutText'].fillna('').apply(len)
df['short_length']      = df['ShortDescrip'].fillna('').apply(len)
df['detail_length']     = df['DetailedDescrip'].fillna('').apply(len)
df['has_pc_min_reqs']   = df['PCMinReqsText'].fillna('').apply(
    lambda x: 1 if len(x.strip()) > 5 else 0)
df['has_pc_rec_reqs']   = df['PCRecReqsText'].fillna('').apply(
    lambda x: 1 if len(x.strip()) > 5 else 0)
df['has_linux_min_reqs']= df['LinuxMinReqsText'].fillna('').apply(
    lambda x: 1 if len(x.strip()) > 5 else 0)
df['has_mac_min_reqs']  = df['MacMinReqsText'].fillna('').apply(
    lambda x: 1 if len(x.strip()) > 5 else 0)
df['has_drm']           = df['DRMNotice'].fillna('').apply(
    lambda x: 1 if len(x.strip()) > 1 else 0)
print(" Content flag features created")

#log transforms for SteamSpy
steamspy_cols = ['SteamSpyOwners', 'SteamSpyOwnersVariance',
                 'SteamSpyPlayersEstimate', 'SteamSpyPlayersVariance']
for c in steamspy_cols:
    if c in df.columns:
        df[f'{c}_log'] = np.log1p(df[c])
print(" Log transforms on SteamSpy columns")

#interaction features
df['price_per_language']    = df['PriceFinal'] / (df['num_languages'] + 1)
df['metacritic_x_age']      = df['has_metacritic'] * df['game_age_days']
df['owners_per_achievement']= df['SteamSpyOwners_log'] / (df['AchievementCount'] + 1)
df['dlc_x_owners']          = np.log1p(df['DLCCount']) * df['SteamSpyOwners_log']
df['movie_x_owners']        = df['MovieCount'] * df['SteamSpyOwners_log']
print(" Interaction features created")

#3:drop all text string columns 
drop_text_cols = [
    'QueryName', 'ResponseName', 'Website', 'SupportEmail', 'SupportURL',
    'LegalNotice', 'Reviews', 'SupportedLanguages', 'ShortDescrip',
    'DetailedDescrip', 'DRMNotice', 'ExtUserAcctNotice', 'PriceCurrency',
    'Background', 'HeaderImage', 'AboutText', 'PCMinReqsText',
    'PCRecReqsText', 'LinuxMinReqsText', 'MacMinReqsText', 'MacRecReqsText',
    'LinuxRecReqsText',
]
df.drop(columns=[c for c in drop_text_cols if c in df.columns], inplace=True)
remaining_text = df.select_dtypes(include='object').columns.tolist()
# Drop any remaining object columns EXCEPT target
remaining_text = [c for c in remaining_text if c != 'GamePopularity']
df.drop(columns=remaining_text, inplace=True)
print(f" Dropped text columns. Remaining object cols dropped: {remaining_text}")

#4 encode target
print("\n" + "="*60)
print("STEP 4: Encode Target — GamePopularity")
print("="*60)

le = LabelEncoder()
df['GamePopularity_encoded'] = le.fit_transform(df['GamePopularity'])
print(f"Classes: {le.classes_}  →  {list(range(len(le.classes_)))}")
joblib.dump(le, './models/label_encoder.pkl')
print(" Saved: models/label_encoder.pkl")

#5 train/test split
print("\n" + "="*60)
print("STEP 5: Train/Test Split (80/20)")
print("="*60)

X = df.drop(columns=['GamePopularity', 'GamePopularity_encoded'])
y = df['GamePopularity_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y  # stratify keeps class balance
)

# Save indices for NLP script alignment
np.save('./data/processed/idx_train.npy', X_train.index.values)
np.save('./data/processed/idx_test.npy',  X_test.index.values)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(" Saved split indices for NLP alignment")

#6: fill missing values with train medians 
print("\n" + "="*60)
print("STEP 6: Fill Missing Values (Train Medians)")
print("="*60)

num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
train_medians = X_train[num_cols].median()

X_train[num_cols] = X_train[num_cols].fillna(train_medians)
X_test[num_cols]  = X_test[num_cols].fillna(train_medians)  # use TRAIN medians on test

#save medians for test script
joblib.dump(train_medians, './models/train_medians.pkl')
print(f" Filled NaNs using train medians")
print(f"   Remaining NaNs — Train: {X_train.isnull().sum().sum()}, Test: {X_test.isnull().sum().sum()}")
print(" Saved: models/train_medians.pkl")

# NOTE: SMOTE is applied AFTER IQR capping and scaling (correct order)
# so synthetic samples are generated in the same scaled/capped space as real data

# 7 IQR
print("\n" + "="*60)
print("STEP 7: IQR Outlier Capping")
print("="*60)

NO_IQR_COLS = ['RequiredAge', 'DemoCount', 'DeveloperCount',
               'DLCCount', 'PackageCount', 'PublisherCount']
iqr_bounds = {}

CONTINUOUS_COLS = [
    'RequiredAge', 'DemoCount', 'DeveloperCount', 'DLCCount',
    'MovieCount', 'PackageCount', 'PublisherCount', 'ScreenshotCount',
    'SteamSpyOwners', 'SteamSpyOwnersVariance',
    'SteamSpyPlayersEstimate', 'SteamSpyPlayersVariance',
    'AchievementCount', 'AchievementHighlightedCount',
    'PriceInitial', 'PriceFinal',
    'release_year', 'release_month', 'game_age_days', 'num_languages',
    'about_length', 'short_length', 'detail_length',
    'SteamSpyOwners_log', 'SteamSpyOwnersVariance_log',
    'SteamSpyPlayersEstimate_log', 'SteamSpyPlayersVariance_log',
    'price_per_language', 'metacritic_x_age',
    'owners_per_achievement', 'dlc_x_owners', 'movie_x_owners',
]
cont_feat_cols = [c for c in CONTINUOUS_COLS if c in X_train.columns]

for col in cont_feat_cols:
    if col in NO_IQR_COLS:
        X_train[col] = np.log1p(X_train[col])
        X_test[col]  = np.log1p(X_test[col])
        continue
    Q1, Q3 = X_train[col].quantile(0.25), X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0:
        continue
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    iqr_bounds[col] = (lower, upper)
    X_train[col] = X_train[col].clip(lower, upper)
    X_test[col]  = X_test[col].clip(lower, upper)  # clip test using TRAIN bounds

#save IQR bounds for test script
joblib.dump(iqr_bounds, './models/iqr_bounds.pkl')
print(f" IQR capped {len(iqr_bounds)} columns")
print(" Saved: models/iqr_bounds.pkl")

#8 standard scalling
print("\n" + "="*60)
print("STEP 8: Standard Scaling")
print("="*60)

scaler = StandardScaler()
X_train[cont_feat_cols] = scaler.fit_transform(X_train[cont_feat_cols])
X_test[cont_feat_cols]  = scaler.transform(X_test[cont_feat_cols])  # transform only

joblib.dump(scaler, './models/scaler.pkl')
print(f" Scaled {len(cont_feat_cols)} continuous columns")
print(" Saved: models/scaler.pkl")

# ── FIX: Apply SMOTE AFTER IQR capping and scaling ──────────────
# Correct order: clean real data first, then generate synthetic samples
# in the same scaled/capped space. Doing SMOTE before scaling would
# produce synthetic points in the raw unscaled space.
from imblearn.combine import SMOTEENN
print("\n" + "="*60)
print("STEP 8b: SMOTE Oversampling (applied after scaling)")
print("="*60)
print("Before SMOTE:", pd.Series(y_train).value_counts().sort_index().to_dict())

smote_enn = SMOTEENN(random_state=42)

X_train_sm, y_train_sm = smote_enn.fit_resample(X_train, y_train)

X_train = pd.DataFrame(X_train_sm, columns=X_train.columns)
y_train = pd.Series(y_train_sm)

print("After  SMOTE:", pd.Series(y_train).value_counts().sort_index().to_dict())
print(" SMOTE applied — minority classes balanced to match majority")

#9 feature selection
print("\n" + "="*60)
print("STEP 9: Feature Selection — mutual_info_classif (MI > 0.01)")
print("="*60)

# NOTE: MI is computed on SMOTE-augmented train — this is correct because
# we want to find features that distinguish classes in the balanced space.
# The selected_features list is saved and used identically in the test script.
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_df = pd.DataFrame({
    'feature': X_train.columns,
    'MI': mi_scores
}).sort_values('MI', ascending=False)

print("\nTop 20 features by MI:")
print(mi_df.head(20).to_string())

# ── FIX: Use MI > 0.01 threshold instead of arbitrary top-30 ────
# top-30 is too restrictive — it discards features that ARE informative
# but happen to rank 31st. MI > 0.01 keeps all meaningfully informative
# features (NLP features will be added later in phase3).
MI_THRESHOLD = 0.005
selected_features = mi_df[mi_df['MI'] > MI_THRESHOLD]['feature'].tolist()
print(f"\n Features with MI > {MI_THRESHOLD}: {len(selected_features)} / {len(mi_df)}")

# Plot MI
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F7F4')
ax.set_facecolor('#F0EFE8')
top20 = mi_df.head(20)
ax.barh(top20['feature'][::-1], top20['MI'][::-1],
        color='#7F77DD', alpha=0.85, edgecolor='white')
ax.set_title('Top 20 Features — Mutual Information with GamePopularity',
             fontsize=11, fontweight='500', color='#2C2C2A')
ax.set_xlabel('MI Score (classif)', fontsize=9)
ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig('./plots/phase2_mutual_information.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: plots/phase2_mutual_information.png")

joblib.dump(selected_features, './models/selected_features.pkl')
print(f" Saved: models/selected_features.pkl  ({len(selected_features)} features)")

#apply feature selection
X_train = X_train[selected_features]
X_test  = X_test[selected_features]

# 10: Save Processed Data 
print("\n" + "="*60)
print("STEP 10: Saving Processed Data")
print("="*60)

train_df = X_train.copy()
train_df['GamePopularity_encoded'] = y_train.values
test_df  = X_test.copy()
test_df['GamePopularity_encoded']  = y_test.values

train_df.to_csv('./data/processed/train.csv', index=False)
test_df.to_csv('./data/processed/test.csv',   index=False)

print(f" Saved: data/processed/train.csv  → {train_df.shape}")
print(f" Saved: data/processed/test.csv   → {test_df.shape}")

print("\n" + "="*60)
print("PHASE 2 COMPLETE — All models saved:")
print("="*60)
print("  models/label_encoder.pkl")
print("  models/train_medians.pkl")
print("  models/iqr_bounds.pkl")
print("  models/scaler.pkl")
print("  models/selected_features.pkl")
print(f"\nFinal feature count : {len(selected_features)}")
print(f"X_train shape       : {X_train.shape}")
print(f"X_test  shape       : {X_test.shape}")