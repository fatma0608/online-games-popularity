import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv('./data/raw/train_data.csv')
print(df.head())

missing_val = pd.DataFrame({
    "Nulls": df.isnull().sum(),
    "Percent": df.isnull().mean() * 100
}).sort_values(by="Percent", ascending=False)
print("\nMissing Values:\n", missing_val)

high_missing = df.isnull().mean()
drop_cols = high_missing[high_missing > 0.5].index
df.drop(columns=drop_cols, inplace=True)
print("\nDropped high missing columns:", list(drop_cols))

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

df['has_metacritic']     = (df['Metacritic'] > 0).astype(int)
df['num_languages']      = df['SupportedLanguages'].fillna('').apply(
    lambda x: len([w for w in x.split(' ') if len(w) > 2]))

df['has_website']        = df['Website'].notna().astype(int)
df['has_support_email']  = df['SupportEmail'].notna().astype(int)
df['has_support_url']    = df['SupportURL'].notna().astype(int)
df['has_legal_notice']   = df['LegalNotice'].fillna('').apply(lambda x: 1 if len(x.strip()) > 1 else 0)
df['has_reviews_text']   = df['Reviews'].fillna('').apply(lambda x: 1 if len(x.strip()) > 5 else 0)
df['about_length']       = df['AboutText'].fillna('').apply(len)
df['short_length']       = df['ShortDescrip'].fillna('').apply(len)
df['detail_length']      = df['DetailedDescrip'].fillna('').apply(len)
df['has_pc_min_reqs']    = df['PCMinReqsText'].fillna('').apply(lambda x: 1 if len(x.strip()) > 5 else 0)
df['has_pc_rec_reqs']    = df['PCRecReqsText'].fillna('').apply(lambda x: 1 if len(x.strip()) > 5 else 0)
df['has_linux_min_reqs'] = df['LinuxMinReqsText'].fillna('').apply(lambda x: 1 if len(x.strip()) > 5 else 0)
df['has_mac_min_reqs']   = df['MacMinReqsText'].fillna('').apply(lambda x: 1 if len(x.strip()) > 5 else 0)
df['has_drm']            = df['DRMNotice'].fillna('').apply(lambda x: 1 if len(x.strip()) > 1 else 0)
df['has_ext_account']    = df['ExtUserAcctNotice'].fillna('').apply(lambda x: 1 if len(x.strip()) > 1 else 0)

steamspy_cols = [
    'SteamSpyOwners', 'SteamSpyOwnersVariance',
    'SteamSpyPlayersEstimate', 'SteamSpyPlayersVariance'
]
for c in steamspy_cols:
    df[f'{c}_log'] = np.log1p(df[c])

df['target_log'] = np.log1p(df['RecommendationCount'])

# ─────────────────────────────────────────────
# DROP ALL TEXT / STRING COLUMNS (no text features)
# ─────────────────────────────────────────────
drop_text_cols = [
    'QueryName', 'ResponseName', 'Website', 'SupportEmail', 'SupportURL',
    'LegalNotice', 'Reviews', 'SupportedLanguages', 'ShortDescrip',
    'DetailedDescrip', 'DRMNotice', 'ExtUserAcctNotice', 'PriceCurrency',
    'Background', 'HeaderImage',
    # Also drop raw text source columns used to derive features
    'AboutText', 'PCMinReqsText', 'PCRecReqsText',
    'LinuxMinReqsText', 'MacMinReqsText',
]
df.drop(columns=[c for c in drop_text_cols if c in df.columns], inplace=True)

# Drop any remaining object/string columns
remaining_text = df.select_dtypes(include='object').columns.tolist()
if remaining_text:
    print(f"\nDropping remaining object columns: {remaining_text}")
    df.drop(columns=remaining_text, inplace=True)

print(f"\nColumns after dropping text features: {df.columns.tolist()}")

# ─────────────────────────────────────────────
# DEFINE BINARY / CATEGORICAL FLAGS
# ─────────────────────────────────────────────
BINARY_PATTERNS = [
    'IsFree', 'FreeVerAvail', 'PurchaseAvail', 'SubscriptionAvail',
    'ControllerSupport',
    'PlatformLinux', 'PlatformMac',
    'PCReqsHaveMin', 'PCReqsHaveRec',
    'LinuxReqsHaveMin', 'LinuxReqsHaveRec',
    'MacReqsHaveMin', 'MacReqsHaveRec',
    'CategorySinglePlayer', 'CategoryMultiplayer', 'CategoryCoop',
    'CategoryMMO', 'CategoryInAppPurchase', 'CategoryIncludeSrcSDK',
    'CategoryIncludeLevelEditor', 'CategoryVRSupport',
    'GenreIsNonGame', 'GenreIsIndie', 'GenreIsAction', 'GenreIsAdventure',
    'GenreIsCasual', 'GenreIsStrategy', 'GenreIsRPG', 'GenreIsSimulation',
    'GenreIsEarlyAccess', 'GenreIsFreeToPlay', 'GenreIsSports',
    'GenreIsRacing', 'GenreIsMassivelyMultiplayer',
    'is_effectively_free', 'has_metacritic',
    'has_website', 'has_support_email', 'has_support_url',
    'has_legal_notice', 'has_reviews_text',
    'has_pc_min_reqs', 'has_pc_rec_reqs',
    'has_linux_min_reqs', 'has_mac_min_reqs',
    'has_drm', 'has_ext_account',
    'discount_ratio',
]

CONTINUOUS_COLS = [
    'RequiredAge', 'DemoCount', 'DeveloperCount', 'DLCCount',
    'MovieCount', 'PackageCount', 'RecommendationCount', 'PublisherCount',
    'ScreenshotCount',
    'SteamSpyOwners', 'SteamSpyOwnersVariance',
    'SteamSpyPlayersEstimate', 'SteamSpyPlayersVariance',
    'AchievementCount', 'AchievementHighlightedCount',
    'PriceInitial', 'PriceFinal',
    'release_year', 'release_month', 'game_age_days',
    'num_languages',
    'about_length', 'short_length', 'detail_length',
    'SteamSpyOwners_log', 'SteamSpyOwnersVariance_log',
    'SteamSpyPlayersEstimate_log', 'SteamSpyPlayersVariance_log',
    'target_log',
]

CONTINUOUS_COLS = [c for c in CONTINUOUS_COLS if c in df.columns]
print(f"\n>>> Continuous columns to process ({len(CONTINUOUS_COLS)}):\n", CONTINUOUS_COLS)

df['price_per_language'] = df['PriceFinal'] / (df['num_languages'] + 1)
df['metacritic_x_age']   = df['has_metacritic'] * df['game_age_days']
df['owners_per_achievement'] = df['SteamSpyOwners_log'] / (df['AchievementCount'] + 1)

df['dlc_x_owners'] = np.log1p(df['DLCCount']) * df['SteamSpyOwners_log']
df['movie_x_owners']  = df['MovieCount'] * df['SteamSpyOwners_log']
CONTINUOUS_COLS += [
    'price_per_language',
    'metacritic_x_age',
    'owners_per_achievement',
    'dlc_x_owners',
    'movie_x_owners',
]

CONTINUOUS_COLS = [c for c in CONTINUOUS_COLS if c in df.columns]
# ─────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────
X = df.drop(columns=['RecommendationCount', 'target_log'])
y = df['target_log']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)

# Fill numeric NaNs using train medians
num_cols_all = X_train.select_dtypes(include=np.number).columns
X_train[num_cols_all] = X_train[num_cols_all].fillna(X_train[num_cols_all].median())
# X_val[num_cols_all]   = X_val[num_cols_all].fillna(X_train[num_cols_all].median())
X_test[num_cols_all]  = X_test[num_cols_all].fillna(X_train[num_cols_all].median())

print("\nRemaining NaNs after median fill:",
      X_train.isnull().sum().sum(),
    #   y_train.isnull().sum().sum(),
      X_test.isnull().sum().sum())

cont_feat_cols = [c for c in CONTINUOUS_COLS if c in X_train.columns]
# ─────────────────────────────────────────────────────────────────
# HELPER: grid of before/after histograms
# ─────────────────────────────────────────────────────────────────
PLOT_COLS_SAMPLE = [
    'Metacritic', 'MovieCount',
    'SteamSpyOwners', 'AchievementCount', 'PriceInitial',
    'game_age_days', 'about_length', 'detail_length',
    'SteamSpyOwners_log', 'SteamSpyPlayersEstimate_log', 'release_year','game_age_days',
]


PLOT_COLS_SAMPLE = [c for c in PLOT_COLS_SAMPLE if c in cont_feat_cols]

def plot_distributions(before_df, after_df, cols, title, filename,
                       before_label='Before', after_label='After',
                       color_before='#4C8EDA', color_after='#E8593C'):
    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows * 2, ncols,
                             figsize=(ncols * 4.5, nrows * 5),
                             facecolor='#F8F7F4')
    fig.suptitle(title, fontsize=16, fontweight='600', y=1.01, color='#2C2C2A')

    for i, col in enumerate(cols):
        row_before = (i // ncols) * 2
        row_after  = row_before + 1
        col_idx    = i % ncols

        ax_b = axes[row_before][col_idx]
        ax_a = axes[row_after][col_idx]

        # Before
        vals_b = before_df[col].dropna()
        ax_b.hist(vals_b, bins=50, color=color_before, alpha=0.85, edgecolor='white', linewidth=0.3)
        ax_b.set_title(col, fontsize=9, fontweight='500', color='#2C2C2A', pad=4)
        ax_b.set_ylabel(before_label, fontsize=8, color='#5F5E5A')
        ax_b.tick_params(labelsize=7)
        ax_b.set_facecolor('#F0EFE8')
        _add_stats(ax_b, vals_b, color_before)

        # After
        vals_a = after_df[col].dropna()
        ax_a.hist(vals_a, bins=50, color=color_after, alpha=0.85, edgecolor='white', linewidth=0.3)
        ax_a.set_ylabel(after_label, fontsize=8, color='#5F5E5A')
        ax_a.tick_params(labelsize=7)
        ax_a.set_facecolor('#F0EFE8')
        _add_stats(ax_a, vals_a, color_after)

    # Hide unused axes
    total_slots = nrows * 2 * ncols
    for j in range(n, (nrows) * ncols):
        r_b = (j // ncols) * 2
        r_a = r_b + 1
        c_j = j % ncols
        axes[r_b][c_j].set_visible(False)
        axes[r_a][c_j].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
    plt.close()
    print(f"  Saved: {filename}")


def _add_stats(ax, vals, color):
    mu, med = vals.mean(), vals.median()
    ax.axvline(mu,  color='#2C2C2A', linewidth=1.2, linestyle='--', alpha=0.7, label=f'μ={mu:.2f}')
    ax.axvline(med, color='#888780', linewidth=1.0, linestyle=':',  alpha=0.7, label=f'med={med:.2f}')
    ax.legend(fontsize=6.5, framealpha=0.6, loc='upper right')


def plot_boxplots(before_df, after_df, cols, title, filename):
    """Side-by-side box plots before / after for each column."""
    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 3.5),
                             facecolor='#F8F7F4')
    fig.suptitle(title, fontsize=15, fontweight='600', y=1.01, color='#2C2C2A')
    axes = axes.flatten()

    bp_props = dict(patch_artist=True, notch=False, vert=True,
                    medianprops=dict(color='#2C2C2A', linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    flierprops=dict(marker='.', markersize=2, alpha=0.4))

    for i, col in enumerate(cols):
        ax = axes[i]
        ax.set_facecolor('#F0EFE8')
        vb = before_df[col].dropna().values
        va = after_df[col].dropna().values
        bp = ax.boxplot([vb, va], labels=['Before', 'After'], **bp_props)
        bp['boxes'][0].set_facecolor('#4C8EDA')
        bp['boxes'][0].set_alpha(0.75)
        bp['boxes'][1].set_facecolor('#E8593C')
        bp['boxes'][1].set_alpha(0.75)
        ax.set_title(col, fontsize=9, fontweight='500', color='#2C2C2A', pad=4)
        ax.tick_params(labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
    plt.close()
    print(f"  Saved: {filename}")



# ─────────────────────────────────────────────
# STEP 1 — OUTLIER CAPPING via IQR
# ─────────────────────────────────────────────
# RequiredAge,DemoCount,DeveloperCount,DLCCountPackageCount,PublisherCount
NO_IQR_COLS = [
    'RequiredAge',
    'DemoCount',
    'DeveloperCount',
    'DLCCount',
    'PackageCount',
    'PublisherCount'
]
print("\n" + "="*60)

print("STEP 1: IQR Outlier Capping")
print("="*60)

iqr_bounds = {}
X_train_raw = X_train.copy()  # for before/after comparison plots
for col in cont_feat_cols:

    if col in NO_IQR_COLS:
        print(f"  ⏭️ Skipping IQR for {col} (log transform instead)")
        continue

    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1

    # optional safety
    if IQR == 0:
        print(f"  ⚠️ Skipping {col} (IQR=0)")
        continue

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    n_before = ((X_train[col] < lower) | (X_train[col] > upper)).sum()

    X_train[col] = X_train[col].clip(lower, upper)
    # X_val[col]   = X_val[col].clip(lower, upper)
    X_test[col]  = X_test[col].clip(lower, upper)

    print(f"  {col:45s}  bounds=[{lower:10.2f}, {upper:10.2f}]  clipped={n_before}")


for col in NO_IQR_COLS:
    X_train[col] = np.log1p(X_train[col])
    # X_val[col]   = np.log1p(X_val[col])
    X_test[col]  = np.log1p(X_test[col])

    print(f"  🔄 Log transformed {col}")

# remove constant/low variance columns after outlier capping (if any)
threshold = 0.001  

low_var_cols = []

for col in X_train.columns:
    var = X_train[col].var()
    
    if var < threshold:
        low_var_cols.append(col)

print("Low variance columns:", low_var_cols)


#X_train = X_train.drop(columns=low_var_cols)
#X_val   = X_val.drop(columns=low_var_cols)
#X_test  = X_test.drop(columns=low_var_cols)

# Plot: distributions before vs after outlier capping
print("\nPlotting outlier capping distributions …")
plot_distributions(
    X_train_raw, X_train, PLOT_COLS_SAMPLE,
    title="Outlier Capping — Distribution: Before vs After",
    filename="./plots/outlier_distributions.png",
    before_label="Raw",
    after_label="IQR Capped",
)

# Plot: box plots before vs after outlier capping
print("Plotting outlier capping box plots …")
plot_boxplots(
    X_train_raw, X_train, PLOT_COLS_SAMPLE,
    title="Outlier Capping — Box Plots: Before vs After",
    filename="./plots/outlier_boxplots.png",
)
# ─────────────────────────────────────────────
# ISOLATION FOREST (comparison with IQR capping)
# ─────────────────────────────────────────────
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
outlier_mask = iso.fit_predict(X_train[cont_feat_cols]) == 1

print(f"IQR kept     : {len(X_train)} rows")
print(f"ISO kept     : {outlier_mask.sum()} rows")
print(f"ISO removed  : {(~outlier_mask).sum()} rows")

X_train_iso = X_train[outlier_mask].copy()
y_train_iso = y_train[outlier_mask].copy()
# ─────────────────────────────────────────────
# STEP 2 — STANDARD SCALING
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Standard Scaling (continuous columns only)")
print("="*60)
X_train_precap = X_train.copy()  # for before/after comparison plots
scaler = StandardScaler()
X_train[cont_feat_cols] = scaler.fit_transform(X_train[cont_feat_cols])
# X_val[cont_feat_cols]   = scaler.transform(X_val[cont_feat_cols])
X_test[cont_feat_cols]  = scaler.transform(X_test[cont_feat_cols])

print(f"\nScaled {len(cont_feat_cols)} continuous columns.")
print("Binary/flag columns left untouched:")
binary_present = [c for c in BINARY_PATTERNS if c in X_train.columns]
print("Binary/flag columns left untouched:", binary_present)
scaled_stats = X_train[cont_feat_cols].describe().T[['mean', 'std']].round(4)
print("\nPost-scaling stats (should be ~mean=0, std=1):")
print(scaled_stats.to_string())

# Plot: distributions before vs after scaling
print("\nPlotting scaling distributions …")
plot_distributions(
    X_train_precap, X_train, PLOT_COLS_SAMPLE,
    title="Standard Scaling — Distribution: Before vs After",
    filename="./plots/scaling_distributions.png",
    before_label="IQR Capped",
    after_label="Standardized",
    color_before='#1D9E75',
    color_after='#7F77DD',
)

# Plot: box plots before vs after scaling
print("Plotting scaling box plots …")
plot_boxplots(
    X_train_precap, X_train, PLOT_COLS_SAMPLE,
    title="Standard Scaling — Box Plots: Before vs After",
    filename="./plots/scaling_boxplots.png",
)
# ─────────────────────────────────────────────
# STEP 3 — MUTUAL INFORMATION (Feature Selection)
# ─────────────────────────────────────────────
from sklearn.feature_selection import mutual_info_regression

mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
mi_df = pd.DataFrame({
    'feature': X_train.columns,
    'MI': mi_scores
}).sort_values('MI', ascending=False)

print("\nTop 20 features by MI:")
print(mi_df.head(20).to_string())

top20 = mi_df.head(20)
plt.figure(figsize=(10, 6), facecolor='#F8F7F4')
ax = plt.gca()
ax.set_facecolor('#F0EFE8')
ax.barh(top20['feature'][::-1], top20['MI'][::-1],
        color='#7F77DD', alpha=0.85)
ax.set_title('Top 20 Features — Mutual Information with Target',
             fontsize=11, fontweight='500', color='#2C2C2A')
ax.set_xlabel('MI Score', fontsize=9)
ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig('./plots/mutual_information.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/mutual_information.png")

TOP_FEATURES = mi_df.head(5)['feature'].tolist()
print("\nTop 5 for Polynomial Features:", TOP_FEATURES)
# ─────────────────────────────────────────────────────────────────
# SAVE PROCESSED DATA
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SAVING PROCESSED DATA")
print("="*60)

os.makedirs('./data/processed', exist_ok=True)

# Combine X and y back together for saving
train_df = X_train.copy()
train_df['target_log'] = y_train.values

# val_df = X_val.copy()
# val_df['target_log'] = y_val.values

test_df = X_test.copy()
test_df['target_log'] = y_test.values

train_df.to_csv('./data/processed/train.csv', index=False)
# val_df.to_csv('./data/processed/val.csv', index=False)
test_df.to_csv('./data/processed/test.csv', index=False)

print(f"  Saved: ./data/processed/train.csv  → shape {train_df.shape}")
# print(f"  Saved: ./data/processed/val.csv    → shape {val_df.shape}")
print(f"  Saved: ./data/processed/test.csv   → shape {test_df.shape}")
"""
SCALER SAVE PATCH
=================
Add this to your preprocess.py AFTER you fit the StandardScaler
(after the line: X_train[cont_feat_cols] = scaler.fit_transform(X_train[cont_feat_cols]))

This saves the scaler so predict_page.py can load and apply it at inference time.

The scaler must be fitted on the SAME continuous columns that you scaled during training.
predict_page.py will load scaler.pkl and call scaler.transform() on the matching columns
of the new game's feature vector.
"""

import joblib, os

# ─── PASTE THIS BLOCK into preprocess.py right after scaler.fit_transform() ───

# scaler is already defined as StandardScaler() and fitted on cont_feat_cols
# cont_feat_cols is the list of continuous columns that were scaled

os.makedirs('./models', exist_ok=True)
joblib.dump(scaler, './models/scaler.pkl')
print("Saved scaler → ./models/scaler.pkl")


# Also save the column names the scaler was fitted on, embedded directly in the scaler:
# sklearn's StandardScaler stores feature_names_in_ automatically when you pass a DataFrame.
# Make sure you call:
#   scaler.fit_transform(X_train[cont_feat_cols])          # X_train must be a DataFrame
# NOT:
#   scaler.fit_transform(X_train[cont_feat_cols].values)   # .values strips column names!

# ─── HOW predict_page.py USES THE SCALER ──────────────────────────────────────
# At inference time, predict_page.py does:
#
#   scaler = joblib.load('./models/scaler.pkl')
#   scaler_cols = [c for c in scaler.feature_names_in_ if c in feat_df.columns]
#   feat_df[scaler_cols] = scaler.transform(feat_df[scaler_cols])
#
# This applies the exact same mean/std from training to the new game's features.
# Any feature the scaler doesn't know about is left untouched (e.g., binary flags, LSA).

# ─── QUICK VERIFICATION ───────────────────────────────────────────────────────
loaded = joblib.load('./models/scaler.pkl')
print(f"Scaler fitted on {len(loaded.feature_names_in_)} columns:")
print(list(loaded.feature_names_in_)[:10], "...")
# ─────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL DATASET SHAPES")
print("="*60)
print(f"X_train : {X_train.shape}")
# print(f"X_val   : {X_val.shape}")
print(f"X_test  : {X_test.shape}")
print(f"\nContinuous cols scaled   : {len(cont_feat_cols)}")
print(f"Binary/flag cols intact  : {len(binary_present)}")
print(f"Text feature columns     : 0 (all dropped)")
print("\nDone. Processed files saved:")
print("  ./data/processed/train.csv")
print("  ./data/processed/test.csv")