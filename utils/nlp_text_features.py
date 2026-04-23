import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

# Download required NLTK data
for pkg in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'omw-1.4', 'punkt_tab']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

COLORS = ['#4C8EDA', '#E8593C', '#1D9E75', '#7F77DD', '#EF9F27',
          '#F4845F', '#56B4E9', '#CC79A7', '#D55E00', '#009E73', '#F0E442']

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv('./data/raw/train_data.csv')

TEXT_COLS = {
    'about':            'AboutText',
    'short':            'ShortDescrip',
    'detail':           'DetailedDescrip',
    'reviews':          'Reviews',
    'name':             'ResponseName',
    'PCMinReqsText':    'PCMinReqsText',
    'PCRecReqsText':    'PCRecReqsText',
    'LinuxMinReqsText': 'LinuxMinReqsText',
    'LinuxRecReqsText': 'LinuxRecReqsText',
    'MacMinReqsText':   'MacMinReqsText',
    'MacRecReqsText':   'MacRecReqsText',
}

for key, col in TEXT_COLS.items():
    if col not in df.columns:
        TEXT_COLS[key] = None
TEXT_COLS = {k: v for k, v in TEXT_COLS.items() if v is not None}
print("Text columns found:", list(TEXT_COLS.values()))

for col in TEXT_COLS.values():
    df[col] = df[col].fillna('')

text_df = df[list(TEXT_COLS.values())].copy()
text_df.index = df.index

# ─────────────────────────────────────────────────────────────────
# STEP 1 — RAW TEXT STATISTICS + SPARSITY CHECK
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Raw Text Statistics")
print("="*60)

SPARSITY_THRESHOLD = 0.5

raw_stats    = {}
keys_to_drop = []
for key, col in TEXT_COLS.items():
    vals        = text_df[col]
    empty_ratio = (vals.str.len() <= 10).mean()
    raw_stats[key] = {
        'word_count':  vals.apply(lambda x: len(x.split())),
        'has_content': (vals.str.len() > 10).sum(),
        'empty':       (vals.str.len() <= 10).sum(),
        'empty_ratio': empty_ratio,
    }
    print(f"\n  [{col}]")
    print(f"    Non-empty rows : {raw_stats[key]['has_content']:,}")
    print(f"    Empty rows     : {raw_stats[key]['empty']:,}  ({empty_ratio:.1%})")
    print(f"    Avg word count : {raw_stats[key]['word_count'].mean():.1f}")
    if empty_ratio > SPARSITY_THRESHOLD:
        keys_to_drop.append(key)
        print(f"    --> DROPPING (>{SPARSITY_THRESHOLD:.0%} empty)")

for key in keys_to_drop:
    col = TEXT_COLS.pop(key)
    print(f"\nDropped sparse column: {col}")

# ── Plot 1: Raw word count distributions ─────────────────────────
fig, axes = plt.subplots(1, len(TEXT_COLS), figsize=(5 * len(TEXT_COLS), 4), facecolor='#F8F7F4')
if len(TEXT_COLS) == 1:
    axes = [axes]
for i, (key, col) in enumerate(TEXT_COLS.items()):
    ax = axes[i]
    ax.set_facecolor('#F0EFE8')
    wc = raw_stats[key]['word_count']
    ax.hist(wc.clip(0, wc.quantile(0.98)), bins=60,
            color=COLORS[i % len(COLORS)], alpha=0.85, edgecolor='white', linewidth=0.3)
    ax.set_title(col, fontsize=9, fontweight='500', color='#2C2C2A')
    ax.set_xlabel('Word count (98th pct cap)', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)
    ax.tick_params(labelsize=7)
    mu = wc.mean()
    ax.axvline(mu, color='#2C2C2A', linewidth=1.2, linestyle='--', label=f'mean={mu:.0f}')
    ax.legend(fontsize=7)
plt.suptitle("Raw Text — Word Count Distributions", fontsize=13, fontweight='600', color='#2C2C2A', y=1.02)
plt.tight_layout()
plt.savefig('./plot_nlp_01_raw_word_counts.png', dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("\nSaved: plot_nlp_01_raw_word_counts.png")

# ─────────────────────────────────────────────────────────────────
# STEP 2 — TEXT CLEANING
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Text Cleaning")
print("="*60)

lemmatizer = WordNetLemmatizer()
STOP_WORDS  = set(stopwords.words('english'))
HTML_TAG_RE = re.compile(r'<[^>]+>')
URL_RE      = re.compile(r'http\S+|www\.\S+')
PUNCT_RE    = re.compile(r'[^a-zA-Z\s]')
SPACE_RE    = re.compile(r'\s+')

def clean_text(text: str) -> str:
    text = HTML_TAG_RE.sub(' ', text)
    text = URL_RE.sub(' ', text)
    text = text.lower()
    text = PUNCT_RE.sub(' ', text)
    text = SPACE_RE.sub(' ', text).strip()
    tokens = [lemmatizer.lemmatize(t) for t in text.split()
              if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

def safe_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms

cleaned = {}
for key, col in TEXT_COLS.items():
    print(f"  Cleaning {col} ...", end=' ', flush=True)
    cleaned[key] = text_df[col].apply(clean_text)
    print(f"done  (avg tokens: {cleaned[key].apply(lambda x: len(x.split())).mean():.1f})")

# ── Plot 2: Word count before vs after cleaning ───────────────────
fig, axes = plt.subplots(2, len(TEXT_COLS), figsize=(5 * len(TEXT_COLS), 7), facecolor='#F8F7F4')
if len(TEXT_COLS) == 1:
    axes = axes.reshape(2, 1)
for i, (key, col) in enumerate(TEXT_COLS.items()):
    before_wc = raw_stats[key]['word_count']
    after_wc  = cleaned[key].apply(lambda x: len(x.split()))
    cap = before_wc.quantile(0.98)
    ax_b, ax_a = axes[0][i], axes[1][i]
    ax_b.set_facecolor('#F0EFE8')
    ax_b.hist(before_wc.clip(0, cap), bins=60, color='#4C8EDA', alpha=0.85, edgecolor='white', linewidth=0.3)
    ax_b.set_title(col, fontsize=9, fontweight='500', color='#2C2C2A')
    ax_b.set_ylabel('Raw', fontsize=8, color='#4C8EDA')
    ax_b.tick_params(labelsize=7)
    ax_b.axvline(before_wc.mean(), color='#2C2C2A', linewidth=1, linestyle='--')
    ax_a.set_facecolor('#F0EFE8')
    ax_a.hist(after_wc.clip(0, cap * 0.6), bins=60, color='#E8593C', alpha=0.85, edgecolor='white', linewidth=0.3)
    ax_a.set_ylabel('Cleaned', fontsize=8, color='#E8593C')
    ax_a.tick_params(labelsize=7)
    ax_a.axvline(after_wc.mean(), color='#2C2C2A', linewidth=1, linestyle='--')
plt.suptitle("Text Cleaning — Word Count Before vs After", fontsize=13, fontweight='600', color='#2C2C2A', y=1.01)
plt.tight_layout()
plt.savefig('./plot_nlp_02_cleaning_before_after.png', dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: plot_nlp_02_cleaning_before_after.png")

# ─────────────────────────────────────────────────────────────────
# STEP 3 — TF-IDF + LSA PER FIELD (15 components each)
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: TF-IDF + LSA per Field (15 components each)")
print("="*60)

TFIDF_CONFIG = {
    'about':            dict(max_features=500, ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True),
    'detail':           dict(max_features=500, ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True),
    'short':            dict(max_features=300, ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True),
    'reviews':          dict(max_features=300, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True),
    'name':             dict(max_features=100, ngram_range=(1, 1), min_df=2, max_df=0.90, sublinear_tf=True),
    'PCMinReqsText':    dict(max_features=200, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True),
    'PCRecReqsText':    dict(max_features=200, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True),
    'LinuxMinReqsText': dict(max_features=100, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True),
    'LinuxRecReqsText': dict(max_features=100, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True),
    'MacMinReqsText':   dict(max_features=100, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True),
    'MacRecReqsText':   dict(max_features=100, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True),
}
TFIDF_CONFIG = {k: v for k, v in TFIDF_CONFIG.items() if k in cleaned}

N_COMPONENTS = 15

idx = np.arange(len(df))
idx_train, idx_test = train_test_split(idx, test_size=0.15, random_state=42)

tfidf_vectorizers = {}
svd_models        = {}
tfidf_train_mats  = {}   # store fitted train matrices for plots
lsa_train_parts   = []
lsa_test_parts    = []

for key, cfg in TFIDF_CONFIG.items():
    texts       = cleaned[key].values
    train_texts = texts[idx_train]
    test_texts  = texts[idx_test]

    # TF-IDF
    vec  = TfidfVectorizer(**cfg)
    X_tr = vec.fit_transform(train_texts)
    X_te = vec.transform(test_texts)
    tfidf_vectorizers[key] = vec
    tfidf_train_mats[key]  = X_tr          # keep for plotting

    # LSA (TruncatedSVD)
    svd_field = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    lsa_tr    = svd_field.fit_transform(X_tr)
    lsa_te    = svd_field.transform(X_te)
    svd_models[key] = svd_field

    lsa_tr = safe_normalize(lsa_tr)
    lsa_te = safe_normalize(lsa_te)

    explained = svd_field.explained_variance_ratio_.cumsum()[-1]
    print(f"  [{key}]  vocab={len(vec.vocabulary_):,}  "
          f"tfidf={X_tr.shape}  lsa=({X_tr.shape[0]},{N_COMPONENTS})  "
          f"var_explained={explained*100:.1f}%")

    col_names = [f'lsa_{key}_{j}' for j in range(N_COMPONENTS)]
    lsa_train_parts.append(pd.DataFrame(lsa_tr, columns=col_names))
    lsa_test_parts.append(pd.DataFrame(lsa_te,  columns=col_names))

nlp_train = pd.concat(lsa_train_parts, axis=1)
nlp_test  = pd.concat(lsa_test_parts,  axis=1)
nlp_train.index = idx_train
nlp_test.index  = idx_test

print(f"\n  Total LSA features : {len(TFIDF_CONFIG)} fields × {N_COMPONENTS} = {nlp_train.shape[1]}")
print(f"  Train shape        : {nlp_train.shape}")
print(f"  Test shape         : {nlp_test.shape}")

# ── Plot 3: Explained variance per field ──────────────────────────
fig, ax = plt.subplots(figsize=(9, 4), facecolor='#F8F7F4')
ax.set_facecolor('#F0EFE8')
field_vars = [(k, svd_models[k].explained_variance_ratio_.cumsum()[-1] * 100)
              for k in TFIDF_CONFIG]
field_vars.sort(key=lambda x: x[1], reverse=True)
keys_sorted = [x[0] for x in field_vars]
vars_sorted = [x[1] for x in field_vars]
bars = ax.barh(range(len(keys_sorted)), vars_sorted,
               color=[COLORS[i % len(COLORS)] for i in range(len(keys_sorted))],
               alpha=0.85, height=0.6)
ax.set_yticks(range(len(keys_sorted)))
ax.set_yticklabels(keys_sorted, fontsize=9)
ax.set_xlabel(f'Cumulative Explained Variance (%) — {N_COMPONENTS} LSA components', fontsize=9)
ax.set_title('LSA Variance Explained per Field', fontsize=11, fontweight='500', color='#2C2C2A')
ax.axvline(80, color='#888780', linewidth=1, linestyle='--', label='80%')
ax.legend(fontsize=8)
for bar, val in zip(bars, vars_sorted):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%', va='center', fontsize=8)
plt.tight_layout()
plt.savefig('./plot_nlp_03_lsa_variance_per_field.png', dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("\nSaved: plot_nlp_03_lsa_variance_per_field.png")

# ── Plot 4: Top TF-IDF terms per field ───────────────────────────
fig, axes = plt.subplots(1, len(TFIDF_CONFIG), figsize=(5 * len(TFIDF_CONFIG), 4), facecolor='#F8F7F4')
if len(TFIDF_CONFIG) == 1:
    axes = [axes]
for i, key in enumerate(TFIDF_CONFIG):
    ax          = axes[i]
    ax.set_facecolor('#F0EFE8')
    mat         = tfidf_train_mats[key]          # already fitted train matrix
    vocab       = tfidf_vectorizers[key].get_feature_names_out()
    mean_scores = np.asarray(mat.mean(axis=0)).flatten()
    top40_idx   = mean_scores.argsort()[-40:][::-1]
    ax.barh(range(40), mean_scores[top40_idx][::-1],
            color=COLORS[i % len(COLORS)], alpha=0.85, height=0.75)
    ax.set_yticks(range(40))
    ax.set_yticklabels(vocab[top40_idx][::-1], fontsize=6.5)
    ax.set_title(f'{TEXT_COLS[key]}\nTop-40 TF-IDF terms', fontsize=8.5, fontweight='500', color='#2C2C2A')
    ax.tick_params(axis='x', labelsize=7)
plt.suptitle("TF-IDF — Top Terms per Field", fontsize=13, fontweight='600', color='#2C2C2A', y=1.02)
plt.tight_layout()
plt.savefig('./plot_nlp_04_tfidf_top_terms.png', dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: plot_nlp_04_tfidf_top_terms.png")

# ── Plot 5: Top LSA component words per field (first 4 fields) ────
# Shows which words drive each field's LSA dimensions
n_fields_plot = min(4, len(TFIDF_CONFIG))
fields_to_plot = list(TFIDF_CONFIG.keys())[:n_fields_plot]

fig, axes = plt.subplots(n_fields_plot, 3, figsize=(15, 4 * n_fields_plot), facecolor='#F8F7F4')
if n_fields_plot == 1:
    axes = axes.reshape(1, 3)

for row, key in enumerate(fields_to_plot):
    vocab   = tfidf_vectorizers[key].get_feature_names_out()
    svd_f   = svd_models[key]
    for comp_i in range(3):           # show first 3 components per field
        ax      = axes[row][comp_i]
        ax.set_facecolor('#F0EFE8')
        loading = svd_f.components_[comp_i]
        top_pos = loading.argsort()[-10:][::-1]
        top_neg = loading.argsort()[:5]
        top_idx = np.concatenate([top_pos, top_neg])
        top_scores = loading[top_idx]
        bar_colors = ['#E8593C' if s > 0 else '#4C8EDA' for s in top_scores]
        ax.barh(range(len(top_idx)), top_scores[::-1],
                color=bar_colors[::-1], alpha=0.85, height=0.75)
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels(vocab[top_idx][::-1], fontsize=7)
        ax.set_title(f'{key} — LSA component {comp_i}',
                     fontsize=8.5, fontweight='500', color='#2C2C2A')
        ax.axvline(0, color='#2C2C2A', linewidth=0.8)
        ax.tick_params(axis='x', labelsize=7)

plt.suptitle("Top Words per LSA Component per Field (red=positive, blue=negative)",
             fontsize=12, fontweight='600', color='#2C2C2A', y=1.01)
plt.tight_layout()
plt.savefig('./plot_nlp_05_lsa_components_per_field.png', dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: plot_nlp_05_lsa_components_per_field.png")

# ── Plot 6: LSA feature distributions train vs test ───────────────
# First 2 components from each field
sample_cols = []
for key in TFIDF_CONFIG:
    sample_cols += [f'lsa_{key}_0', f'lsa_{key}_1']
sample_cols = sample_cols[:10]   # cap at 10 for readability

fig, axes = plt.subplots(2, 5, figsize=(18, 6), facecolor='#F8F7F4')
axes = axes.flatten()
for i, col in enumerate(sample_cols):
    ax = axes[i]
    ax.set_facecolor('#F0EFE8')
    ax.hist(nlp_train[col], bins=50, color='#4C8EDA', alpha=0.65,
            label='Train', edgecolor='white', linewidth=0.2)
    ax.hist(nlp_test[col],  bins=50, color='#E8593C', alpha=0.65,
            label='Test',  edgecolor='white', linewidth=0.2)
    ax.set_title(col, fontsize=7.5, fontweight='500', color='#2C2C2A')
    ax.tick_params(labelsize=6.5)
    if i == 0:
        ax.legend(fontsize=7)
plt.suptitle("LSA Feature Distributions — Train vs Test (first 2 components per field)",
             fontsize=12, fontweight='600', color='#2C2C2A', y=1.02)
plt.tight_layout()
plt.savefig('./plot_nlp_06_lsa_train_vs_test.png', dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: plot_nlp_06_lsa_train_vs_test.png")

# ── Plot 7: Top LSA features correlated with target ───────────────
if 'RecommendationCount' in df.columns:
    target_s = pd.Series(
        np.log1p(df['RecommendationCount'].iloc[idx_train]).values,
        index=nlp_train.index
    )
    corrs = nlp_train.corrwith(target_s).abs().sort_values(ascending=False)
    top20 = corrs.head(20)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor='#F8F7F4')
    ax.set_facecolor('#F0EFE8')
    # Color bars by field
    bar_colors = []
    for c in top20.index[::-1]:
        field = c.replace('lsa_', '').rsplit('_', 1)[0]
        field_idx = list(TFIDF_CONFIG.keys()).index(field) if field in TFIDF_CONFIG else 0
        bar_colors.append(COLORS[field_idx % len(COLORS)])
    ax.barh(range(len(top20)), top20.values[::-1], color=bar_colors, alpha=0.85, height=0.75)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index[::-1], fontsize=8)
    ax.set_xlabel('|Pearson correlation| with log(RecommendationCount)', fontsize=9)
    ax.set_title('Top 20 LSA Features by Correlation with Target\n(colour = field)',
                 fontsize=10, fontweight='500', color='#2C2C2A')
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig('./plot_nlp_07_lsa_feature_correlations.png', dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
    plt.close()
    print("Saved: plot_nlp_07_lsa_feature_correlations.png")

# ─────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────
nlp_train.to_csv('./data/processed/nlp_features_train.csv', index=False)
nlp_test.to_csv('./data/processed/nlp_features_test.csv',   index=False)

os.makedirs('./models', exist_ok=True)
joblib.dump(tfidf_vectorizers, './models/tfidf_vectorizers.pkl')
joblib.dump(svd_models,        './models/svd_models.pkl')

print("\n" + "="*60)
print("OUTPUTS SAVED")
print("="*60)
print(f"  Fields used      : {list(TFIDF_CONFIG.keys())}")
print(f"  Fields dropped   : {keys_to_drop}")
print(f"  Components/field : {N_COMPONENTS}")
print(f"  Total features   : {nlp_train.shape[1]}")
print(f"  Train shape      : {nlp_train.shape}")
print(f"  Test shape       : {nlp_test.shape}")
print()
print("PLOTS:")
for fname in [
    "plot_nlp_01_raw_word_counts.png          — raw word count distributions",
    "plot_nlp_02_cleaning_before_after.png    — word count before vs after cleaning",
    "plot_nlp_03_lsa_variance_per_field.png   — LSA variance explained per field",
    "plot_nlp_04_tfidf_top_terms.png          — top TF-IDF terms per field",
    "plot_nlp_05_lsa_components_per_field.png — top words per LSA component per field",
    "plot_nlp_06_lsa_train_vs_test.png        — LSA feature distributions train vs test",
    "plot_nlp_07_lsa_feature_correlations.png — top LSA features correlated with target",
]:
    print(f"  {fname}")

# ── Prediction helper ─────────────────────────────────────────────
#
# def predict_from_text(text_inputs: dict, model):
#     """
#     text_inputs = {'about': '...', 'reviews': '...', ...}
#     Missing keys are treated as empty string.
#     """
#     tfidf_vectorizers = joblib.load('./models/tfidf_vectorizers.pkl')
#     svd_models        = joblib.load('./models/svd_models.pkl')
#     parts = []
#     for key in tfidf_vectorizers:
#         raw          = text_inputs.get(key, '')
#         cleaned_text = clean_text(raw)
#         tfidf_vec    = tfidf_vectorizers[key].transform([cleaned_text])
#         lsa_vec      = svd_models[key].transform(tfidf_vec)
#         lsa_vec      = safe_normalize(lsa_vec)
#         col_names    = [f'lsa_{key}_{j}' for j in range(lsa_vec.shape[1])]
#         parts.append(pd.DataFrame(lsa_vec, columns=col_names))
#     nlp_features = pd.concat(parts, axis=1)
#     # combine with your numeric features then:
#     # return model.predict(final_features)
