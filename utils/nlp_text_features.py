import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import re
import os
import warnings
import pickle
import joblib
warnings.filterwarnings('ignore')

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD          # LSA on TF-IDF matrix
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, save_npz, load_npz
import scipy.sparse as sp

# Download required NLTK data
for pkg in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'omw-1.4', 'punkt_tab']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv('./data/raw/train_data.csv')

# ─── keep text columns alive before feature engineering drops them ───
TEXT_COLS = {
    'about':   'AboutText',
    'short':   'ShortDescrip',
    'detail':  'DetailedDescrip',
    'reviews': 'Reviews',
    'name':    'ResponseName',
}

for key, col in TEXT_COLS.items():
    if col not in df.columns:
        TEXT_COLS[key] = None

TEXT_COLS = {k: v for k, v in TEXT_COLS.items() if v is not None}
print("Text columns found:", list(TEXT_COLS.values()))

# Fill NaN in text columns
for col in TEXT_COLS.values():
    df[col] = df[col].fillna('')

# Keep a clean backup for NLP (before any downstream drops)
text_df = df[list(TEXT_COLS.values())].copy()
text_df.index = df.index

# ─────────────────────────────────────────────────────────────────
# STEP 1 — RAW TEXT STATISTICS (before any cleaning)
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Raw Text Statistics")
print("="*60)

raw_stats = {}
for key, col in TEXT_COLS.items():
    vals = text_df[col]
    raw_stats[key] = {
        'char_len':  vals.apply(len),
        'word_count': vals.apply(lambda x: len(x.split())),
        'has_content': (vals.str.len() > 10).sum(),
        'empty': (vals.str.len() <= 10).sum(),
    }
    print(f"\n  [{col}]")
    print(f"    Non-empty rows : {raw_stats[key]['has_content']:,}")
    print(f"    Empty rows     : {raw_stats[key]['empty']:,}")
    print(f"    Avg word count : {raw_stats[key]['word_count'].mean():.1f}")
    print(f"    Max word count : {raw_stats[key]['word_count'].max():,}")

# ── Plot 1: raw word count distributions ─────────────────────────
fig, axes = plt.subplots(1, len(TEXT_COLS), figsize=(5 * len(TEXT_COLS), 4),
                          facecolor='#F8F7F4')
if len(TEXT_COLS) == 1:
    axes = [axes]
COLORS = ['#4C8EDA', '#E8593C', '#1D9E75', '#7F77DD', '#EF9F27']
for i, (key, col) in enumerate(TEXT_COLS.items()):
    ax = axes[i]
    ax.set_facecolor('#F0EFE8')
    wc = raw_stats[key]['word_count']
    ax.hist(wc.clip(0, wc.quantile(0.98)), bins=60,
            color=COLORS[i % len(COLORS)], alpha=0.85,
            edgecolor='white', linewidth=0.3)
    ax.set_title(col, fontsize=9, fontweight='500', color='#2C2C2A')
    ax.set_xlabel('Word count (98th pct cap)', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)
    ax.tick_params(labelsize=7)
    mu = wc.mean()
    ax.axvline(mu, color='#2C2C2A', linewidth=1.2, linestyle='--',
               label=f'mean={mu:.0f}')
    ax.legend(fontsize=7)

plt.suptitle("Raw Text — Word Count Distributions", fontsize=13,
             fontweight='600', color='#2C2C2A', y=1.02)
plt.tight_layout()
plt.savefig('./plot_nlp_01_raw_word_counts.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("\nSaved: plot_nlp_01_raw_word_counts.png")

# ─────────────────────────────────────────────────────────────────
# STEP 2 — TEXT CLEANING
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Text Cleaning")
print("="*60)

lemmatizer  = WordNetLemmatizer()
STOP_WORDS  = set(stopwords.words('english'))

# Extra domain-specific stopwords for game descriptions
DOMAIN_STOP = {
    'game', 'games', 'play', 'player', 'players', 'feature', 'features',
    'include', 'includes', 'new', 'get', 'also', 
     'available', 'download', 'update', 'version',
    'support', 'use', 'using', 'system', 'may', 'will', 'can',
}
STOP_WORDS.update(DOMAIN_STOP)

HTML_TAG_RE  = re.compile(r'<[^>]+>')
URL_RE       = re.compile(r'http\S+|www\.\S+')
PUNCT_RE     = re.compile(r'[^a-zA-Z\s]')
SPACE_RE     = re.compile(r'\s+')

def clean_text(text: str) -> str:
    """Full cleaning pipeline: HTML → URLs → lowercase → punct → stopwords → lemmatize."""
    text = HTML_TAG_RE.sub(' ', text)        # strip HTML tags
    text = URL_RE.sub(' ', text)             # strip URLs
    text = text.lower()                      # lowercase
    text = PUNCT_RE.sub(' ', text)           # remove punctuation / numbers
    text = SPACE_RE.sub(' ', text).strip()   # normalise whitespace

    # tokenise → remove stopwords → lemmatise
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


cleaned = {}
for key, col in TEXT_COLS.items():
    print(f"  Cleaning {col} …", end=' ', flush=True)
    cleaned[key] = text_df[col].apply(clean_text)
    print(f"done  (avg tokens after: {cleaned[key].apply(lambda x: len(x.split())).mean():.1f})")

# ── Plot 2: before vs after cleaning word count ───────────────────
fig, axes = plt.subplots(2, len(TEXT_COLS),
                          figsize=(5 * len(TEXT_COLS), 7),
                          facecolor='#F8F7F4')
if len(TEXT_COLS) == 1:
    axes = axes.reshape(2, 1)

for i, (key, col) in enumerate(TEXT_COLS.items()):
    before_wc = raw_stats[key]['word_count']
    after_wc  = cleaned[key].apply(lambda x: len(x.split()))
    cap = before_wc.quantile(0.98)

    ax_b = axes[0][i]
    ax_a = axes[1][i]

    ax_b.set_facecolor('#F0EFE8')
    ax_b.hist(before_wc.clip(0, cap), bins=60, color='#4C8EDA',
              alpha=0.85, edgecolor='white', linewidth=0.3)
    ax_b.set_title(col, fontsize=9, fontweight='500', color='#2C2C2A')
    ax_b.set_ylabel('Raw', fontsize=8, color='#4C8EDA')
    ax_b.tick_params(labelsize=7)
    ax_b.axvline(before_wc.mean(), color='#2C2C2A', linewidth=1, linestyle='--')

    ax_a.set_facecolor('#F0EFE8')
    ax_a.hist(after_wc.clip(0, cap * 0.6), bins=60, color='#E8593C',
              alpha=0.85, edgecolor='white', linewidth=0.3)
    ax_a.set_ylabel('Cleaned', fontsize=8, color='#E8593C')
    ax_a.tick_params(labelsize=7)
    ax_a.axvline(after_wc.mean(), color='#2C2C2A', linewidth=1, linestyle='--')

plt.suptitle("Text Cleaning — Word Count Before vs After",
             fontsize=13, fontweight='600', color='#2C2C2A', y=1.01)
plt.tight_layout()
plt.savefig('./plot_nlp_02_cleaning_before_after.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: plot_nlp_02_cleaning_before_after.png")

# ─────────────────────────────────────────────────────────────────
# STEP 4 — TF-IDF EXTRACTION
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: TF-IDF Feature Extraction")
print("="*60)

# TF-IDF config per field  (max_features tuned to field importance)
TFIDF_CONFIG = {
    'about':   dict(max_features=200, ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True),
    'detail':  dict(max_features=300, ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True),
    'short':   dict(max_features=100, ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True),
    'reviews': dict(max_features=100, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True),
    'name':    dict(max_features=50,  ngram_range=(1, 1), min_df=2, max_df=0.90, sublinear_tf=True),
}
TFIDF_CONFIG = {k: v for k, v in TFIDF_CONFIG.items() if k in cleaned}

# ── Train/test split indices ──────────────────────────────────────
idx = np.arange(len(df))
idx_temp, idx_test = train_test_split(idx, test_size=0.15, random_state=42)
idx_train, idx_val = train_test_split(idx_temp, test_size=0.1765, random_state=42)

tfidf_train_parts = []
tfidf_val_parts   = []
tfidf_test_parts  = []
tfidf_vectorizers = {}
tfidf_feature_names = {}

for key, cfg in TFIDF_CONFIG.items():
    texts = cleaned[key].values
    train_texts = texts[idx_train]
    val_texts   = texts[idx_val]
    test_texts  = texts[idx_test]

    vec = TfidfVectorizer(**cfg)
    X_tr = vec.fit_transform(train_texts)
    X_v  = vec.transform(val_texts)
    X_te = vec.transform(test_texts)

    tfidf_vectorizers[key]    = vec
    tfidf_feature_names[key]  = vec.get_feature_names_out()
    tfidf_train_parts.append(X_tr)
    tfidf_val_parts.append(X_v)
    tfidf_test_parts.append(X_te)

    print(f"  [{key}]  vocab={len(vec.vocabulary_):,}  "
          f"matrix={X_tr.shape}  nnz={X_tr.nnz:,}")

# Stack all TF-IDF matrices horizontally
tfidf_train = sp.hstack(tfidf_train_parts, format='csr')
tfidf_val   = sp.hstack(tfidf_val_parts,   format='csr')
tfidf_test  = sp.hstack(tfidf_test_parts,  format='csr')
print(f"\n  Combined TF-IDF train: {tfidf_train.shape}")

# ── Plot 4: TF-IDF sparsity ───────────────────────────────────────
fig, axes = plt.subplots(1, len(TFIDF_CONFIG), figsize=(5 * len(TFIDF_CONFIG), 4),
                          facecolor='#F8F7F4')
if len(TFIDF_CONFIG) == 1:
    axes = [axes]

for i, (key, cfg) in enumerate(TFIDF_CONFIG.items()):
    ax = axes[i]
    ax.set_facecolor('#F0EFE8')
    mat = tfidf_train_parts[i]
    # Plot mean TF-IDF score of top-40 terms
    mean_scores = np.asarray(mat.mean(axis=0)).flatten()
    vocab       = tfidf_vectorizers[key].get_feature_names_out()
    top40_idx   = mean_scores.argsort()[-40:][::-1]
    top40_terms = vocab[top40_idx]
    top40_scores = mean_scores[top40_idx]
    ax.barh(range(40), top40_scores[::-1], color=COLORS[i % len(COLORS)],
            alpha=0.85, height=0.75)
    ax.set_yticks(range(40))
    ax.set_yticklabels(top40_terms[::-1], fontsize=6.5)
    ax.set_title(f'{TEXT_COLS[key]}\nTop-40 terms by mean TF-IDF',
                 fontsize=8.5, fontweight='500', color='#2C2C2A')
    ax.tick_params(axis='x', labelsize=7)

plt.suptitle("TF-IDF — Top Terms per Text Field",
             fontsize=13, fontweight='600', color='#2C2C2A', y=1.02)
plt.tight_layout()
plt.savefig('./plot_nlp_04_tfidf_top_terms.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: plot_nlp_04_tfidf_top_terms.png")

# ── Plot 5: TF-IDF sparsity heatmap (random 200 docs × 200 terms) ─
fig, axes = plt.subplots(1, min(3, len(TFIDF_CONFIG)),
                          figsize=(6 * min(3, len(TFIDF_CONFIG)), 5),
                          facecolor='#F8F7F4')
if len(TFIDF_CONFIG) == 1:
    axes = [axes]

for i, key in enumerate(list(TFIDF_CONFIG.keys())[:3]):
    ax = axes[i]
    ax.set_facecolor('#F0EFE8')
    mat   = tfidf_train_parts[i]
    n_doc = min(200, mat.shape[0])
    n_ter = min(200, mat.shape[1])
    sample = mat[:n_doc, :n_ter].toarray()
    im = ax.imshow(sample, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_title(f'{TEXT_COLS[key]}\n(200 docs × 200 terms)',
                 fontsize=9, fontweight='500', color='#2C2C2A')
    ax.set_xlabel('Term index', fontsize=8)
    ax.set_ylabel('Document index', fontsize=8)
    ax.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax, shrink=0.7)

plt.suptitle("TF-IDF Sparsity Pattern (non-zero = coloured)",
             fontsize=12, fontweight='600', color='#2C2C2A', y=1.01)
plt.tight_layout()
plt.savefig('./plot_nlp_05_tfidf_sparsity.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: plot_nlp_05_tfidf_sparsity.png")

# ─────────────────────────────────────────────────────────────────
# STEP 5 — LSA (Latent Semantic Analysis = SVD on TF-IDF)
# Reduces high-dimensional TF-IDF → dense 50-dim semantic vectors
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: LSA (TruncatedSVD) for Semantic Compression")
print("="*60)

N_COMPONENTS = 50   # tune: 30–100 for most datasets

svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
lsa_train = svd.fit_transform(tfidf_train)
lsa_val   = svd.transform(tfidf_val)
lsa_test  = svd.transform(tfidf_test)

# Normalise LSA vectors (cosine-normalisation)
lsa_train = normalize(lsa_train)
lsa_val   = normalize(lsa_val)
lsa_test  = normalize(lsa_test)

explained = svd.explained_variance_ratio_.cumsum()
print(f"  LSA shape: {lsa_train.shape}")
print(f"  Explained variance @ {N_COMPONENTS} components: {explained[-1]*100:.1f}%")

lsa_col_names = [f'lsa_{i}' for i in range(N_COMPONENTS)]
lsa_train_df  = pd.DataFrame(lsa_train, columns=lsa_col_names, index=idx_train)
lsa_val_df    = pd.DataFrame(lsa_val,   columns=lsa_col_names, index=idx_val)
lsa_test_df   = pd.DataFrame(lsa_test,  columns=lsa_col_names, index=idx_test)

# ── Plot 6: Explained variance + top-component word loadings ─────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), facecolor='#F8F7F4')

# Scree plot
ax1.set_facecolor('#F0EFE8')
ax1.plot(range(1, N_COMPONENTS + 1),
         svd.explained_variance_ratio_ * 100,
         color='#4C8EDA', linewidth=1.8, marker='o', markersize=3)
ax1.fill_between(range(1, N_COMPONENTS + 1),
                 svd.explained_variance_ratio_ * 100,
                 alpha=0.2, color='#4C8EDA')
ax1.set_xlabel('LSA Component', fontsize=9)
ax1.set_ylabel('Explained Variance (%)', fontsize=9)
ax1.set_title('LSA Scree Plot', fontsize=11, fontweight='500', color='#2C2C2A')
ax1.tick_params(labelsize=8)

# Cumulative
ax2.set_facecolor('#F0EFE8')
ax2.plot(range(1, N_COMPONENTS + 1), explained * 100,
         color='#E8593C', linewidth=1.8)
ax2.fill_between(range(1, N_COMPONENTS + 1), explained * 100,
                 alpha=0.2, color='#E8593C')
ax2.axhline(80, color='#888780', linewidth=1, linestyle='--', label='80%')
ax2.axhline(90, color='#444441', linewidth=1, linestyle='--', label='90%')
ax2.legend(fontsize=8)
ax2.set_xlabel('LSA Component', fontsize=9)
ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=9)
ax2.set_title('LSA Cumulative Variance', fontsize=11, fontweight='500', color='#2C2C2A')
ax2.tick_params(labelsize=8)

plt.suptitle("LSA (TruncatedSVD) Variance Explained",
             fontsize=13, fontweight='600', color='#2C2C2A', y=1.02)
plt.tight_layout()
plt.savefig('./plot_nlp_06_lsa_variance.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: plot_nlp_06_lsa_variance.png")

# ── Plot 7: Top words per LSA component (first 6 components) ──────
all_feature_names = np.concatenate([tfidf_feature_names[k] for k in TFIDF_CONFIG])
n_comp_plot = min(6, N_COMPONENTS)
fig, axes = plt.subplots(2, 3, figsize=(15, 7), facecolor='#F8F7F4')
axes = axes.flatten()

for comp_i in range(n_comp_plot):
    ax = axes[comp_i]
    ax.set_facecolor('#F0EFE8')
    loading = svd.components_[comp_i]
    # Top positive
    top_pos = loading.argsort()[-15:][::-1]
    top_neg = loading.argsort()[:5]
    top_idx = np.concatenate([top_pos, top_neg])
    top_words  = all_feature_names[top_idx]
    top_scores = loading[top_idx]

    bar_colors = ['#E8593C' if s > 0 else '#4C8EDA' for s in top_scores]
    ax.barh(range(len(top_idx)), top_scores[::-1], color=bar_colors[::-1], alpha=0.85, height=0.75)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(top_words[::-1], fontsize=7)
    ax.set_title(f'LSA Component {comp_i + 1}', fontsize=9,
                 fontweight='500', color='#2C2C2A')
    ax.axvline(0, color='#2C2C2A', linewidth=0.8)
    ax.tick_params(axis='x', labelsize=7)

plt.suptitle("Top Words per LSA Component (red=positive, blue=negative)",
             fontsize=12, fontweight='600', color='#2C2C2A', y=1.02)
plt.tight_layout()
plt.savefig('./plot_nlp_07_lsa_components.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: plot_nlp_07_lsa_components.png")

# ─────────────────────────────────────────────────────────────────
# STEP 6 — COMBINE EVERYTHING:
#   LSA features → final NLP feature frame
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6: Assembling Final NLP Feature Sets")
print("="*60)

def build_split(idx_split, lsa_df):
    return lsa_df.loc[idx_split].reset_index(drop=True)

lsa_all_df = pd.concat([lsa_train_df, lsa_val_df, lsa_test_df]).sort_index()

nlp_train = build_split(idx_train,  lsa_all_df)
nlp_val   = build_split(idx_val,   lsa_all_df)
nlp_test  = build_split(idx_test,   lsa_all_df)

print(f"  NLP feature shape  — train : {nlp_train.shape}")
print(f"  NLP feature shape  — val   : {nlp_val.shape}")
print(f"  NLP feature shape  — test  : {nlp_test.shape}")

# ── Plot 8: LSA feature distribution comparison (train vs test) ───
fig, axes = plt.subplots(2, 5, figsize=(18, 6), facecolor='#F8F7F4')
axes = axes.flatten()
for i in range(10):
    col = f'lsa_{i}'
    ax  = axes[i]
    ax.set_facecolor('#F0EFE8')
    ax.hist(nlp_train[col], bins=50, color='#4C8EDA', alpha=0.65,
            label='Train', edgecolor='white', linewidth=0.2)
    ax.hist(nlp_test[col],  bins=50, color='#E8593C', alpha=0.65,
            label='Test',  edgecolor='white', linewidth=0.2)
    ax.set_title(col, fontsize=8, fontweight='500', color='#2C2C2A')
    ax.tick_params(labelsize=6.5)
    if i == 0:
        ax.legend(fontsize=7)

plt.suptitle("LSA Feature Distributions — Train vs Test",
             fontsize=13, fontweight='600', color='#2C2C2A', y=1.02)
plt.tight_layout()
plt.savefig('./plot_nlp_08_lsa_train_vs_test.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: plot_nlp_08_lsa_train_vs_test.png")

# ── Plot 9: Correlation of NLP features with target ───────────────
if 'RecommendationCount' in df.columns:
    target_series = np.log1p(df['RecommendationCount'])
    target_train  = target_series.iloc[idx_train].reset_index(drop=True)

    corrs_nlp = nlp_train.corrwith(target_train).abs().sort_values(ascending=False)
    top20 = corrs_nlp.head(20)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#F8F7F4')
    ax.set_facecolor('#F0EFE8')
    bar_colors = ['#7F77DD' if 'lsa' in c else '#1D9E75' for c in top20.index]
    ax.barh(range(len(top20)), top20.values[::-1],
            color=bar_colors[::-1], alpha=0.85, height=0.75)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index[::-1], fontsize=8)
    ax.set_xlabel('|Pearson correlation| with log(RecommendationCount)', fontsize=9)
    ax.set_title('Top 20 NLP Features by Correlation with Target\n'
                 '(purple = LSA)',
                 fontsize=10, fontweight='500', color='#2C2C2A')
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig('./plot_nlp_09_nlp_feature_correlations.png', dpi=130,
                bbox_inches='tight', facecolor='#F8F7F4')
    plt.close()
    print("Saved: plot_nlp_09_nlp_feature_correlations.png")

# ── Save outputs ──────────────────────────────────────────────────
nlp_train.to_csv('./data/processed/nlp_features_train.csv', index=False)
nlp_val.to_csv('./data/processed/nlp_features_val.csv',     index=False)
nlp_test.to_csv('./data/processed/nlp_features_test.csv',   index=False)



# Save
# joblib.dump(tfidf_vectorizers, './models/tfidf.pkl')
# joblib.dump(svd,               './models/lsa.pkl')

# Load later
# tfidf_vectorizers = joblib.load('./models/tfidf.pkl')
# svd               = joblib.load('./models/lsa.pkl')
# Also save sparse TF-IDF matrices for downstream use
# save_npz('./tfidf_train.npz', tfidf_train)
# save_npz('./tfidf_val.npz',   tfidf_val)
# save_npz('./tfidf_test.npz',  tfidf_test)

print("\n" + "="*60)
print("OUTPUTS SAVED")
print("="*60)
print("  nlp_features_train.csv  — dense NLP features (handcrafted + LSA)")
print("  nlp_features_val.csv")
print("  nlp_features_test.csv")
print("  tfidf_train.npz          — raw sparse TF-IDF matrices")
print("  tfidf_val.npz")
print("  tfidf_test.npz")
print()
print("PLOTS:")
for i in range(1, 9):
    fnames = {
        1: "plot_nlp_01_raw_word_counts.png          — raw word count distributions",
        2: "plot_nlp_02_cleaning_before_after.png    — word count before/after cleaning",
        3: "plot_nlp_04_tfidf_top_terms.png          — top TF-IDF terms per field",
        4: "plot_nlp_05_tfidf_sparsity.png           — TF-IDF sparsity heatmap",
        5: "plot_nlp_06_lsa_variance.png             — LSA explained variance (scree)",
        6: "plot_nlp_07_lsa_components.png           — top words per LSA component",
        7: "plot_nlp_08_lsa_train_vs_test.png        — LSA feature distributions",
        8: "plot_nlp_09_nlp_feature_correlations.png — NLP feature vs target correlation",
    }
    print(f"  {fnames[i]}")

print(f"\nTotal NLP features per split: {nlp_train.shape[1]}")
print(f"  LSA (dense) : {N_COMPONENTS}")
print(f"  TF-IDF (sparse, optional): {tfidf_train.shape[1]}")
