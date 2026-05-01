import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

for pkg in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'omw-1.4', 'punkt_tab']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

COLORS = ['#4C8EDA', '#E8593C', '#1D9E75', '#7F77DD', '#EF9F27',
          '#F4845F', '#56B4E9', '#CC79A7']


#3 nlp text feature
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
    'MacMinReqsText':   'MacMinReqsText',
}
TEXT_COLS = {k: v for k, v in TEXT_COLS.items() if v in df.columns}
print(f"Text columns found: {list(TEXT_COLS.values())}")

for col in TEXT_COLS.values():
    df[col] = df[col].fillna('')

# load split indices from phase2_preprocessing.py
idx_train = np.load('./data/processed/idx_train.npy')
idx_test  = np.load('./data/processed/idx_test.npy')
print(f"Split loaded — train: {len(idx_train)}, test: {len(idx_test)}")

text_df = df[list(TEXT_COLS.values())].copy()
text_df.index = df.index

#1: Sparsity Check
print("\n" + "="*60)
print("STEP 1: Sparsity Check — Drop columns >50% empty")
print("="*60)

SPARSITY_THRESHOLD = 0.5
keys_to_drop = []
for key, col in TEXT_COLS.items():
    empty_ratio = (text_df[col].str.len() <= 10).mean()
    print(f"  [{col}]  empty: {empty_ratio:.1%}", end='')
    if empty_ratio > SPARSITY_THRESHOLD:
        keys_to_drop.append(key)
        print("  → DROPPING")
    else:
        print("  → KEEPING")

for key in keys_to_drop:
    TEXT_COLS.pop(key)
print(f"\nColumns kept for NLP: {list(TEXT_COLS.values())}")

#2: Text Cleaning
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
    avg_tokens = cleaned[key].apply(lambda x: len(x.split())).mean()
    print(f"done  (avg tokens after cleaning: {avg_tokens:.1f})")

# ── STEP 3: TF-IDF + LSA per text column ─────────────────────────
print("\n" + "="*60)
print("STEP 3: TF-IDF + LSA (fit on TRAIN only)")
print("="*60)

N_COMPONENTS = 30  # LSA components per text field

TFIDF_CONFIG = {
    key: dict(max_features=5000, ngram_range=(1, 2),
              sublinear_tf=True, min_df=3, max_df=0.95)
    for key in TEXT_COLS
}

tfidf_vectorizers = {}
svd_models        = {}
lsa_train_parts   = []
lsa_test_parts    = []

for key, col in TEXT_COLS.items():
    cfg = TFIDF_CONFIG[key]

    train_texts = cleaned[key].iloc[
        np.where(np.isin(text_df.index, idx_train))[0]]
    test_texts  = cleaned[key].iloc[
        np.where(np.isin(text_df.index, idx_test))[0]]

    # Fit TF-IDF on TRAIN only
    vec  = TfidfVectorizer(**cfg)
    X_tr = vec.fit_transform(train_texts)
    X_te = vec.transform(test_texts)         # transform test using train vocab
    tfidf_vectorizers[key] = vec

    # Fit SVD on TRAIN only
    svd_field = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    lsa_tr    = svd_field.fit_transform(X_tr)
    lsa_te    = svd_field.transform(X_te)    # transform test using train svd
    svd_models[key] = svd_field

    lsa_tr = safe_normalize(lsa_tr)
    lsa_te = safe_normalize(lsa_te)

    explained = svd_field.explained_variance_ratio_.cumsum()[-1]
    print(f"  [{key}]  vocab={len(vec.vocabulary_):,}  "
          f"lsa=({X_tr.shape[0]},{N_COMPONENTS})  "
          f"var_explained={explained*100:.1f}%")

    col_names = [f'lsa_{key}_{j}' for j in range(N_COMPONENTS)]
    lsa_train_parts.append(pd.DataFrame(lsa_tr, columns=col_names))
    lsa_test_parts.append(pd.DataFrame(lsa_te,  columns=col_names))

nlp_train = pd.concat(lsa_train_parts, axis=1)
nlp_test  = pd.concat(lsa_test_parts,  axis=1)

print(f"\nTotal NLP features: {len(TEXT_COLS)} fields × {N_COMPONENTS} = {nlp_train.shape[1]}")
print(f"NLP train shape   : {nlp_train.shape}")
print(f"NLP test shape    : {nlp_test.shape}")

# ── STEP 4: Correlation with GamePopularity (classification) ─────
# Use encoded target for correlation check
print("\n" + "="*60)
print("STEP 4: NLP Feature Relevance Check")
print("="*60)

if 'GamePopularity' in df.columns:
    from sklearn.preprocessing import LabelEncoder
    le_check = LabelEncoder()
    target_encoded = le_check.fit_transform(df['GamePopularity'])
    target_train = pd.Series(
        target_encoded[np.where(np.isin(df.index, idx_train))[0]],
        index=range(len(idx_train))
    )
    corrs = nlp_train.corrwith(target_train).abs().sort_values(ascending=False)
    top15 = corrs.head(15)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor='#F8F7F4')
    ax.set_facecolor('#F0EFE8')
    bar_colors = []
    for c in top15.index[::-1]:
        field = c.replace('lsa_', '').rsplit('_', 1)[0]
        field_idx = list(TEXT_COLS.keys()).index(field) if field in TEXT_COLS else 0
        bar_colors.append(COLORS[field_idx % len(COLORS)])
    ax.barh(range(len(top15)), top15.values[::-1],
            color=bar_colors, alpha=0.85, height=0.75)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15.index[::-1], fontsize=8)
    ax.set_xlabel('|Correlation| with GamePopularity (encoded)', fontsize=9)
    ax.set_title('Top 15 NLP (LSA) Features by Correlation with Target',
                 fontsize=10, fontweight='500', color='#2C2C2A')
    plt.tight_layout()
    plt.savefig('./plots/phase3_nlp_feature_correlations.png', dpi=130,
                bbox_inches='tight', facecolor='#F8F7F4')
    plt.close()
    print("Saved: plots/phase3_nlp_feature_correlations.png")

#5: Merge NLP with Preprocessed Features 
print("\n" + "="*60)
print("STEP 5: Merging NLP features with Preprocessed features")
print("="*60)

train_base = pd.read_csv('./data/processed/train.csv')
test_base  = pd.read_csv('./data/processed/test.csv')

# Reset indices before concat
nlp_train_reset = nlp_train.reset_index(drop=True)
nlp_test_reset  = nlp_test.reset_index(drop=True)
train_base_reset = train_base.reset_index(drop=True)
test_base_reset  = test_base.reset_index(drop=True)

train_final = pd.concat([train_base_reset, nlp_train_reset], axis=1)
test_final  = pd.concat([test_base_reset,  nlp_test_reset],  axis=1)

train_final.to_csv('./data/processed/train_with_nlp.csv', index=False)
test_final.to_csv('./data/processed/test_with_nlp.csv',   index=False)

print(f"✅ Merged train: {train_final.shape}")
print(f"✅ Merged test : {test_final.shape}")

# ── Save NLP models
os.makedirs('./models', exist_ok=True)
joblib.dump(tfidf_vectorizers, './models/tfidf_vectorizers.pkl')
joblib.dump(svd_models,        './models/svd_models.pkl')
joblib.dump(list(TEXT_COLS.keys()), './models/nlp_text_col_keys.pkl')

print("\n" + "="*60)
print("PHASE 3 COMPLETE — Saved:")
print("="*60)
print("  models/tfidf_vectorizers.pkl")
print("  models/svd_models.pkl")
print("  models/nlp_text_col_keys.pkl")
print("  data/processed/train_with_nlp.csv")
print("  data/processed/test_with_nlp.csv")
