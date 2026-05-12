"""
pages/predict_page.py  ──  SteamML · Predict New Game (Classification)
Pipeline:
  - Models: Logistic Regression, Random Forest, SVM, XGBoost, LightGBM, CatBoost
  - Target: GamePopularity (High=0, Low=1, Medium=2)
  - Feature selection: dominant-value + VarianceThreshold + RF importance (>= 0.001)
  - Preprocessing mirrors Preprocessing_m2.py + nlp.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, re, traceback, io, warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
MODELS_DIR  = "./trained_models"
PROC_DIR    = "./dataset/processed"
SEL_DIR     = "./dataset/selected"

TARGET      = "GamePopularity_enc"
DROP_COLS   = ["GamePopularity", "GamePopularity_enc"]
CLASS_NAMES = ["High", "Low", "Medium"]   # index 0, 1, 2
CLASS_COLORS = {
    "High"  : "#5bf6c8",
    "Low"   : "#f65b8d",
    "Medium": "#f6c85b",
}
VALID_CLASSES = {0, 1, 2}

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest"      : "random_forest.pkl",
    "SVM"                : "svm.pkl",
    "XGBoost"            : "xgboost.pkl",
    "LightGBM"           : "lightgbm.pkl",
    "CatBoost"           : "catboost.pkl",
}
MODEL_COLORS = {
    "Logistic Regression": "#5b8df6",
    "Random Forest"      : "#1D9E75",
    "SVM"                : "#f6c85b",
    "XGBoost"            : "#f65b8d",
    "LightGBM"           : "#a78bfa",
    "CatBoost"           : "#5bf6c8",
}
MODEL_DESC = {
    "Logistic Regression": "Linear classifier with L2 regularisation. Fast, interpretable baseline.",
    "Random Forest"      : "Ensemble of decorrelated decision trees. Robust, handles non-linearities.",
    "SVM"                : "Support Vector Machine with RBF kernel. Strong on high-dimensional data.",
    "XGBoost"            : "Gradient-boosted trees (XGBoost). High performance, regularised.",
    "LightGBM"           : "Gradient-boosted trees (LightGBM). Fast training, leaf-wise growth.",
    "CatBoost"           : "Gradient-boosted trees (CatBoost). Handles categoricals natively.",
}

# Feature-selection constants
DOMINANT_THRESHOLD = 0.95
VAR_THRESHOLD      = 0.001
IMP_THRESHOLD      = 0.001

# NLP field map
NLP_FIELD_MAP = {
    "about"           : "AboutText",
    "short"           : "ShortDescrip",
    "detail"          : "DetailedDescrip",
    "reviews"         : "Reviews",
    "name"            : "ResponseName",
    "PCMinReqsText"   : "PCMinReqsText",
    "PCRecReqsText"   : "PCRecReqsText",
    "LinuxMinReqsText": "LinuxMinReqsText",
    "MacMinReqsText"  : "MacMinReqsText",
}

CONT_FEAT_COLS_FOR_SCALING = [
    "Metacritic", "MovieCount",
    "SteamSpyOwners", "SteamSpyOwnersVariance",
    "SteamSpyPlayersEstimate", "SteamSpyPlayersVariance",
    "AchievementCount", "AchievementHighlightedCount",
    "PriceInitial", "PriceFinal",
    "release_year", "release_month", "game_age_days",
    "num_languages", "about_length", "short_length", "detail_length",
    "SteamSpyOwners_log", "SteamSpyOwnersVariance_log",
    "SteamSpyPlayersEstimate_log", "SteamSpyPlayersVariance_log",
    "price_per_language", "metacritic_x_age",
    "owners_per_achievement", "dlc_x_owners", "movie_x_owners",
    "RequiredAge", "DemoCount", "DeveloperCount",
    "DLCCount", "PackageCount", "PublisherCount",
    "ScreenshotCount",
    # classification-specific engineered features
    "platform_count", "genre_count", "category_count", "price_tier",
    "owners_tier", "achievement_tier", "players_tier",
]

RAW_COLS = [
    "QueryID","ResponseID","QueryName","ResponseName","ReleaseDate",
    "RequiredAge","DemoCount","DeveloperCount","DLCCount","Metacritic",
    "MovieCount","PackageCount","RecommendationCount","PublisherCount",
    "ScreenshotCount","SteamSpyOwners","SteamSpyOwnersVariance",
    "SteamSpyPlayersEstimate","SteamSpyPlayersVariance",
    "AchievementCount","AchievementHighlightedCount",
    "ControllerSupport","IsFree","FreeVerAvail","PurchaseAvail",
    "SubscriptionAvail","PlatformWindows","PlatformLinux","PlatformMac",
    "PCReqsHaveMin","PCReqsHaveRec","LinuxReqsHaveMin","LinuxReqsHaveRec",
    "MacReqsHaveMin","MacReqsHaveRec","CategorySinglePlayer",
    "CategoryMultiplayer","CategoryCoop","CategoryMMO",
    "CategoryInAppPurchase","CategoryIncludeSrcSDK",
    "CategoryIncludeLevelEditor","CategoryVRSupport",
    "GenreIsNonGame","GenreIsIndie","GenreIsAction","GenreIsAdventure",
    "GenreIsCasual","GenreIsStrategy","GenreIsRPG","GenreIsSimulation",
    "GenreIsEarlyAccess","GenreIsFreeToPlay","GenreIsSports",
    "GenreIsRacing","GenreIsMassivelyMultiplayer",
    "PriceCurrency","PriceInitial","PriceFinal",
    "SupportEmail","SupportURL","AboutText","Background","ShortDescrip",
    "DetailedDescrip","DRMNotice","ExtUserAcctNotice","HeaderImage",
    "LegalNotice","Reviews","SupportedLanguages","Website",
    "PCMinReqsText","PCRecReqsText","LinuxMinReqsText","LinuxRecReqsText",
    "MacMinReqsText","MacRecReqsText","GamePopularity",
]

FORM_NUMERIC = {
    "Metacritic"               : (0,   100,   0),
    "PriceInitial"             : (0.0, 200.0, 9.99),
    "PriceFinal"               : (0.0, 200.0, 9.99),
    "RequiredAge"              : (0,   18,    0),
    "DLCCount"                 : (0,   200,   0),
    "MovieCount"               : (0,   50,    2),
    "ScreenshotCount"          : (0,   100,   5),
    "AchievementCount"         : (0,   2000,  0),
    "RecommendationCount"      : (0,   500000, 1000),
    "PackageCount"             : (0,   50,    1),
    "DeveloperCount"           : (0,   20,    1),
    "PublisherCount"           : (0,   10,    1),
    "DemoCount"                : (0,   5,     0),
    "SteamSpyOwners"           : (0,   50_000_000, 10_000),
    "SteamSpyOwnersVariance"   : (0,   50_000_000, 5_000),
    "SteamSpyPlayersEstimate"  : (0,   50_000_000, 5_000),
    "SteamSpyPlayersVariance"  : (0,   50_000_000, 2_000),
    "AchievementHighlightedCount": (0, 200, 0),
}
FORM_BOOL = [
    "IsFree","FreeVerAvail","PurchaseAvail","SubscriptionAvail",
    "PlatformLinux","PlatformMac",
    "PCReqsHaveMin","PCReqsHaveRec",
    "CategorySinglePlayer","CategoryMultiplayer","CategoryCoop",
    "CategoryMMO","CategoryInAppPurchase","CategoryVRSupport",
    "GenreIsIndie","GenreIsAction","GenreIsAdventure","GenreIsCasual",
    "GenreIsStrategy","GenreIsRPG","GenreIsSimulation","GenreIsEarlyAccess",
    "GenreIsFreeToPlay","GenreIsSports","GenreIsRacing",
    "GenreIsMassivelyMultiplayer","ControllerSupport",
]
BINARY_FLAGS = [
    "IsFree","FreeVerAvail","PurchaseAvail","SubscriptionAvail",
    "ControllerSupport","PlatformLinux","PlatformMac",
    "PCReqsHaveMin","PCReqsHaveRec","LinuxReqsHaveMin","LinuxReqsHaveRec",
    "MacReqsHaveMin","MacReqsHaveRec","CategorySinglePlayer",
    "CategoryMultiplayer","CategoryCoop","CategoryMMO",
    "CategoryInAppPurchase","CategoryIncludeSrcSDK",
    "CategoryIncludeLevelEditor","CategoryVRSupport",
    "GenreIsNonGame","GenreIsIndie","GenreIsAction","GenreIsAdventure",
    "GenreIsCasual","GenreIsStrategy","GenreIsRPG","GenreIsSimulation",
    "GenreIsEarlyAccess","GenreIsFreeToPlay","GenreIsSports",
    "GenreIsRacing","GenreIsMassivelyMultiplayer",
]

# ─────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))
_HTML_RE    = re.compile(r"<[^>]+>")
_URL_RE     = re.compile(r"http\S+|www\.\S+")
_PUNCT_RE   = re.compile(r"[^a-zA-Z\s]")
_SPACE_RE   = re.compile(r"\s+")

def _clean_text(text: str) -> str:
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in text.split()
        if t not in _stop_words and len(t) > 2
    ]
    return " ".join(tokens)

def _safe_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms

# ─────────────────────────────────────────────────────────────────
# CACHED RESOURCE LOADERS
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def _load_nlp_models():
    tv = os.path.join(MODELS_DIR, "tfidf_vectorizers.pkl")
    sv = os.path.join(MODELS_DIR, "svd_models.pkl")
    if not (os.path.exists(tv) and os.path.exists(sv)):
        return None, None
    return joblib.load(tv), joblib.load(sv)

@st.cache_resource
def _load_scaler():
    sp = os.path.join(MODELS_DIR, "scaler.pkl")
    if not os.path.exists(sp):
        return None
    return joblib.load(sp)

@st.cache_resource
def _load_label_encoder():
    lp = os.path.join(MODELS_DIR, "label_encoder.pkl")
    if not os.path.exists(lp):
        return None
    return joblib.load(lp)

@st.cache_resource
def _load_model(name):
    return joblib.load(os.path.join(MODELS_DIR, MODEL_FILES[name]))

def _model_exists(name):
    return os.path.exists(os.path.join(MODELS_DIR, MODEL_FILES[name]))

def _get_feature_names(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None

def _align(model, X: pd.DataFrame) -> pd.DataFrame:
    expected = _get_feature_names(model)
    if expected is None:
        return X
    X = X.copy()
    for c in expected:
        if c not in X.columns:
            X[c] = 0.0
    return X[expected]

def _data_ready():
    return (
        os.path.exists(os.path.join(SEL_DIR, "train_selected.csv")) and
        os.path.exists(os.path.join(SEL_DIR, "test_selected.csv"))
    )

def _decode_class(enc_val):
    """Map encoded int → class name string."""
    mapping = {0: "High", 1: "Low", 2: "Medium"}
    return mapping.get(int(enc_val), str(enc_val))

# ─────────────────────────────────────────────────────────────────
# CORE PIPELINE: raw row dict → feature vector
# ─────────────────────────────────────────────────────────────────
def _preprocess_row(raw: dict, tfidf_vecs, svd_mods, scaler) -> pd.DataFrame:
    """Mirror of Preprocessing_m2.py + nlp.py for a single raw row."""

    def flt(k, default=0.0):
        v = raw.get(k, default)
        try:
            return float(v) if v not in (None, "", "nan", float("nan")) else default
        except Exception:
            return default

    def txt(k):
        v = raw.get(k, "")
        return str(v) if v not in (None, float("nan")) else ""

    # 1. Date features
    try:
        rd = pd.to_datetime(txt("ReleaseDate"), errors="coerce")
    except Exception:
        rd = pd.NaT

    release_year  = float(rd.year)  if pd.notna(rd) else 2020.0
    release_month = float(rd.month) if pd.notna(rd) else 6.0
    game_age_days = float((pd.Timestamp.today() - rd).days) if pd.notna(rd) else 1000.0

    # 2. Price / free features
    price_initial = flt("PriceInitial", 0.0)
    price_final   = flt("PriceFinal",   0.0)
    is_free       = flt("IsFree", 0.0)

    discount_ratio      = float(np.clip((price_initial - price_final) / (price_initial + 1e-9), 0, 1))
    is_effectively_free = float((price_initial == 0) or (is_free == 1))

    # 3. Metacritic / language / presence flags
    metacritic     = flt("Metacritic", 0.0)
    has_metacritic = float(metacritic > 0)

    lang_str      = txt("SupportedLanguages")
    num_languages = float(len([w for w in lang_str.split() if len(w) > 2]))

    has_website        = float(bool(txt("Website").strip()))
    has_support_email  = float(bool(txt("SupportEmail").strip()))
    has_support_url    = float(bool(txt("SupportURL").strip()))
    has_legal_notice   = float(len(txt("LegalNotice").strip())       > 1)
    has_reviews_text   = float(len(txt("Reviews").strip())           > 5)
    has_pc_min_reqs    = float(len(txt("PCMinReqsText").strip())     > 5)
    has_pc_rec_reqs    = float(len(txt("PCRecReqsText").strip())     > 5)
    has_linux_min_reqs = float(len(txt("LinuxMinReqsText").strip())  > 5)
    has_mac_min_reqs   = float(len(txt("MacMinReqsText").strip())    > 5)
    has_drm            = float(len(txt("DRMNotice").strip())         > 1)
    has_ext_account    = float(len(txt("ExtUserAcctNotice").strip()) > 1)

    about_length  = float(len(txt("AboutText")))
    short_length  = float(len(txt("ShortDescrip")))
    detail_length = float(len(txt("DetailedDescrip")))

    # 4. SteamSpy log transforms
    ss_own    = flt("SteamSpyOwners",          0)
    ss_ownv   = flt("SteamSpyOwnersVariance",  0)
    ss_play   = flt("SteamSpyPlayersEstimate", 0)
    ss_playv  = flt("SteamSpyPlayersVariance", 0)
    ss_own_log   = float(np.log1p(ss_own))
    ss_ownv_log  = float(np.log1p(ss_ownv))
    ss_play_log  = float(np.log1p(ss_play))
    ss_playv_log = float(np.log1p(ss_playv))

    # 5. NO_IQR log transforms
    req_age     = float(np.log1p(flt("RequiredAge",    0)))
    demo_cnt    = float(np.log1p(flt("DemoCount",      0)))
    dev_cnt     = float(np.log1p(flt("DeveloperCount", 1)))
    dlc_cnt_log = float(np.log1p(flt("DLCCount",       0)))
    pkg_cnt     = float(np.log1p(flt("PackageCount",   1)))
    pub_cnt     = float(np.log1p(flt("PublisherCount", 1)))

    ach_count   = flt("AchievementCount", 0)
    movie_count = flt("MovieCount",       0)

    # 6. Interaction features
    price_per_language     = price_final / (num_languages + 1)
    metacritic_x_age       = has_metacritic * game_age_days
    owners_per_achievement = ss_own_log / (ach_count + 1)
    dlc_x_owners           = dlc_cnt_log * ss_own_log
    movie_x_owners         = movie_count  * ss_own_log

    # 7. Tier / bin features — mirrors Preprocessing_m2.py exactly
    rec_count = flt("RecommendationCount", 0)

    # platform_count, genre_count, category_count  (sum of flag columns from raw input)
    genre_flags    = ["GenreIsNonGame","GenreIsIndie","GenreIsAction","GenreIsAdventure",
                      "GenreIsCasual","GenreIsStrategy","GenreIsRPG","GenreIsSimulation",
                      "GenreIsEarlyAccess","GenreIsFreeToPlay","GenreIsSports",
                      "GenreIsRacing","GenreIsMassivelyMultiplayer"]
    category_flags = ["CategorySinglePlayer","CategoryMultiplayer","CategoryCoop",
                      "CategoryMMO","CategoryInAppPurchase","CategoryIncludeSrcSDK",
                      "CategoryIncludeLevelEditor","CategoryVRSupport"]
    genre_count    = float(sum(flt(f, 0.0) for f in genre_flags))
    category_count = float(sum(flt(f, 0.0) for f in category_flags))
    platform_count = flt("PlatformWindows", 1) + flt("PlatformLinux", 0) + flt("PlatformMac", 0)

    # price_tier: pd.cut with exact same bins as Preprocessing_m2.py
    price_tier = float(pd.cut(
        [price_final],
        bins=[-0.01, 0.0, 5.0, 15.0, 30.0, float("inf")],
        labels=[0, 1, 2, 3, 4]
    )[0])

    # owners_tier / players_tier: 5 equal bins on log values (Preprocessing_m2.py uses bins=5)
    # We replicate with fixed quantile-like edges derived from log scale
    def _log_tier5(log_val):
        """5 equal-width bins on the log-transformed value, clamped 0–4."""
        # log1p(50M) ≈ 17.7  →  each bin ≈ 3.54
        edges = [0, 3.54, 7.08, 10.62, 14.16, float("inf")]
        for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
            if log_val <= hi:
                return float(i)
        return 4.0

    owners_tier  = _log_tier5(ss_own_log)
    players_tier = _log_tier5(ss_play_log)

    achievement_tier = float(pd.cut(
        [ach_count],
        bins=[-1, 0, 20, 100, float("inf")],
        labels=[0, 1, 2, 3]
    )[0])

    # 8. Assemble numeric row dict
    row_dict = {
        "RequiredAge"                  : req_age,
        "DemoCount"                    : demo_cnt,
        "DeveloperCount"               : dev_cnt,
        "DLCCount"                     : dlc_cnt_log,
        "Metacritic"                   : metacritic,
        "MovieCount"                   : movie_count,
        "PackageCount"                 : pkg_cnt,
        "PublisherCount"               : pub_cnt,
        "RecommendationCount"          : rec_count,
        "ScreenshotCount"              : flt("ScreenshotCount", 0),
        "SteamSpyOwners"               : ss_own,
        "SteamSpyOwnersVariance"       : ss_ownv,
        "SteamSpyPlayersEstimate"      : ss_play,
        "SteamSpyPlayersVariance"      : ss_playv,
        "AchievementCount"             : ach_count,
        "AchievementHighlightedCount"  : flt("AchievementHighlightedCount", 0),
        "PriceInitial"                 : price_initial,
        "PriceFinal"                   : price_final,
        "release_year"                 : release_year,
        "release_month"                : release_month,
        "game_age_days"                : game_age_days,
        "num_languages"                : num_languages,
        "about_length"                 : about_length,
        "short_length"                 : short_length,
        "detail_length"                : detail_length,
        "SteamSpyOwners_log"           : ss_own_log,
        "SteamSpyOwnersVariance_log"   : ss_ownv_log,
        "SteamSpyPlayersEstimate_log"  : ss_play_log,
        "SteamSpyPlayersVariance_log"  : ss_playv_log,
        "discount_ratio"               : discount_ratio,
        "is_effectively_free"          : is_effectively_free,
        "has_metacritic"               : has_metacritic,
        "has_website"                  : has_website,
        "has_support_email"            : has_support_email,
        "has_support_url"              : has_support_url,
        "has_legal_notice"             : has_legal_notice,
        "has_reviews_text"             : has_reviews_text,
        "has_pc_min_reqs"              : has_pc_min_reqs,
        "has_pc_rec_reqs"              : has_pc_rec_reqs,
        "has_linux_min_reqs"           : has_linux_min_reqs,
        "has_mac_min_reqs"             : has_mac_min_reqs,
        "has_drm"                      : has_drm,
        "has_ext_account"              : has_ext_account,
        "price_per_language"           : price_per_language,
        "metacritic_x_age"             : metacritic_x_age,
        "owners_per_achievement"       : owners_per_achievement,
        "dlc_x_owners"                 : dlc_x_owners,
        "movie_x_owners"               : movie_x_owners,
        "owners_tier"                  : owners_tier,
        "achievement_tier"             : achievement_tier,
        "players_tier"                 : players_tier,
        "platform_count"               : platform_count,
        "genre_count"                  : genre_count,
        "category_count"               : category_count,
        "price_tier"                   : price_tier,
    }

    for bf in BINARY_FLAGS:
        row_dict[bf] = flt(bf, 0.0)

    feat_df = pd.DataFrame([row_dict])

    # 9. Scale continuous features
    if scaler is not None:
        scaler_cols_path = os.path.join(MODELS_DIR, "scaler_columns.pkl")
        if os.path.exists(scaler_cols_path):
            fitted_cols = joblib.load(scaler_cols_path)
        elif hasattr(scaler, "feature_names_in_"):
            fitted_cols = scaler.feature_names_in_.tolist()
        else:
            fitted_cols = CONT_FEAT_COLS_FOR_SCALING
        cols_to_use = [c for c in fitted_cols if c in feat_df.columns]
        if cols_to_use:
            feat_df[cols_to_use] = scaler.transform(feat_df[cols_to_use])

    # 10. NLP / LSA features
    if tfidf_vecs is not None and svd_mods is not None:
        lsa_parts = []
        for key, col in NLP_FIELD_MAP.items():
            if key not in tfidf_vecs or key not in svd_mods:
                continue
            raw_text  = txt(col)
            cleaned   = _clean_text(raw_text)
            vec       = tfidf_vecs[key]
            svd_model = svd_mods[key]
            tfidf_mat = vec.transform([cleaned])
            lsa_vec   = svd_model.transform(tfidf_mat)
            lsa_vec   = _safe_normalize(lsa_vec)
            n_comp    = lsa_vec.shape[1]
            col_names = [f"lsa_{key}_{j}" for j in range(n_comp)]
            lsa_parts.append(pd.DataFrame(lsa_vec, columns=col_names))

        if lsa_parts:
            lsa_df  = pd.concat(lsa_parts, axis=1)
            feat_df = pd.concat([feat_df.reset_index(drop=True),
                                  lsa_df.reset_index(drop=True)], axis=1)

    return feat_df


# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLES
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SteamML · Classify",
    page_icon="🏷️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{
  --bg:#0d0f14;--surface:#13161e;--surface2:#1a1e2b;--border:#252a38;
  --accent:#5b8df6;--accent2:#f65b8d;--accent3:#5bf6c8;--accent4:#f6c85b;
  --text:#e8eaf2;--muted:#6b7280;
  --mono:'Space Mono',monospace;--sans:'DM Sans',sans-serif;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:var(--sans)!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
header[data-testid="stHeader"]{background:transparent!important;}
#MainMenu,footer{visibility:hidden;}
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] select,
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea{
    background:var(--surface2)!important;color:var(--text)!important;
    border:1px solid var(--border)!important;border-radius:6px!important;
    font-family:var(--mono)!important;font-size:11px!important;}
[data-testid="stTabs"] button{font-family:var(--mono)!important;font-size:11px!important;color:var(--muted)!important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--accent)!important;border-bottom-color:var(--accent)!important;}
[data-testid="stMetric"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:10px!important;padding:14px!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:11px!important;}
[data-testid="stMetricValue"]{color:var(--text)!important;}
.ph{background:linear-gradient(135deg,#0d0f14,#111827,#0d1420);border:1px solid var(--border);
    border-left:4px solid var(--accent);border-radius:12px;padding:28px 36px;margin-bottom:24px;}
.ph h1{font-size:22px;font-weight:600;margin:0 0 4px;color:var(--text);}
.ph p{font-size:12px;color:var(--muted);margin:0;}
.sec{display:flex;align-items:center;gap:10px;margin:20px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--border);}
.sec-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.sec-lbl{font-family:var(--mono);font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--text);}
.result-box{border-radius:12px;padding:28px;text-align:center;margin-top:20px;}
.result-box .big{font-size:52px;font-weight:700;font-family:var(--mono);line-height:1;}
.result-box .sub{font-size:11px;color:var(--muted);margin-top:6px;}
.pipeline-step{background:var(--surface);border:1px solid var(--border);border-radius:8px;
    padding:10px 16px;margin:4px 0;font-size:11px;font-family:var(--mono);color:var(--muted);}
.pipeline-step.done{border-left:3px solid var(--accent3);color:var(--accent3);}
.pipeline-step.err{border-left:3px solid #f65b8d;color:#f65b8d;}
.tip{background:#13161e;border:1px solid #252a38;border-radius:8px;padding:10px 14px;
    font-size:11px;color:#6b7280;margin-bottom:12px;}
.stButton>button{background:var(--accent)!important;color:#0d0f14!important;border:none!important;
    border-radius:8px!important;font-family:var(--mono)!important;font-size:12px!important;
    letter-spacing:1px!important;padding:12px 28px!important;font-weight:700!important;width:100%!important;}
.log-box{background:#080a0e;border:1px solid var(--border);border-radius:8px;padding:14px 18px;
    font-family:monospace;font-size:11px;color:var(--accent3);white-space:pre-wrap;
    max-height:380px;overflow-y:auto;line-height:1.8;}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:10px;
    font-family:var(--mono);font-weight:700;margin-right:6px;}
.status-ok{color:#5bf6c8;font-family:monospace;font-size:11px;}
.status-no{color:#6b7280;font-family:monospace;font-size:11px;}
.warn-box{background:#13161e;border:1px solid #f6c85b44;border-left:3px solid #f6c85b;
    border-radius:8px;padding:10px 14px;font-size:11px;color:#f6c85b;margin-bottom:12px;}
.prob-bar-wrap{background:#1a1e2b;border-radius:6px;height:14px;margin:4px 0;overflow:hidden;}
.info-pill{display:inline-block;padding:3px 12px;border-radius:20px;font-size:10px;
    font-family:var(--mono);background:#1a1e2b;border:1px solid var(--border);color:var(--muted);margin-right:6px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
tfidf_vecs, svd_mods = _load_nlp_models()
scaler               = _load_scaler()
label_encoder        = _load_label_encoder()
nlp_ok               = tfidf_vecs is not None
scaler_ok            = scaler is not None
le_ok                = label_encoder is not None
saved_models         = [n for n in MODEL_FILES if _model_exists(n)]

with st.sidebar:
    st.markdown(
        "<div style='font-family:monospace;font-size:18px;font-weight:700;"
        "color:#5b8df6;padding:8px 0 20px'>🎮 SteamML</div>",
        unsafe_allow_html=True,
    )
    st.page_link("app.py",                      label="⬡  Dashboard")
    st.page_link("pages/preprocessing_page.py", label="⬡  Preprocessing & NLP")
    st.page_link("pages/model_page.py",         label="⬡  Models")
    st.markdown("---")

    def _dot(ok): return ("●", "#5bf6c8") if ok else ("○", "#6b7280")

    d, c = _dot(nlp_ok)
    st.markdown(f"<div style='font-size:12px;line-height:2.4'>"
                f"<span style='color:{c}'>{d}</span>&nbsp;NLP models {'loaded ✓' if nlp_ok else 'not found'}<br>",
                unsafe_allow_html=True)
    d, c = _dot(scaler_ok)
    st.markdown(f"<span style='color:{c}'>{d}</span>&nbsp;Scaler {'loaded ✓' if scaler_ok else 'not found'}<br>",
                unsafe_allow_html=True)
    d, c = _dot(le_ok)
    st.markdown(f"<span style='color:{c}'>{d}</span>&nbsp;Label encoder {'loaded ✓' if le_ok else 'not found'}<br>",
                unsafe_allow_html=True)
    d, c = _dot(bool(saved_models))
    st.markdown(
        f"<span style='color:{c}'>{d}</span>&nbsp;{len(saved_models)} / {len(MODEL_FILES)} models saved<br>"
        f"<span style='color:#5b8df6'>●</span>&nbsp;Target: High · Low · Medium</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""<div class="ph">
    <h1>🏷️  SteamML · Classify Game Popularity</h1>
    <p>Predict GamePopularity class for a new game · Feature selection · Load &amp; compare classifiers
    &nbsp;·&nbsp; <span style="color:#5bf6c8;font-family:monospace;font-size:11px">
    Classes: High (0) · Low (1) · Medium (2)</span></p>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────
tab_predict, tab_fs, tab_compare = st.tabs([
    "🏷️  PREDICT NEW GAME",
    "🔍  FEATURE SELECTION",
    "📊  MODEL COMPARISON",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT NEW GAME
# ══════════════════════════════════════════════════════════════════
with tab_predict:
    if not saved_models:
        st.error("⚠️  No trained classifier models found in ./trained_models/.")
        st.stop()

    if not nlp_ok:
        st.markdown(
            "<div class='warn-box'>⚠️  NLP models not found. LSA text features will be zero.</div>",
            unsafe_allow_html=True,
        )
    if not scaler_ok:
        st.markdown(
            "<div class='warn-box'>⚠️  Scaler (scaler.pkl) not found. Continuous features will not be scaled.</div>",
            unsafe_allow_html=True,
        )

    # Model selector
    col_ms, col_info = st.columns([2, 3], gap="large")
    with col_ms:
        sel_model = st.selectbox("Classifier", saved_models, key="pred_model_tab")
    with col_info:
        mc = MODEL_COLORS.get(sel_model, "#5b8df6")
        pred_model = _load_model(sel_model)
        feat_names = _get_feature_names(pred_model)
        n_feats    = len(feat_names) if feat_names else "?"
        has_proba  = hasattr(pred_model, "predict_proba")
        st.markdown(
            f"<div style='background:var(--surface);border:1px solid var(--border);"
            f"border-left:4px solid {mc};border-radius:8px;padding:12px 16px;margin-top:22px;"
            f"font-size:11px;color:var(--muted);font-family:var(--mono)'>"
            f"<span style='color:{mc};font-weight:700'>{sel_model}</span> · "
            f"{n_feats} features · predicts High / Low / Medium · "
            f"{'probabilities ✓' if has_proba else 'no predict_proba'}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Input tabs
    itab_form, itab_paste, itab_upload = st.tabs([
        "📝  FILL FORM",
        "📋  PASTE CSV ROW",
        "📁  UPLOAD CSV FILE",
    ])
    raw_input: dict | None = None

    # ── TAB A: Structured form ─────────────────────────────────
    with itab_form:
        st.markdown(
            "<div class='tip'>Fill in the game data below. "
            "All text fields contribute to NLP/LSA features.</div>",
            unsafe_allow_html=True,
        )
        form_data = {}

        st.markdown('<div class="sec"><div class="sec-dot" style="background:#f6c85b"></div>'
                    '<div class="sec-lbl">Game Identity &amp; Text</div></div>', unsafe_allow_html=True)
        ct1, ct2 = st.columns(2)
        with ct1:
            form_data["ResponseName"]      = st.text_input("Game Name",    value="My Awesome Game", key="f_name")
            form_data["ReleaseDate"]        = st.text_input("Release Date", value="2022-06-15",      key="f_date")
            form_data["SupportedLanguages"] = st.text_input("Supported Languages",
                                                             value="English French German", key="f_lang")
            form_data["Website"]            = st.text_input("Website URL", value="", key="f_web")
            form_data["SupportEmail"]       = st.text_input("Support Email", value="", key="f_email")
        with ct2:
            form_data["AboutText"]          = st.text_area("About Text",        value="", height=80, key="f_about")
            form_data["ShortDescrip"]       = st.text_area("Short Description", value="", height=60, key="f_short")

        form_data["DetailedDescrip"]        = st.text_area("Detailed Description", value="", height=100, key="f_detail")
        form_data["Reviews"]                = st.text_area("Reviews Text",          value="", height=60,  key="f_reviews")

        pc1, pc2 = st.columns(2)
        with pc1:
            form_data["PCMinReqsText"] = st.text_area("PC Min Requirements",
                                                       value="OS: Windows 10\nCPU: Intel i3\nRAM: 4 GB",
                                                       height=70, key="f_pcmin")
        with pc2:
            form_data["PCRecReqsText"] = st.text_area("PC Rec Requirements",
                                                       value="OS: Windows 11\nCPU: Intel i7\nRAM: 16 GB",
                                                       height=70, key="f_pcrec")

        st.markdown('<div class="sec"><div class="sec-dot" style="background:#5b8df6"></div>'
                    '<div class="sec-lbl">Numeric Attributes</div></div>', unsafe_allow_html=True)
        num_keys = list(FORM_NUMERIC.keys())
        for row_chunk in [num_keys[i:i+4] for i in range(0, len(num_keys), 4)]:
            cols_n = st.columns(len(row_chunk))
            for col_w, feat in zip(cols_n, row_chunk):
                mn, mx, dv = FORM_NUMERIC[feat]
                form_data[feat] = col_w.number_input(feat,
                    min_value=float(mn), max_value=float(mx), value=float(dv),
                    key=f"form_{feat}")

        st.markdown('<div class="sec"><div class="sec-dot" style="background:#5bf6c8"></div>'
                    '<div class="sec-lbl">Flags &amp; Categories</div></div>', unsafe_allow_html=True)
        for brow in [FORM_BOOL[i:i+5] for i in range(0, len(FORM_BOOL), 5)]:
            bcols = st.columns(len(brow))
            for bcol, bf in zip(bcols, brow):
                form_data[bf] = int(bcol.checkbox(bf, value=False, key=f"form_{bf}"))

        for c in RAW_COLS:
            if c not in form_data:
                form_data[c] = ""

        _, btn_col, _ = st.columns([2, 2, 2])
        with btn_col:
            if st.button("🏷️  CLASSIFY FROM FORM", key="btn_form"):
                raw_input = form_data

    # ── TAB B: Paste CSV ───────────────────────────────────────
    with itab_paste:
        st.markdown(
            "<div class='tip'>"
            "Paste a header row + one data row (comma-separated, original raw columns)."
            "</div>",
            unsafe_allow_html=True,
        )
        pasted = st.text_area(
            "Paste CSV here",
            height=180,
            placeholder=f"Paste header + data row:\n{','.join(RAW_COLS[:10])},...",
            key="paste_area",
        )
        _, btn_col2, _ = st.columns([2, 2, 2])
        with btn_col2:
            if st.button("🏷️  CLASSIFY FROM PASTE", key="btn_paste"):
                if not pasted.strip():
                    st.error("Please paste some CSV content first.")
                else:
                    try:
                        df_paste = pd.read_csv(io.StringIO(pasted.strip()))
                        if df_paste.shape[0] == 0:
                            vals = list(next(iter(
                                pd.read_csv(io.StringIO(pasted.strip()), header=None).itertuples(index=False)
                            )))
                            df_paste = pd.DataFrame([vals], columns=RAW_COLS[:len(vals)])
                        raw_input = df_paste.iloc[0].to_dict()
                    except Exception as e:
                        st.error(f"Could not parse CSV: {e}")

    # ── TAB C: Upload CSV ──────────────────────────────────────
    with itab_upload:
        st.markdown(
            "<div class='tip'>"
            "Upload a CSV with the same columns as train_data.csv (raw, before preprocessing). "
            "If multiple rows, the first data row is used."
            "</div>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="upload_csv")
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded)
                st.success(f"File loaded · {df_up.shape[0]} rows · {df_up.shape[1]} columns")
                if df_up.shape[0] > 1:
                    st.info(f"Multiple rows detected — using row 1 of {df_up.shape[0]}.")
                st.dataframe(df_up.head(3), use_container_width=True)
                _, btn_col3, _ = st.columns([2, 2, 2])
                with btn_col3:
                    if st.button("🏷️  CLASSIFY FROM FILE", key="btn_upload"):
                        raw_input = df_up.copy()
                       # raw_input = df_up.iloc[0].to_dict()
            except Exception as e:
                st.error(f"Could not read file: {e}")

    # ── PREDICTION EXECUTION ───────────────────────────────────
    if raw_input is not None:
        st.markdown("---")
        st.markdown('<div class="sec"><div class="sec-dot" style="background:#5bf6c8"></div>'
                    '<div class="sec-lbl">Pipeline Execution</div></div>', unsafe_allow_html=True)

        _steps: list[tuple[str, str]] = []
        success      = False
        pred_class   = None
        pred_label   = None
        pred_proba   = None
        X_new        = None

        def _render_steps(steps):
            parts = []
            for msg, kind in steps:
                if kind == "done":
                    style = "border-left:3px solid #5bf6c8;color:#5bf6c8"
                    icon  = "✓"
                elif kind == "warn":
                    style = "border-left:3px solid #f6c85b;color:#f6c85b"
                    icon  = "⚠"
                else:
                    style = "border-left:3px solid #f65b8d;color:#f65b8d"
                    icon  = "✗"
                safe_msg = str(msg).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                parts.append(
                    f'<div class="pipeline-step" style="{style}">{icon} {safe_msg}</div>'
                )
            pipeline_placeholder.markdown("\n".join(parts), unsafe_allow_html=True)

        pipeline_placeholder = st.empty()

        try:
            _steps.append(("Date engineering (release_year / month / game_age_days)", "done"))
            _steps.append(("Numeric feature engineering (log transforms, interactions, tiers…)", "done"))

            if nlp_ok:
                _steps.append(("TF-IDF + LSA applied to text fields", "done"))
            else:
                _steps.append(("NLP models missing — LSA features set to 0", "warn"))

            if scaler_ok:
                _steps.append(("StandardScaler loaded — continuous features will be scaled", "done"))
            else:
                _steps.append(("Scaler not found — features will not be scaled", "warn"))

            _render_steps(_steps)
            if isinstance(raw_input, pd.DataFrame):
                 processed_rows = []
                 for _, row in raw_input.iterrows():       
                         feat = _preprocess_row(
                         row.to_dict(),
                         tfidf_vecs,
                         svd_mods,
                         scaler
                         )
                         processed_rows.append(feat)
                 feat_df = pd.concat(processed_rows, ignore_index=True)
            else:

                  feat_df = _preprocess_row(
                  raw_input,
                  tfidf_vecs,
                  svd_mods,
                  scaler
               )

            
            _steps.append((f"Feature vector assembled — {feat_df.shape[1]} columns before alignment", "done"))
            _render_steps(_steps)

            X_new = _align(pred_model, feat_df)
            _steps.append((f"Feature alignment → {X_new.shape[1]} features matched to model", "done"))
            _render_steps(_steps)
            # ================= CLASSIFICATION PREDICTIONS =================

            preds = pred_model.predict(X_new)

            # decode labels
            decoded_preds = [_decode_class(int(p)) for p in preds]

            # probabilities
            pred_proba = None
            if hasattr(pred_model, "predict_proba"):
                pred_proba = pred_model.predict_proba(X_new)

            _steps.append((f"Classification complete → {len(preds)} samples predicted", "done"))
            _render_steps(_steps)

            success = True

            # ============================================================
            # SINGLE SAMPLE
            # ============================================================

            if len(preds) == 1:

                pred_class = int(preds[0])
                pred_label = decoded_preds[0]

                cls_color = CLASS_COLORS.get(pred_label, "#5bf6c8")

                st.markdown(
                    f"<div class='result-box' style='background:linear-gradient(135deg,#0d0f14,#111827);"
                    f"border:1px solid {cls_color}'>"
                    f"<div style='font-size:11px;color:var(--muted);font-family:var(--mono);"
                    f"margin-bottom:10px'>{sel_model} · GamePopularity Classification</div>"
                    f"<div class='big' style='color:{cls_color}'>{pred_label}</div>"
                    f"<div class='sub'>predicted class · encoded {pred_class}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # ================= PROBABILITIES =================
                if pred_proba is not None:

                    st.markdown(
                        '<div class="sec"><div class="sec-dot" style="background:#5b8df6"></div>'
                        '<div class="sec-lbl">Class Probabilities</div></div>',
                        unsafe_allow_html=True
                    )

                    prob_cols = st.columns(len(CLASS_NAMES))

                    for i, (cls_name, p_col) in enumerate(zip(CLASS_NAMES, prob_cols)):

                        prob_val = float(pred_proba[0][i]) if i < len(pred_proba[0]) else 0.0

                        cls_clr = CLASS_COLORS[cls_name]
                        is_pred = cls_name == pred_label

                        border = f"border:1px solid {cls_clr};" if is_pred else ""

                        p_col.markdown(
                            f"<div style='background:var(--surface);{border}"
                            f"border-radius:10px;padding:14px;text-align:center'>"
                            f"<div style='font-size:10px;color:var(--muted);font-family:var(--mono)'>{cls_name}</div>"
                            f"<div style='font-size:28px;font-weight:700;font-family:var(--mono);"
                            f"color:{cls_clr}'>{prob_val:.1%}</div>"
                            f"<div class='prob-bar-wrap'>"
                            f"<div style='width:{prob_val*100:.1f}%;height:100%;background:{cls_clr};"
                            f"border-radius:6px'></div>"
                            f"</div></div>",
                            unsafe_allow_html=True,
                        )

            # ============================================================
            # MULTIPLE SAMPLES
            # ============================================================

            else:

                result_df = pd.DataFrame({
                    "Row": range(1, len(preds) + 1),
                    "Predicted_Class": preds,
                    "Predicted_Label": decoded_preds,
                })

                # add confidence if available
                if pred_proba is not None:
                    result_df["Confidence"] = pred_proba.max(axis=1)

                st.dataframe(result_df, use_container_width=True)

                # ================= DOWNLOAD CSV =================
                csv_data = result_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇ Download Predictions CSV",
                    data=csv_data,
                    file_name="classification_predictions.csv",
                    mime="text/csv",
                )

                # ========================================================
                # METRICS (IF TARGET EXISTS)
                # ========================================================

                if isinstance(raw_input, pd.DataFrame) and TARGET in raw_input.columns:

                    from sklearn.metrics import (
                        accuracy_score,
                        precision_score,
                        recall_score,
                        f1_score,
                        classification_report,
                        confusion_matrix,
                    )

                    actual = raw_input[TARGET].values

                    acc  = accuracy_score(actual, preds)
                    prec = precision_score(actual, preds, average="weighted", zero_division=0)
                    rec  = recall_score(actual, preds, average="weighted", zero_division=0)
                    f1   = f1_score(actual, preds, average="weighted", zero_division=0)

                    m1, m2, m3, m4 = st.columns(4)

                    m1.metric("Accuracy",  f"{acc:.4f}")
                    m2.metric("Precision", f"{prec:.4f}")
                    m3.metric("Recall",    f"{rec:.4f}")
                    m4.metric("F1 Score",  f"{f1:.4f}")

                    st.markdown(
                        f"<div style='background:#13161e;border:1px solid #5bf6c8;"
                        f"border-radius:10px;padding:16px 20px;margin-top:12px;text-align:center'>"
                        f"<div style='font-size:11px;color:#6b7280;font-family:monospace'>Classification Accuracy</div>"
                        f"<div style='font-size:36px;font-weight:700;color:#5bf6c8;font-family:monospace'>{acc:.4f}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # ================= CLASSIFICATION REPORT =================
                    with st.expander("📋 Classification Report"):

                        report = classification_report(
                            actual,
                            preds,
                            target_names=CLASS_NAMES,
                            output_dict=True,
                            zero_division=0
                        )

                        st.dataframe(
                            pd.DataFrame(report).transpose(),
                            use_container_width=True
                        )

                    # ================= CONFUSION MATRIX =================
                    with st.expander("📊 Confusion Matrix"):

                        cm = confusion_matrix(actual, preds)

                        cm_df = pd.DataFrame(
                            cm,
                            index=[f"Actual {c}" for c in CLASS_NAMES],
                            columns=[f"Pred {c}" for c in CLASS_NAMES]
                        )

                        st.dataframe(cm_df, use_container_width=True)

            # ============================================================
            # FEATURE VECTOR
            # ============================================================

            with st.expander("🔍 Show processed feature vector"):
                st.dataframe(X_new, use_container_width=True)

        except Exception as e:
            _steps.append((f"ERROR: {e}", "err"))
            _render_steps(_steps)

            st.error("Prediction pipeline failed.")
            st.code(traceback.format_exc(), language="python")

# ══════════════════════════════════════════════════════════════════
# TAB 2 — FEATURE SELECTION
# Dominant-value → VarianceThreshold → RF importance
# ══════════════════════════════════════════════════════════════════
with tab_fs:
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.ensemble import RandomForestClassifier

    st.markdown(
        "<div style='background:#13161e;border:1px solid #252a38;border-radius:10px;"
        "padding:16px 20px;font-size:12px;color:#6b7280;line-height:2;margin-bottom:20px'>"
        "Pipeline: "
        "<b style='color:#e8eaf2'>① Dominant-value filter</b> (≥ 95%) → "
        "<b style='color:#e8eaf2'>② VarianceThreshold</b> (var &lt; 0.001) → "
        "<b style='color:#e8eaf2'>③ RF importance filter</b> (importance ≥ 0.001) → "
        "Save to <code>./dataset/selected/</code>"
        "</div>",
        unsafe_allow_html=True,
    )

    prereq_ok = (
        os.path.exists(os.path.join(PROC_DIR, "train.csv")) and
        os.path.exists(os.path.join(PROC_DIR, "test.csv"))
    )
    nlp_avail = (
        os.path.exists(os.path.join(PROC_DIR, "nlp_features_train.csv")) and
        os.path.exists(os.path.join(PROC_DIR, "nlp_features_test.csv"))
    )

    if not prereq_ok:
        st.warning("⚠️  Run Preprocessing first to generate processed CSVs.")

    fl, fr = st.columns([1, 1], gap="large")
    with fl:
        if _data_ready():
            st.success("✓ Selected features already saved — click to re-run and overwrite.")
        else:
            st.info("ℹ️  Selected features not found — run the pipeline below.")
        run_fs = st.button("▶  RUN FEATURE SELECTION", key="run_fs", disabled=not prereq_ok)
    with fr:
        st.markdown(
            "<div style='background:#13161e;border:1px solid #252a38;border-radius:10px;"
            "padding:16px 18px;font-size:12px;color:#6b7280;line-height:2'>"
            "① Merge processed + NLP CSVs<br>"
            "② Dominant-value filter (≥ 95%)<br>"
            "③ VarianceThreshold (threshold = 0.001)<br>"
            "④ RF importance filter (importance ≥ 0.001)<br>"
            "⑤ Save → ./dataset/selected/"
            "</div>",
            unsafe_allow_html=True,
        )

    if run_fs:
        log_lines = []
        L = lambda m, k="ok": log_lines.append((m, k))

        def render_log_fs(lines):
            html = "".join(
                '<div style="{c}">{m}</div>'.format(
                    c=('color:#6b7280' if k == 'muted' else
                       'color:#f65b8d' if k == 'err' else 'color:#5bf6c8'),
                    m=m.replace('<', '&lt;').replace('>', '&gt;')
                )
                for m, k in lines
            )
            st.markdown(f'<div class="log-box">{html}</div>', unsafe_allow_html=True)

        try:
            L("// Feature Selection started ─────────────────────────────────")

            def merge_split(num_df, nlp_df=None):
                num_df = num_df.reset_index(drop=True)
                if nlp_df is not None:
                    nlp_df = nlp_df.reset_index(drop=True)
                    min_len = min(len(num_df), len(nlp_df))
                    merged  = pd.concat([num_df.iloc[:min_len], nlp_df.iloc[:min_len]], axis=1)
                else:
                    merged = num_df
                return merged.loc[:, ~merged.columns.duplicated()]

            train_num = pd.read_csv(os.path.join(PROC_DIR, "train.csv"))
            test_num  = pd.read_csv(os.path.join(PROC_DIR, "test.csv"))

            if nlp_avail:
                train_nlp = pd.read_csv(os.path.join(PROC_DIR, "nlp_features_train.csv"))
                test_nlp  = pd.read_csv(os.path.join(PROC_DIR, "nlp_features_test.csv"))
                train_df  = merge_split(train_num, train_nlp)
                test_df   = merge_split(test_num, test_nlp)
                L(f"  merged numeric + NLP  train={train_df.shape}  test={test_df.shape}")
            else:
                train_df = merge_split(train_num)
                test_df  = merge_split(test_num)
                L("  NLP features not found — using numeric only", "muted")

            X_train = train_df.drop(columns=DROP_COLS, errors="ignore")
            y_train = train_df[TARGET] if TARGET in train_df.columns else None
            X_test  = test_df.drop(columns=DROP_COLS, errors="ignore")
            y_test  = test_df[TARGET] if TARGET in test_df.columns else None

            if y_train is None:
                raise ValueError(f"Target column '{TARGET}' not found in train.csv")

            initial = X_train.shape[1]
            L(f"  initial features: {initial}")

            # ① Dominant-value filter
            dom_drop = [col for col in X_train.columns
                        if X_train[col].value_counts(normalize=True, dropna=False).max() >= DOMINANT_THRESHOLD]
            X_train.drop(columns=dom_drop, inplace=True)
            X_test.drop(columns=dom_drop,  inplace=True)
            L(f"  [①] dominant-value filter → dropped {len(dom_drop):3d}  remaining: {X_train.shape[1]}", "muted")

            # ② VarianceThreshold
            before_var = X_train.columns.tolist()
            vt         = VarianceThreshold(threshold=VAR_THRESHOLD)
            Xtr_a      = vt.fit_transform(X_train)
            Xte_a      = vt.transform(X_test)
            sel_vt     = [c for c, k in zip(before_var, vt.get_support()) if k]
            X_train    = pd.DataFrame(Xtr_a, columns=sel_vt)
            X_test     = pd.DataFrame(Xte_a, columns=sel_vt)
            L(f"  [②] VarianceThreshold={VAR_THRESHOLD} → dropped {len(before_var)-len(sel_vt):3d}  remaining: {X_train.shape[1]}", "muted")

            # ③ RF importance (classifier)
            L("  [③] fitting RandomForestClassifier for importance … (n_estimators=200)")
            rf_sel = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            rf_sel.fit(X_train, y_train)
            importances   = rf_sel.feature_importances_
            imp_mask      = importances >= IMP_THRESHOLD
            selected_cols = X_train.columns[imp_mask].tolist()
            X_train = X_train[selected_cols]
            X_test  = X_test[selected_cols]

            imp_df = pd.DataFrame({
                "feature":    selected_cols,
                "importance": importances[imp_mask]
            }).sort_values("importance", ascending=False)

            L(f"  [③] RF importance >= {IMP_THRESHOLD} → remaining: {X_train.shape[1]}", "muted")

            os.makedirs(SEL_DIR, exist_ok=True)
            out_tr = X_train.copy(); out_tr[TARGET] = y_train.values
            out_te = X_test.copy();  out_te[TARGET] = y_test.values if y_test is not None else 0
            out_tr.to_csv(os.path.join(SEL_DIR, "train_selected.csv"), index=False)
            out_te.to_csv(os.path.join(SEL_DIR, "test_selected.csv"),  index=False)
            L("  saved to ./dataset/selected/")
            L(f"  initial={initial}  final={X_train.shape[1]}  removed={initial - X_train.shape[1]}")
            L("// Feature Selection Done ✓ ─────────────────────────────────")

            st.session_state["fs_imp_df"]    = imp_df
            st.session_state["fs_sel_feats"] = selected_cols

        except Exception as e:
            L(f"ERROR: {e}", "err")
            L(traceback.format_exc(), "err")

        render_log_fs(log_lines)
        st.rerun()

    if _data_ready():
        train_sel = pd.read_csv(os.path.join(SEL_DIR, "train_selected.csv"))
        test_sel  = pd.read_csv(os.path.join(SEL_DIR, "test_selected.csv"))
        feat_cols = [c for c in train_sel.columns if c not in DROP_COLS and c != TARGET]

        st.markdown('<div class="sec"><div class="sec-dot" style="background:#5bf6c8"></div>'
                    '<div class="sec-lbl">Selection Summary</div></div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Features", len(feat_cols))
        c2.metric("Train Rows",     f"{len(train_sel):,}")
        c3.metric("Test Rows",      f"{len(test_sel):,}")
        c4.metric("Target",         "High / Low / Medium")

        imp_df = st.session_state.get("fs_imp_df")
        if imp_df is not None:
            st.markdown('<div class="sec"><div class="sec-dot" style="background:#f6c85b"></div>'
                        '<div class="sec-lbl">RandomForest Feature Importances — Top 25</div></div>', unsafe_allow_html=True)
            top25 = imp_df.head(25)
            fig, ax = plt.subplots(figsize=(14, 7), facecolor="#13161e")
            ax.set_facecolor("#1a1e2b")
            clrs = [MODEL_COLORS["Random Forest"] if "lsa" not in f else "#7F77DD" for f in top25["feature"]]
            ax.barh(range(len(top25)), top25["importance"].values[::-1],
                    color=clrs[::-1], alpha=0.85, height=0.75)
            ax.set_yticks(range(len(top25)))
            ax.set_yticklabels(top25["feature"].values[::-1], fontsize=8, color="#e8eaf2")
            ax.set_title("Top 25 Feature Importances (green=numeric, purple=NLP/LSA)",
                         color="#e8eaf2", fontsize=9, fontfamily="monospace")
            ax.set_xlabel("Importance", fontsize=8, color="#6b7280")
            ax.tick_params(colors="#6b7280", labelsize=7)
            for sp in ax.spines.values(): sp.set_color("#252a38")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()


# ══════════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════
with tab_compare:
    saved_now = [n for n in MODEL_FILES if _model_exists(n)]

    if not saved_now:
        st.warning("⚠️  No saved classifier models found.")
    elif not (os.path.exists(os.path.join(PROC_DIR, "test.csv"))):
        st.warning("⚠️  Processed test.csv not found. Run preprocessing first.")
    else:
        @st.cache_data
        def _load_test_data():
            te = pd.read_csv(os.path.join(PROC_DIR, "test.csv"))
            nlp_path = os.path.join(PROC_DIR, "nlp_features_test.csv")
            if os.path.exists(nlp_path):
                nlp = pd.read_csv(nlp_path)
                min_len = min(len(te), len(nlp))
                te = pd.concat([te.iloc[:min_len].reset_index(drop=True),
                                nlp.iloc[:min_len].reset_index(drop=True)], axis=1)
                te = te.loc[:, ~te.columns.duplicated()]
            X_te = te.drop(columns=DROP_COLS, errors="ignore")
            y_te = te[TARGET].astype(int) if TARGET in te.columns else None
            return X_te, y_te

        X_te_c, y_te_c = _load_test_data()

        if y_te_c is None:
            st.error(f"Target column '{TARGET}' missing from test data.")
            st.stop()

        results, skipped = {}, []
        for name in saved_now:
            try:
                m      = _load_model(name)
                X_al   = _align(m, X_te_c)
                preds  = np.array(m.predict(X_al)).flatten().astype(int)
                acc    = accuracy_score(y_te_c, preds)
                mac_f1 = f1_score(y_te_c, preds, average="macro")
                per_f1 = f1_score(y_te_c, preds, average=None, labels=[0, 1, 2])
                proba  = m.predict_proba(X_al) if hasattr(m, "predict_proba") else None
                results[name] = dict(
                    preds=preds, acc=acc, mac_f1=mac_f1,
                    per_f1=per_f1, proba=proba, model=m
                )
            except Exception as e:
                skipped.append((name, str(e)))

        for sn, se in skipped:
            st.warning(f"⚠️  Skipped **{sn}**: {se}")

        if results:
            # Summary table
            st.markdown('<div class="sec"><div class="sec-dot" style="background:#5bf6c8"></div>'
                        '<div class="sec-lbl">Performance Summary — Test Set</div></div>', unsafe_allow_html=True)
            rows = []
            for name, res in results.items():
                rows.append({
                    "Model"     : name,
                    "Accuracy"  : round(res["acc"],         4),
                    "Macro-F1"  : round(res["mac_f1"],      4),
                    "F1-High"   : round(float(res["per_f1"][0]), 4),
                    "F1-Low"    : round(float(res["per_f1"][1]), 4),
                    "F1-Medium" : round(float(res["per_f1"][2]), 4),
                })
            df_res = pd.DataFrame(rows).sort_values("Macro-F1", ascending=False)
            st.dataframe(df_res, use_container_width=True, hide_index=True)

            best_name  = df_res.iloc[0]["Model"]
            best_f1    = df_res.iloc[0]["Macro-F1"]
            best_color = MODEL_COLORS.get(best_name, "#5bf6c8")
            st.markdown(
                f"<div style='margin-top:8px;font-size:12px;font-family:monospace'>"
                f"🏆 Best by Macro-F1: <span style='color:{best_color};font-weight:700'>{best_name}</span>"
                f" · Macro-F1 = <span style='color:#5bf6c8'>{best_f1}</span></div>",
                unsafe_allow_html=True,
            )

            # Bar charts — Accuracy, Macro-F1, per-class F1
            st.markdown('<div class="sec"><div class="sec-dot" style="background:#f6c85b"></div>'
                        '<div class="sec-lbl">Metric Comparison</div></div>', unsafe_allow_html=True)
            names  = list(results.keys())
            colors = [MODEL_COLORS.get(n, "#888") for n in names]

            fig, axs = plt.subplots(1, 2, figsize=(18, 5), facecolor="#13161e")
            for ax, (vals, label) in zip(axs, [
                ([results[n]["acc"]    for n in names], "Accuracy"),
                ([results[n]["mac_f1"] for n in names], "Macro-F1"),
            ]):
                ax.set_facecolor("#1a1e2b")
                bars = ax.bar(names, vals, color=colors, alpha=0.85, width=0.5)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.005,
                            f"{v:.4f}", ha="center", va="bottom",
                            fontsize=7, color="#e8eaf2")
                ax.set_title(label, color="#e8eaf2", fontsize=9, fontfamily="monospace")
                ax.set_ylim(0, 1.15)
                ax.tick_params(labelsize=7, colors="#6b7280")
                ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
                for sp in ax.spines.values(): sp.set_color("#252a38")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Per-class F1 grouped bar
            st.markdown('<div class="sec"><div class="sec-dot" style="background:#5b8df6"></div>'
                        '<div class="sec-lbl">Per-Class F1 Score</div></div>', unsafe_allow_html=True)
            n_m = len(names)
            x   = np.arange(n_m)
            w   = 0.25
            fig, ax = plt.subplots(figsize=(max(12, 3*n_m), 5), facecolor="#13161e")
            ax.set_facecolor("#1a1e2b")
            for i, (cls_name, cls_clr) in enumerate(CLASS_COLORS.items()):
                vals = [float(results[n]["per_f1"][i]) for n in names]
                bars = ax.bar(x + (i-1)*w, vals, w,
                              label=cls_name, color=cls_clr, alpha=0.85, edgecolor="none")
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.005,
                            f"{v:.3f}", ha="center", va="bottom",
                            fontsize=6.5, color="#e8eaf2")
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=9, color="#e8eaf2", rotation=15, ha="right")
            ax.set_ylabel("F1 Score", fontsize=8, color="#6b7280")
            ax.set_ylim(0, 1.15)
            ax.set_title("Per-Class F1 Score by Model", color="#e8eaf2",
                         fontsize=9, fontfamily="monospace")
            ax.legend(fontsize=8, framealpha=0.3)
            ax.tick_params(colors="#6b7280", labelsize=7)
            for sp in ax.spines.values(): sp.set_color("#252a38")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Confusion matrices
            st.markdown('<div class="sec"><div class="sec-dot" style="background:#f65b8d"></div>'
                        '<div class="sec-lbl">Confusion Matrices — Test Set</div></div>', unsafe_allow_html=True)
            cm_cols = st.columns(min(n_m, 3))
            for i, (name, res) in enumerate(results.items()):
                ax_col = cm_cols[i % 3]
                cm     = confusion_matrix(y_te_c, res["preds"], labels=[0, 1, 2])
                fig2, ax2 = plt.subplots(figsize=(4, 3.5), facecolor="#13161e")
                ax2.set_facecolor("#1a1e2b")
                im = ax2.imshow(cm, interpolation="nearest",
                                cmap=plt.cm.Blues, vmin=0)
                plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                ax2.set_xticks([0, 1, 2])
                ax2.set_yticks([0, 1, 2])
                ax2.set_xticklabels(CLASS_NAMES, fontsize=7, color="#e8eaf2")
                ax2.set_yticklabels(CLASS_NAMES, fontsize=7, color="#e8eaf2")
                ax2.set_xlabel("Predicted", fontsize=7, color="#6b7280")
                ax2.set_ylabel("True",      fontsize=7, color="#6b7280")
                ax2.set_title(f"{name}\nAcc={res['acc']:.4f}",
                              color="#e8eaf2", fontsize=8, fontfamily="monospace")
                thresh = cm.max() / 2.0
                for rr in range(cm.shape[0]):
                    for cc in range(cm.shape[1]):
                        ax2.text(cc, rr, f"{cm[rr,cc]}",
                                 ha="center", va="center",
                                 color="white" if cm[rr,cc] > thresh else "black",
                                 fontsize=8)
                for sp in ax2.spines.values(): sp.set_color("#252a38")
                plt.tight_layout()
                ax_col.pyplot(fig2, use_container_width=True)
                plt.close()

            # Full classification report for best model
            st.markdown(f'<div class="sec"><div class="sec-dot" style="background:#a78bfa"></div>'
                        f'<div class="sec-lbl">Full Report — {best_name}</div></div>', unsafe_allow_html=True)
            report = classification_report(
                y_te_c, results[best_name]["preds"],
                target_names=CLASS_NAMES, digits=4
            )
            st.code(report, language="text")