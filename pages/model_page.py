"""
pages/predict_page.py  ──  SteamML · Predict New Game
Full-featured page with:
  - Predict tab  : CSV paste / CSV upload / structured form → full pipeline → prediction
  - Retrain tab  : Train / load saved models (from model_page.py)
  - Feature Sel. : Feature selection pipeline (from model_page.py)

Preprocessing mirror
---------------------
1. Date engineering  (release_year / release_month / game_age_days)
2. Derived numeric   (discount_ratio, is_effectively_free, …)
3. Text-length flags (about_length, short_length, detail_length, …)
4. SteamSpy log transforms
5. Load saved TF-IDF + SVD  →  LSA features
6. Load saved StandardScaler  →  scale continuous features
7. Assemble final feature vector, align to model, predict
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
from sklearn.linear_model      import Ridge
from sklearn.tree              import DecisionTreeRegressor
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.model_selection   import RandomizedSearchCV, train_test_split
from sklearn.metrics           import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, mutual_info_regression
from sklearn.preprocessing     import StandardScaler
from scipy.stats               import loguniform, randint, uniform

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
MODELS_DIR = "./models"
SEL_DIR    = "./data/selected"
PROC_DIR   = "./data/processed"
TARGET     = "target_log"

MODEL_FILES = {
    "Ridge"            : "ridge_tuned.pkl",
    "Decision Tree"    : "decision_tree_tuned.pkl",
    "Random Forest"    : "random_forest_tuned.pkl",
    "Gradient Boosting": "gradient_boosting_tuned.pkl",
}
MODEL_COLORS = {
    "Ridge"            : "#4C8EDA",
    "Decision Tree"    : "#8D6E63",
    "Random Forest"    : "#1D9E75",
    "Gradient Boosting": "#E8593C",
}
MODEL_DESC = {
    "Ridge": (
        "Linear model with L2 regularisation. Fast, interpretable, strong baseline.",
        ["alpha", "fit_intercept", "solver"]
    ),
    "Decision Tree": (
        "Single tree; interpretable and fast. High variance without regularisation.",
        ["max_depth", "min_samples_leaf", "min_samples_split", "max_features", "criterion"]
    ),
    "Random Forest": (
        "Ensemble of decorrelated decision trees. Naturally captures non-linearities.",
        ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features"]
    ),
    "Gradient Boosting": (
        "Sequential boosting; each tree corrects the residuals of the previous one.",
        ["n_estimators", "learning_rate", "max_depth", "subsample", "min_samples_leaf"]
    ),
}

# Columns that should NOT be IQR-capped (log-transformed instead)
NO_IQR_COLS = [
    "RequiredAge", "DemoCount", "DeveloperCount",
    "DLCCount", "PackageCount", "PublisherCount",
]

# Continuous columns that the scaler was fitted on (order matters!)
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
    # NO_IQR_COLS (log-transformed versions)
    "RequiredAge", "DemoCount", "DeveloperCount",
    "DLCCount", "PackageCount", "PublisherCount",
    "ScreenshotCount",
]

# NLP field map: key → raw column name
NLP_FIELD_MAP = {
    "about"          : "AboutText",
    "short"          : "ShortDescrip",
    "detail"         : "DetailedDescrip",
    "reviews"        : "Reviews",
    "name"           : "ResponseName",
    "PCMinReqsText"  : "PCMinReqsText",
    "PCRecReqsText"  : "PCRecReqsText",
    "LinuxMinReqsText": "LinuxMinReqsText",
    "MacMinReqsText" : "MacMinReqsText",
}

# All raw input columns
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
    "MacMinReqsText","MacRecReqsText",
]

FORM_NUMERIC = {
    "Metacritic"               : (0,   100,  0),
    "PriceInitial"             : (0.0, 200.0, 9.99),
    "PriceFinal"               : (0.0, 200.0, 9.99),
    "RequiredAge"              : (0,   18,    0),
    "DLCCount"                 : (0,   200,   0),
    "MovieCount"               : (0,   50,    2),
    "ScreenshotCount"          : (0,   100,   5),
    "AchievementCount"         : (0,   2000,  0),
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
# TEXT CLEANING  (mirrors preprocess.py / nlp_features.py)
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


def _compute_metrics(model, X_tr, y_tr, X_te, y_te):
    X_tr = _align(model, X_tr)
    X_te = _align(model, X_te)
    p_tr = model.predict(X_tr)
    p_te = model.predict(X_te)

    def _m(y_true, y_pred):
        return dict(
            rmse=np.sqrt(mean_squared_error(y_true, y_pred)),
            mae=mean_absolute_error(y_true, y_pred),
            r2=r2_score(y_true, y_pred),
            rmse_orig=np.sqrt(mean_squared_error(np.expm1(y_true), np.expm1(y_pred))),
            mae_orig=mean_absolute_error(np.expm1(y_true), np.expm1(y_pred)),
        )
    return _m(y_tr, p_tr), _m(y_te, p_te), p_tr, p_te


# ─────────────────────────────────────────────────────────────────
# CORE PIPELINE: raw row dict → feature vector
# ─────────────────────────────────────────────────────────────────
def _preprocess_row(raw: dict, tfidf_vecs, svd_mods, scaler) -> pd.DataFrame:
    """
    Full mirror of preprocess.py + nlp_features.py for a single raw row.
    Returns a one-row DataFrame of processed + scaled features.
    """

    def flt(k, default=0.0):
        v = raw.get(k, default)
        try:
            return float(v) if v not in (None, "", "nan", float("nan")) else default
        except Exception:
            return default

    def txt(k):
        v = raw.get(k, "")
        return str(v) if v not in (None, float("nan")) else ""

    # ── 1. Date features ──────────────────────────────────────
    try:
        rd = pd.to_datetime(txt("ReleaseDate"), errors="coerce")
    except Exception:
        rd = pd.NaT

    release_year  = float(rd.year)  if pd.notna(rd) else 2020.0
    release_month = float(rd.month) if pd.notna(rd) else 6.0
    game_age_days = float((pd.Timestamp.today() - rd).days) if pd.notna(rd) else 1000.0

    # ── 2. Price / free features ───────────────────────────────
    price_initial = flt("PriceInitial", 0.0)
    price_final   = flt("PriceFinal",   0.0)
    is_free       = flt("IsFree", 0.0)

    discount_ratio      = float(np.clip((price_initial - price_final) / (price_initial + 1e-9), 0, 1))
    is_effectively_free = float((price_initial == 0) or (is_free == 1))

    # ── 3. Metacritic / language / presence flags ──────────────
    metacritic     = flt("Metacritic", 0.0)
    has_metacritic = float(metacritic > 0)

    lang_str      = txt("SupportedLanguages")
    num_languages = float(len([w for w in lang_str.split() if len(w) > 2]))

    has_website        = float(bool(txt("Website").strip()))
    has_support_email  = float(bool(txt("SupportEmail").strip()))
    has_support_url    = float(bool(txt("SupportURL").strip()))
    has_legal_notice   = float(len(txt("LegalNotice").strip())      > 1)
    has_reviews_text   = float(len(txt("Reviews").strip())          > 5)
    has_pc_min_reqs    = float(len(txt("PCMinReqsText").strip())    > 5)
    has_pc_rec_reqs    = float(len(txt("PCRecReqsText").strip())    > 5)
    has_linux_min_reqs = float(len(txt("LinuxMinReqsText").strip()) > 5)
    has_mac_min_reqs   = float(len(txt("MacMinReqsText").strip())   > 5)
    has_drm            = float(len(txt("DRMNotice").strip())        > 1)
    has_ext_account    = float(len(txt("ExtUserAcctNotice").strip()) > 1)

    about_length  = float(len(txt("AboutText")))
    short_length  = float(len(txt("ShortDescrip")))
    detail_length = float(len(txt("DetailedDescrip")))

    # ── 4. SteamSpy log transforms ────────────────────────────
    ss_own    = flt("SteamSpyOwners",          0)
    ss_ownv   = flt("SteamSpyOwnersVariance",  0)
    ss_play   = flt("SteamSpyPlayersEstimate", 0)
    ss_playv  = flt("SteamSpyPlayersVariance", 0)
    ss_own_log   = float(np.log1p(ss_own))
    ss_ownv_log  = float(np.log1p(ss_ownv))
    ss_play_log  = float(np.log1p(ss_play))
    ss_playv_log = float(np.log1p(ss_playv))

    # ── 5. NO_IQR log transforms ──────────────────────────────
    req_age     = float(np.log1p(flt("RequiredAge",    0)))
    demo_cnt    = float(np.log1p(flt("DemoCount",      0)))
    dev_cnt     = float(np.log1p(flt("DeveloperCount", 1)))
    dlc_cnt_log = float(np.log1p(flt("DLCCount",       0)))
    pkg_cnt     = float(np.log1p(flt("PackageCount",   1)))
    pub_cnt     = float(np.log1p(flt("PublisherCount", 1)))

    ach_count   = flt("AchievementCount", 0)
    movie_count = flt("MovieCount",       0)

    # ── 6. Interaction features ───────────────────────────────
    price_per_language     = price_final / (num_languages + 1)
    metacritic_x_age       = has_metacritic * game_age_days
    owners_per_achievement = ss_own_log / (ach_count + 1)
    dlc_x_owners           = dlc_cnt_log * ss_own_log
    movie_x_owners         = movie_count  * ss_own_log

    # ── 7. Assemble numeric row dict ──────────────────────────
    row_dict = {
        "RequiredAge"                  : req_age,
        "DemoCount"                    : demo_cnt,
        "DeveloperCount"               : dev_cnt,
        "DLCCount"                     : dlc_cnt_log,
        "Metacritic"                   : metacritic,
        "MovieCount"                   : movie_count,
        "PackageCount"                 : pkg_cnt,
        "PublisherCount"               : pub_cnt,
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
    }

    # Binary flags
    for bf in BINARY_FLAGS:
        row_dict[bf] = flt(bf, 0.0)

    feat_df = pd.DataFrame([row_dict])

    # ── 8. Scale continuous features (if scaler available) ────
    if scaler is not None:
        # Only scale columns the scaler was fitted on that exist in feat_df
        scaler_cols = [c for c in scaler.feature_names_in_ if c in feat_df.columns] \
                      if hasattr(scaler, "feature_names_in_") \
                      else [c for c in CONT_FEAT_COLS_FOR_SCALING if c in feat_df.columns]
        if scaler_cols:
            feat_df[scaler_cols] = scaler.transform(feat_df[scaler_cols])

    # ── 9. NLP / LSA features ─────────────────────────────────
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
st.set_page_config(page_title="SteamML · Predict", page_icon="🎯", layout="wide")

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
.result-box{background:linear-gradient(135deg,#0d0f14,#111827);border:1px solid var(--accent3);
    border-radius:12px;padding:28px;text-align:center;margin-top:20px;}
.result-box .big{font-size:52px;font-weight:700;font-family:var(--mono);color:var(--accent3);line-height:1;}
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
.model-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
    padding:18px 22px;margin-bottom:12px;}
.model-card h3{margin:0 0 6px;font-size:14px;font-family:var(--mono);color:var(--text);}
.model-card p{margin:0;font-size:11px;color:var(--muted);line-height:1.7;}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:10px;
    font-family:var(--mono);font-weight:700;margin-right:6px;}
.status-ok{color:#5bf6c8;font-family:monospace;font-size:11px;}
.status-no{color:#6b7280;font-family:monospace;font-size:11px;}
.warn-box{background:#13161e;border:1px solid #f6c85b44;border-left:3px solid #f6c85b;
    border-radius:8px;padding:10px 14px;font-size:11px;color:#f6c85b;margin-bottom:12px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
tfidf_vecs, svd_mods = _load_nlp_models()
scaler               = _load_scaler()
nlp_ok               = tfidf_vecs is not None
scaler_ok            = scaler is not None
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
    st.markdown(f"<div style='font-size:12px;line-height:2.2'>"
                f"<span style='color:{c}'>{d}</span>&nbsp;NLP models {'loaded ✓' if nlp_ok else 'not found'}<br>",
                unsafe_allow_html=True)
    d, c = _dot(scaler_ok)
    st.markdown(f"<span style='color:{c}'>{d}</span>&nbsp;Scaler {'loaded ✓' if scaler_ok else 'not found (predictions still work)'}<br>",
                unsafe_allow_html=True)
    d, c = _dot(bool(saved_models))
    st.markdown(f"<span style='color:{c}'>{d}</span>&nbsp;{len(saved_models)} / {len(MODEL_FILES)} models saved</div>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""<div class="ph">
    <h1>🎯  SteamML · Predict &amp; Manage Models</h1>
    <p>Predict new game recommendations · Feature selection · Train &amp; compare models</p>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────
tab_predict, tab_fs, tab_train, tab_compare = st.tabs([
    "🎯  PREDICT NEW GAME",
    "🔍  FEATURE SELECTION",
    "⚙️  TRAIN / LOAD MODELS",
    "📊  MODEL COMPARISON",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT NEW GAME
# ══════════════════════════════════════════════════════════════════
with tab_predict:
    if not saved_models:
        st.error("⚠️  No trained models found. Go to **Train / Load Models** and train at least one first.")
        st.stop()

    if not nlp_ok:
        st.markdown(
            "<div class='warn-box'>⚠️  NLP models not found in ./models/. "
            "LSA text features will be zero. Run the NLP preprocessing pipeline first for best accuracy.</div>",
            unsafe_allow_html=True,
        )
    if not scaler_ok:
        st.markdown(
            "<div class='warn-box'>⚠️  Scaler (scaler.pkl) not found in ./models/. "
            "Continuous features will NOT be scaled — retrain preprocessing to save scaler.</div>",
            unsafe_allow_html=True,
        )

    # ── Model selector ─────────────────────────────────────────
    col_ms, col_info = st.columns([2, 3], gap="large")
    with col_ms:
        sel_model = st.selectbox("Model", saved_models, key="pred_model_tab")
    with col_info:
        mc = MODEL_COLORS.get(sel_model, "#5b8df6")
        pred_model = _load_model(sel_model)
        feat_names = _get_feature_names(pred_model)
        n_feats    = len(feat_names) if feat_names else "?"
        st.markdown(
            f"<div style='background:var(--surface);border:1px solid var(--border);"
            f"border-left:4px solid {mc};border-radius:8px;padding:12px 16px;margin-top:22px;"
            f"font-size:11px;color:var(--muted);font-family:var(--mono)'>"
            f"<span style='color:{mc};font-weight:700'>{sel_model}</span> · "
            f"{n_feats} features · pipeline: numeric + IQR + log + scale + LSA"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Input tabs ─────────────────────────────────────────────
    itab_form, itab_paste, itab_upload = st.tabs([
        "📝  FILL FORM",
        "📋  PASTE CSV ROW",
        "📁  UPLOAD CSV FILE",
    ])

    raw_input: dict | None = None

    # ── TAB A: Structured form ─────────────────────────────────
    with itab_form:
        st.markdown(
            "<div class='tip'>Fill in the game data below. Text fields (descriptions, "
            "requirements) are used for NLP/LSA features. Leave unknown fields at defaults.</div>",
            unsafe_allow_html=True,
        )
        form_data = {}

        # Game identity & text
        st.markdown('<div class="sec"><div class="sec-dot" style="background:#f6c85b"></div>'
                    '<div class="sec-lbl">Game Identity &amp; Text</div></div>', unsafe_allow_html=True)
        ct1, ct2 = st.columns(2)
        with ct1:
            form_data["ResponseName"]       = st.text_input("Game Name",       value="My Awesome Game", key="f_name")
            form_data["ReleaseDate"]         = st.text_input("Release Date",    value="2022-06-15",      key="f_date")
            form_data["SupportedLanguages"]  = st.text_input("Supported Languages (space-sep)",
                                                              value="English French German Spanish", key="f_lang")
            form_data["Website"]             = st.text_input("Website URL",     value="", key="f_web")
            form_data["SupportEmail"]        = st.text_input("Support Email",   value="", key="f_email")
        with ct2:
            form_data["AboutText"]           = st.text_area("About Text",       value="", height=80,  key="f_about")
            form_data["ShortDescrip"]        = st.text_area("Short Description",value="", height=60,  key="f_short")

        form_data["DetailedDescrip"]         = st.text_area("Detailed Description", value="", height=100, key="f_detail")
        form_data["Reviews"]                 = st.text_area("Reviews Text",         value="", height=60,  key="f_reviews")

        pc1, pc2 = st.columns(2)
        with pc1:
            form_data["PCMinReqsText"]       = st.text_area("PC Min Requirements",
                                                             value="OS: Windows 10\nCPU: Intel i3\nRAM: 4 GB",
                                                             height=70, key="f_pcmin")
        with pc2:
            form_data["PCRecReqsText"]       = st.text_area("PC Rec Requirements",
                                                             value="OS: Windows 11\nCPU: Intel i7\nRAM: 16 GB",
                                                             height=70, key="f_pcrec")

        # Numeric attributes
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

        # Boolean flags
        st.markdown('<div class="sec"><div class="sec-dot" style="background:#5bf6c8"></div>'
                    '<div class="sec-lbl">Flags &amp; Categories</div></div>', unsafe_allow_html=True)
        for brow in [FORM_BOOL[i:i+5] for i in range(0, len(FORM_BOOL), 5)]:
            bcols = st.columns(len(brow))
            for bcol, bf in zip(bcols, brow):
                form_data[bf] = int(bcol.checkbox(bf, value=False, key=f"form_{bf}"))

        # Defaults for all other raw cols
        for c in RAW_COLS:
            if c not in form_data:
                form_data[c] = ""

        _, btn_col, _ = st.columns([2, 2, 2])
        with btn_col:
            if st.button("🎯  PREDICT FROM FORM", key="btn_form"):
                raw_input = form_data

    # ── TAB B: Paste CSV ───────────────────────────────────────
    with itab_paste:
        st.markdown(
            "<div class='tip'>"
            "Paste either:<br>"
            "① A header row + one data row (comma-separated, original raw columns)<br>"
            "② Just one data row — matched positionally to the expected schema"
            "</div>",
            unsafe_allow_html=True,
        )
        example_header = ",".join(RAW_COLS)
        pasted = st.text_area(
            "Paste CSV here",
            height=180,
            placeholder=f"Paste header + data row:\n{example_header}\n\n0,0,,,2022-06-01,0,0,1,0,...",
            key="paste_area",
        )
        _, btn_col2, _ = st.columns([2, 2, 2])
        with btn_col2:
            if st.button("🎯  PREDICT FROM PASTE", key="btn_paste"):
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
            "Upload a CSV file with the same columns as <code>train_data.csv</code> (raw, before preprocessing). "
            "If the file has multiple rows, the <b>first data row</b> is used."
            "</div>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader("Upload CSV (one game row)", type=["csv"], key="upload_csv")
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded)
                st.success(f"File loaded · {df_up.shape[0]} rows · {df_up.shape[1]} columns")
                if df_up.shape[0] > 1:
                    st.info(f"Multiple rows detected — using row 1 of {df_up.shape[0]}.")
                st.dataframe(df_up.head(3), use_container_width=True)
                _, btn_col3, _ = st.columns([2, 2, 2])
                with btn_col3:
                    if st.button("🎯  PREDICT FROM FILE", key="btn_upload"):
                        raw_input = df_up.iloc[0].to_dict()
            except Exception as e:
                st.error(f"Could not read file: {e}")

    # ── PREDICTION EXECUTION ───────────────────────────────────
    if raw_input is not None:
        st.markdown("---")
        st.markdown('<div class="sec"><div class="sec-dot" style="background:#5bf6c8"></div>'
                    '<div class="sec-lbl">Pipeline Execution</div></div>', unsafe_allow_html=True)

        # Accumulate steps as (message, kind) tuples; render once at the end
        # Kinds: "done" | "warn" | "err"
        _steps: list[tuple[str, str]] = []
        success   = False
        log_pred  = 0.0
        orig_pred = 0
        X_new     = None

        def _render_steps(steps):
            """Build one HTML string from all accumulated steps and write it."""
            parts = []
            for msg, kind in steps:
                if kind == "done":
                    style = "border-left:3px solid #5bf6c8;color:#5bf6c8"
                    icon  = "✓"
                elif kind == "warn":
                    style = "border-left:3px solid #f6c85b;color:#f6c85b"
                    icon  = "⚠"
                else:                        # err
                    style = "border-left:3px solid #f65b8d;color:#f65b8d"
                    icon  = "✗"
                safe_msg = str(msg).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                parts.append(
                    f'<div class="pipeline-step" style="{style}">{icon} {safe_msg}</div>'
                )
            html = "\n".join(parts)
            pipeline_placeholder.markdown(html, unsafe_allow_html=True)

        pipeline_placeholder = st.empty()

        try:
            _steps.append(("Date engineering (release_year / month / game_age_days)", "done"))
            _steps.append(("Numeric feature engineering (discount_ratio, log transforms, interactions…)", "done"))

            if nlp_ok:
                _steps.append(("TF-IDF + LSA applied to text fields", "done"))
            else:
                _steps.append(("NLP models missing — LSA features set to 0", "warn"))

            if scaler_ok:
                _steps.append(("StandardScaler loaded — continuous features will be scaled", "done"))
            else:
                _steps.append(("Scaler (scaler.pkl) not found — features will not be scaled", "warn"))

            _render_steps(_steps)

            feat_df = _preprocess_row(raw_input, tfidf_vecs, svd_mods, scaler)
            _steps.append((f"Feature vector assembled — {feat_df.shape[1]} columns before alignment", "done"))
            _render_steps(_steps)

            X_new = _align(pred_model, feat_df)
            _steps.append((f"Feature alignment → {X_new.shape[1]} features matched to model", "done"))
            _render_steps(_steps)

            log_pred  = float(pred_model.predict(X_new)[0])
            orig_pred = int(np.round(np.expm1(log_pred)))
            _steps.append((f"Prediction complete: log={log_pred:.4f} → {orig_pred:,} recommendations", "done"))
            _render_steps(_steps)
            success = True

        except Exception as e:
            _steps.append((f"ERROR: {e}", "err"))
            _render_steps(_steps)
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())

        if success:
            mc = MODEL_COLORS.get(sel_model, "#5bf6c8")
            st.markdown(
                f"<div class='result-box'>"
                f"<div style='font-size:11px;color:var(--muted);font-family:var(--mono);"
                f"margin-bottom:10px'>{sel_model}</div>"
                f"<div class='big'>{orig_pred:,}</div>"
                f"<div class='sub'>predicted Steam recommendations</div>"
                f"<div style='margin-top:18px;display:flex;justify-content:center;gap:50px'>"
                f"<div><div style='font-size:22px;font-family:var(--mono);color:{mc}'>{log_pred:.4f}</div>"
                f"<div style='font-size:10px;color:var(--muted)'>log-space</div></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

            # Context metrics vs training set
            try:
                train_sel_path = os.path.join(SEL_DIR, "train_selected.csv")
                if os.path.exists(train_sel_path):
                    train_ref   = pd.read_csv(train_sel_path)
                    X_ref       = train_ref.drop(columns=[TARGET], errors="ignore")
                    X_ref_align = _align(pred_model, X_ref)
                    train_preds = pred_model.predict(X_ref_align)
                    train_orig  = np.expm1(train_preds)
                    pct         = float((train_preds < log_pred).mean() * 100)

                    st.markdown('<div class="sec"><div class="sec-dot" style="background:#f6c85b"></div>'
                                '<div class="sec-lbl">Prediction Context</div></div>', unsafe_allow_html=True)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Your Game",    f"{orig_pred:,}")
                    c2.metric("Train Mean",   f"{int(train_orig.mean()):,}")
                    c3.metric("Train Median", f"{int(np.median(train_orig)):,}")
                    c4.metric("Percentile",   f"{pct:.1f}%")

                    fig, ax = plt.subplots(figsize=(12, 3.5), facecolor="#13161e")
                    ax.set_facecolor("#1a1e2b")
                    ax.hist(train_preds, bins=70, color="#4C8EDA", alpha=0.75, edgecolor="none", label="Training games")
                    ax.axvline(log_pred, color="#5bf6c8", lw=2.2, linestyle="--",
                               label=f"Your game ({log_pred:.3f})")
                    ax.set_title("Your game vs training distribution (log-space)",
                                 color="#e8eaf2", fontsize=9, fontfamily="monospace")
                    ax.set_xlabel("Predicted log(recommendations)", fontsize=8, color="#6b7280")
                    ax.tick_params(colors="#6b7280", labelsize=7)
                    for sp in ax.spines.values():
                        sp.set_color("#252a38")
                    ax.legend(fontsize=8, framealpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
            except Exception:
                pass

            with st.expander("🔍  Show processed feature vector"):
                st.dataframe(X_new.T.rename(columns={0: "value"}), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════
with tab_fs:
    st.markdown(
        "<div style='background:#13161e;border:1px solid #252a38;border-radius:10px;"
        "padding:16px 20px;font-size:12px;color:#6b7280;line-height:2;margin-bottom:20px'>"
        "Pipeline: <b style='color:#e8eaf2'>Dominant-value filter</b> → "
        "<b style='color:#e8eaf2'>VarianceThreshold</b> → "
        "<b style='color:#e8eaf2'>High inter-feature correlation</b> → "
        "<b style='color:#e8eaf2'>Low target-correlation</b> → "
        "<b style='color:#e8eaf2'>RandomForest SelectFromModel (median)</b> → "
        "Save to <code>./data/selected/</code>"
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
        st.warning("⚠️  Run the **Preprocessing** pipeline first to generate processed CSVs.")

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
            "① Merge processed + NLP CSVs (if available)<br>"
            "② Dominant-value filter  (≥ 95%)<br>"
            "③ VarianceThreshold  (var &lt; 0.01)<br>"
            "④ High inter-feature corr  (|r| &gt; 0.90)<br>"
            "⑤ Low target-corr  (|r| &lt; 0.01)<br>"
            "⑥ RF SelectFromModel  (threshold = median)<br>"
            "⑦ Save  →  ./data/selected/"
            "</div>",
            unsafe_allow_html=True,
        )

    if run_fs:
        log_lines = []
        L = lambda m, k="ok": log_lines.append((m, k))

        def render_log(lines):
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
            L("// Feature Selection started ─────────────")

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
                L(f"  train={train_df.shape}  test={test_df.shape}")

            val_path = os.path.join(PROC_DIR, "val.csv")
            has_val  = os.path.exists(val_path)
            val_df   = None
            if has_val:
                val_num = pd.read_csv(val_path)
                if nlp_avail and os.path.exists(os.path.join(PROC_DIR, "nlp_features_val.csv")):
                    val_nlp = pd.read_csv(os.path.join(PROC_DIR, "nlp_features_val.csv"))
                    val_df  = merge_split(val_num, val_nlp)
                else:
                    val_df = merge_split(val_num)

            X_train = train_df.drop(columns=[TARGET]); y_train = train_df[TARGET]
            X_test  = test_df.drop(columns=[TARGET]);  y_test  = test_df[TARGET]
            if has_val:
                X_val = val_df.drop(columns=[TARGET]); y_val = val_df[TARGET]

            initial = X_train.shape[1]

            # ① Dominant-value filter
            dom_drop = [c for c in X_train.columns
                        if X_train[c].value_counts(normalize=True, dropna=False).max() >= 0.95]
            for df_ in ([X_train, X_test] + ([X_val] if has_val else [])):
                df_.drop(columns=dom_drop, inplace=True)
            L(f"  [①] dominant-value  → dropped {len(dom_drop):3d}  remaining: {X_train.shape[1]}", "muted")

            # ② VarianceThreshold
            cols_bvt = X_train.columns.tolist()
            vt       = VarianceThreshold(threshold=0.01)
            Xtr_a    = vt.fit_transform(X_train)
            Xte_a    = vt.transform(X_test)
            sel_vt   = [c for c, k in zip(cols_bvt, vt.get_support()) if k]
            X_train  = pd.DataFrame(Xtr_a, columns=sel_vt)
            X_test   = pd.DataFrame(Xte_a, columns=sel_vt)
            if has_val:
                X_val = pd.DataFrame(vt.transform(X_val), columns=sel_vt)
            L(f"  [②] VarianceThreshold → dropped {len(cols_bvt) - len(sel_vt):3d}  remaining: {X_train.shape[1]}", "muted")

            # ③ High inter-feature correlation
            corr_mat  = X_train.corr().abs()
            upper     = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
            tgt_corr  = X_train.corrwith(y_train).abs()
            corr_drop = []
            for col in upper.columns:
                for partner in upper.index[upper[col] > 0.90].tolist():
                    if col not in corr_drop and partner not in corr_drop:
                        drop_col = col if tgt_corr.get(col, 0) < tgt_corr.get(partner, 0) else partner
                        corr_drop.append(drop_col)
            for df_ in ([X_train, X_test] + ([X_val] if has_val else [])):
                df_.drop(columns=corr_drop, inplace=True, errors="ignore")
            L(f"  [③] high corr       → dropped {len(corr_drop):3d}  remaining: {X_train.shape[1]}", "muted")

            # ④ Low target-correlation
            low_drop = X_train.corrwith(y_train).abs()
            low_drop = low_drop[low_drop < 0.01].index.tolist()
            for df_ in ([X_train, X_test] + ([X_val] if has_val else [])):
                df_.drop(columns=low_drop, inplace=True)
            L(f"  [④] low target-corr → dropped {len(low_drop):3d}  remaining: {X_train.shape[1]}", "muted")

            # ⑤ RF SelectFromModel
            L("  [⑤] fitting RandomForest for importance …")
            rf_sel = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_sel.fit(X_train, y_train)
            all_feats = X_train.columns.tolist()
            imp_df    = pd.DataFrame({
                "feature": all_feats, "importance": rf_sel.feature_importances_
            }).sort_values("importance", ascending=False)
            selector  = SelectFromModel(rf_sel, threshold="median", prefit=True)
            sel_rf    = [c for c, k in zip(all_feats, selector.get_support()) if k]
            X_train   = pd.DataFrame(selector.transform(X_train), columns=sel_rf)
            X_test    = pd.DataFrame(selector.transform(X_test),  columns=sel_rf)
            if has_val:
                X_val = pd.DataFrame(selector.transform(X_val), columns=sel_rf)
            L(f"  [⑤] SelectFromModel → kept {len(sel_rf)}  dropped {len(all_feats) - len(sel_rf)}", "muted")

            # Save
            os.makedirs(SEL_DIR, exist_ok=True)
            out_tr = X_train.copy(); out_tr[TARGET] = y_train.values; out_tr.to_csv(os.path.join(SEL_DIR, "train_selected.csv"), index=False)
            out_te = X_test.copy();  out_te[TARGET] = y_test.values;  out_te.to_csv(os.path.join(SEL_DIR, "test_selected.csv"),  index=False)
            if has_val:
                out_va = X_val.copy(); out_va[TARGET] = y_val.values; out_va.to_csv(os.path.join(SEL_DIR, "val_selected.csv"), index=False)
            L("  saved to ./data/selected/")
            L(f"  initial={initial}  final={X_train.shape[1]}  removed={initial - X_train.shape[1]}")
            L("// Feature Selection Done ✓ ──────────────")

            st.session_state["fs_imp_df"]    = imp_df
            st.session_state["fs_sel_feats"] = sel_rf

        except Exception as e:
            L(f"ERROR: {e}", "err")
            L(traceback.format_exc(), "err")

        render_log(log_lines)
        st.rerun()

    if _data_ready():
        train_sel = pd.read_csv(os.path.join(SEL_DIR, "train_selected.csv"))
        test_sel  = pd.read_csv(os.path.join(SEL_DIR, "test_selected.csv"))
        feat_cols = [c for c in train_sel.columns if c != TARGET]

        st.markdown('<div class="sec"><div class="sec-dot" style="background:#5bf6c8"></div>'
                    '<div class="sec-lbl">Selection Summary</div></div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Features", len(feat_cols))
        c2.metric("Train Rows",     f"{len(train_sel):,}")
        c3.metric("Test Rows",      f"{len(test_sel):,}")
        c4.metric("Target",         "target_log (log1p)")

        st.markdown('<div class="sec"><div class="sec-dot" style="background:#5b8df6"></div>'
                    '<div class="sec-lbl">Feature List</div></div>', unsafe_allow_html=True)
        feat_display = pd.DataFrame({
            "Feature": feat_cols,
            "Type":    ["NLP/LSA" if "lsa" in f else "Numeric" for f in feat_cols],
        })
        st.dataframe(feat_display, use_container_width=True, height=280)

        imp_df = st.session_state.get("fs_imp_df")
        if imp_df is not None:
            st.markdown('<div class="sec"><div class="sec-dot" style="background:#f6c85b"></div>'
                        '<div class="sec-lbl">RandomForest Feature Importances — Top 30</div></div>', unsafe_allow_html=True)
            top30 = imp_df.head(30)
            fig, ax = plt.subplots(figsize=(14, 7), facecolor="#13161e")
            ax.set_facecolor("#1a1e2b")
            clrs = [MODEL_COLORS["Random Forest"] if "lsa" not in f else "#7F77DD" for f in top30["feature"]]
            ax.barh(range(len(top30)), top30["importance"].values[::-1], color=clrs[::-1], alpha=0.85, height=0.75)
            ax.set_yticks(range(len(top30)))
            ax.set_yticklabels(top30["feature"].values[::-1], fontsize=8, color="#e8eaf2")
            ax.set_title("Top 30 Feature Importances (green=numeric, purple=NLP/LSA)",
                         color="#e8eaf2", fontsize=9, fontfamily="monospace")
            ax.set_xlabel("Importance", fontsize=8, color="#6b7280")
            ax.tick_params(colors="#6b7280", labelsize=7)
            for sp in ax.spines.values(): sp.set_color("#252a38")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()


# ══════════════════════════════════════════════════════════════════
# TAB 3 — TRAIN / LOAD MODELS
# ══════════════════════════════════════════════════════════════════
with tab_train:
    if not _data_ready():
        st.warning("⚠️  Run **Feature Selection** first to generate selected feature CSVs.")
    else:
        @st.cache_data
        def _load_train_data():
            tr = pd.read_csv(os.path.join(SEL_DIR, "train_selected.csv"))
            te = pd.read_csv(os.path.join(SEL_DIR, "test_selected.csv"))
            X_tr = tr.drop(columns=[TARGET]); y_tr = tr[TARGET]
            X_te = te.drop(columns=[TARGET]); y_te = te[TARGET]
            vp = os.path.join(SEL_DIR, "val_selected.csv")
            if os.path.exists(vp):
                va  = pd.read_csv(vp)
                X_va = va.drop(columns=[TARGET]); y_va = va[TARGET]
                X_tv = pd.concat([X_tr, X_va]).reset_index(drop=True)
                y_tv = pd.concat([y_tr, y_va]).reset_index(drop=True)
            else:
                X_tv, y_tv = X_tr.reset_index(drop=True), y_tr.reset_index(drop=True)
            return X_tr, y_tr, X_te, y_te, X_tv, y_tv

        X_train, y_train, X_test, y_test, X_tv, y_tv = _load_train_data()

        st.markdown(
            f"<div style='background:#13161e;border:1px solid #252a38;border-radius:10px;"
            f"padding:14px 20px;font-size:12px;color:#6b7280;margin-bottom:18px'>"
            f"Data loaded · train+val = <b style='color:#e8eaf2'>{len(X_tv):,}</b> rows · "
            f"test = <b style='color:#e8eaf2'>{len(X_test):,}</b> rows · "
            f"features = <b style='color:#e8eaf2'>{X_train.shape[1]}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

        def render_log(lines):
            html = "".join(
                '<div style="{c}">{m}</div>'.format(
                    c=('color:#6b7280' if k == 'muted' else
                       'color:#f65b8d' if k == 'err' else 'color:#5bf6c8'),
                    m=m.replace('<', '&lt;').replace('>', '&gt;')
                )
                for m, k in lines
            )
            st.markdown(f'<div class="log-box">{html}</div>', unsafe_allow_html=True)

        for name, (desc, hp_list) in MODEL_DESC.items():
            col_hex = MODEL_COLORS[name]
            exists  = _model_exists(name)
            label   = f"{'✅' if exists else '⬜'}  {name}  {'— model saved ✓' if exists else '— not trained yet'}"

            with st.expander(label, expanded=False):
                cl, cr = st.columns([3, 1], gap="large")
                with cl:
                    badges = "".join(
                        f"<span class='badge' style='background:{col_hex}22;"
                        f"color:{col_hex};border:1px solid {col_hex}44'>{h}</span>"
                        for h in hp_list
                    )
                    st.markdown(
                        f"<div class='model-card'>"
                        f"<h3 style='color:{col_hex}'>{name}</h3>"
                        f"<p>{desc}</p>"
                        f"<div style='margin-top:10px'>{badges}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with cr:
                    btn_label = f"🔄  {'Re-train' if exists else 'Train'} {name}"
                    btn_train = st.button(btn_label, key=f"train_{name}")
                    if exists:
                        st.markdown("<span class='status-ok'>● saved</span>", unsafe_allow_html=True)
                        try:
                            m = _load_model(name)
                            params = list(m.get_params().items())[:6]
                            param_html = "<br>".join(
                                f"{k}: <span style='color:#5bf6c8'>{v}</span>"
                                for k, v in params
                            )
                            st.markdown(
                                f"<div style='font-size:10px;color:#6b7280;margin-top:6px'>"
                                f"<b style='color:#e8eaf2'>Best params:</b><br>{param_html}</div>",
                                unsafe_allow_html=True,
                            )
                        except Exception:
                            pass
                    else:
                        st.markdown("<span class='status-no'>○ not trained</span>", unsafe_allow_html=True)

                if btn_train:
                    log_lines = []
                    L = lambda m, k="ok": log_lines.append((m, k))
                    with st.spinner(f"Training {name} …"):
                        try:
                            os.makedirs(MODELS_DIR, exist_ok=True)
                            L(f"// Training {name} ─────────────────────")

                            if name == "Ridge":
                                param_dist = {"alpha": loguniform(1e-3, 1e4),
                                              "fit_intercept": [True, False],
                                              "solver": ["auto", "svd", "cholesky", "lsqr", "saga"]}
                                base = Ridge()
                            elif name == "Decision Tree":
                                param_dist = {"max_depth": [3,4,5,6,8,10,15,None],
                                              "min_samples_split": randint(2, 40),
                                              "min_samples_leaf": randint(1, 30),
                                              "max_features": ["sqrt","log2",0.5,0.7,None],
                                              "criterion": ["squared_error","friedman_mse","absolute_error"]}
                                base = DecisionTreeRegressor(random_state=42)
                            elif name == "Random Forest":
                                param_dist = {"n_estimators": randint(100, 600),
                                              "max_depth": [None,5,10,15,20,30],
                                              "min_samples_leaf": randint(1, 30),
                                              "min_samples_split": randint(2, 20),
                                              "max_features": ["sqrt","log2",0.3,0.5,0.7]}
                                base = RandomForestRegressor(random_state=42, n_jobs=-1)
                            else:
                                param_dist = {"n_estimators": randint(100, 600),
                                              "learning_rate": loguniform(0.01, 0.3),
                                              "max_depth": randint(2, 8),
                                              "subsample": uniform(0.6, 0.4),
                                              "min_samples_leaf": randint(5, 40),
                                              "max_features": ["sqrt","log2",0.5,0.7,None]}
                                base = GradientBoostingRegressor(random_state=42)

                            search = RandomizedSearchCV(
                                base, param_dist, n_iter=40, cv=5,
                                scoring="r2", n_jobs=-1, random_state=42, verbose=0
                            )
                            search.fit(X_tv, y_tv)
                            best  = search.best_estimator_
                            cv_r2 = search.best_score_
                            L(f"  CV R²  = {cv_r2:.4f}")
                            L(f"  params = {search.best_params_}", "muted")

                            X_te_aligned = _align(best, X_test)
                            p_te    = best.predict(X_te_aligned)
                            r2_te   = r2_score(y_test, p_te)
                            rmse_te = np.sqrt(mean_squared_error(y_test, p_te))
                            L(f"  test R² = {r2_te:.4f}  RMSE = {rmse_te:.4f}")

                            joblib.dump(best, os.path.join(MODELS_DIR, MODEL_FILES[name]))
                            L(f"  saved → {MODEL_FILES[name]}")
                            L("// Done ✓ ──────────────────────────────")

                        except Exception as e:
                            L(f"ERROR: {e}", "err")
                            L(traceback.format_exc(), "err")
                    render_log(log_lines)
                    st.rerun()


# ══════════════════════════════════════════════════════════════════
# TAB 4 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════
with tab_compare:
    saved_now = {n: n for n in MODEL_FILES if _model_exists(n)}

    if not saved_now:
        st.warning("⚠️  No saved models found. Train at least one model first.")
    elif not _data_ready():
        st.warning("⚠️  Feature selection data not found.")
    else:
        @st.cache_data
        def _load_cmp_data():
            tr = pd.read_csv(os.path.join(SEL_DIR, "train_selected.csv"))
            te = pd.read_csv(os.path.join(SEL_DIR, "test_selected.csv"))
            X_tr = tr.drop(columns=[TARGET]); y_tr = tr[TARGET]
            X_te = te.drop(columns=[TARGET]); y_te = te[TARGET]
            vp = os.path.join(SEL_DIR, "val_selected.csv")
            if os.path.exists(vp):
                va  = pd.read_csv(vp)
                X_va = va.drop(columns=[TARGET]); y_va = va[TARGET]
                X_tv = pd.concat([X_tr, X_va]).reset_index(drop=True)
                y_tv = pd.concat([y_tr, y_va]).reset_index(drop=True)
            else:
                X_tv, y_tv = X_tr.reset_index(drop=True), y_tr.reset_index(drop=True)
            return X_tv, y_tv, X_te, y_te

        X_tv_c, y_tv_c, X_te_c, y_te_c = _load_cmp_data()

        results, skipped = {}, []
        for name in saved_now:
            try:
                m = _load_model(name)
                tr_m, te_m, p_tr, p_te = _compute_metrics(m, X_tv_c, y_tv_c, X_te_c, y_te_c)
                results[name] = dict(train=tr_m, test=te_m, p_tr=p_tr, p_te=p_te, model=m)
            except Exception as e:
                skipped.append((name, str(e)))

        for sn, se in skipped:
            st.warning(f"⚠️  Skipped **{sn}**: {se}")

        if results:
            st.markdown('<div class="sec"><div class="sec-dot" style="background:#5bf6c8"></div>'
                        '<div class="sec-lbl">Performance Summary — Test Set</div></div>', unsafe_allow_html=True)
            rows = []
            for name, res in results.items():
                te, tr = res["test"], res["train"]
                rows.append({"Model": name,
                             "R² Test": round(te["r2"], 4), "R² Train": round(tr["r2"], 4),
                             "Gap": round(tr["r2"] - te["r2"], 4),
                             "RMSE (log)": round(te["rmse"], 4), "MAE (log)": round(te["mae"], 4),
                             "RMSE (orig)": f"{te['rmse_orig']:,.0f}", "MAE (orig)": f"{te['mae_orig']:,.0f}"})
            df_res = pd.DataFrame(rows).sort_values("R² Test", ascending=False)
            st.dataframe(df_res, use_container_width=True, hide_index=True)

            best_name  = df_res.iloc[0]["Model"]
            best_r2    = df_res.iloc[0]["R² Test"]
            best_color = MODEL_COLORS.get(best_name, "#5bf6c8")
            st.markdown(
                f"<div style='margin-top:8px;font-size:12px;font-family:monospace'>"
                f"🏆 Best model: <span style='color:{best_color};font-weight:700'>{best_name}</span>"
                f" · test R² = <span style='color:#5bf6c8'>{best_r2}</span></div>",
                unsafe_allow_html=True,
            )

            # Bar charts
            st.markdown('<div class="sec"><div class="sec-dot" style="background:#f6c85b"></div>'
                        '<div class="sec-lbl">Metric Comparison</div></div>', unsafe_allow_html=True)
            names  = list(results.keys())
            colors = [MODEL_COLORS.get(n, "#888") for n in names]

            fig, axs = plt.subplots(1, 3, figsize=(18, 5), facecolor="#13161e")
            for ax, (metric, label) in zip(axs, [("r2","R² (↑)"), ("rmse","RMSE log (↓)"), ("mae","MAE log (↓)")]):
                ax.set_facecolor("#1a1e2b")
                vals = [results[n]["test"][metric] for n in names]
                bars = ax.bar(names, vals, color=colors, alpha=0.85, width=0.55)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                            f"{v:.4f}", ha="center", va="bottom", fontsize=7, color="#e8eaf2", rotation=45)
                ax.set_title(label, color="#e8eaf2", fontsize=9, fontfamily="monospace")
                ax.tick_params(labelsize=7, colors="#6b7280")
                ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
                ax.set_ylim(0, max(vals)*1.22)
                for sp in ax.spines.values(): sp.set_color("#252a38")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Actual vs Predicted
            st.markdown('<div class="sec"><div class="sec-dot" style="background:#5b8df6"></div>'
                        '<div class="sec-lbl">Actual vs Predicted — Test Set</div></div>', unsafe_allow_html=True)
            n_m = len(results)
            fig, axs = plt.subplots(1, n_m, figsize=(5*n_m, 5), facecolor="#13161e")
            axs = [axs] if n_m == 1 else list(axs)
            for ax, (name, res) in zip(axs, results.items()):
                ax.set_facecolor("#1a1e2b")
                col_h = MODEL_COLORS.get(name, "#888")
                ax.scatter(y_te_c, res["p_te"], alpha=0.2, s=7, color=col_h, edgecolors="none")
                lo = min(float(y_te_c.min()), float(res["p_te"].min()))
                hi = max(float(y_te_c.max()), float(res["p_te"].max()))
                ax.plot([lo, hi], [lo, hi], color="#e8eaf2", lw=1.2, linestyle="--")
                ax.set_title(f"{name}\nR²={res['test']['r2']:.4f}", color="#e8eaf2", fontsize=9, fontfamily="monospace")
                ax.set_xlabel("Actual (log)", fontsize=8, color="#6b7280")
                ax.set_ylabel("Predicted (log)", fontsize=8, color="#6b7280")
                ax.tick_params(colors="#6b7280", labelsize=7)
                for sp in ax.spines.values(): sp.set_color("#252a38")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Residuals
            st.markdown('<div class="sec"><div class="sec-dot" style="background:#7F77DD"></div>'
                        '<div class="sec-lbl">Residual Distribution — Test Set</div></div>', unsafe_allow_html=True)
            fig, axs = plt.subplots(1, n_m, figsize=(5*n_m, 4), facecolor="#13161e")
            axs = [axs] if n_m == 1 else list(axs)
            for ax, (name, res) in zip(axs, results.items()):
                ax.set_facecolor("#1a1e2b")
                col_h = MODEL_COLORS.get(name, "#888")
                resid = np.array(y_te_c) - res["p_te"]
                ax.hist(resid, bins=60, color=col_h, alpha=0.85, edgecolor="none")
                ax.axvline(0,            color="#e8eaf2", lw=1.2, linestyle="--", label="zero")
                ax.axvline(resid.mean(), color="#5bf6c8", lw=1.0, label=f"mean={resid.mean():.3f}")
                ax.set_title(name, color="#e8eaf2", fontsize=9, fontfamily="monospace")
                ax.set_xlabel("Residual", fontsize=8, color="#6b7280")
                ax.legend(fontsize=7, framealpha=0.3)
                ax.tick_params(colors="#6b7280", labelsize=7)
                for sp in ax.spines.values(): sp.set_color("#252a38")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()