import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, os, warnings, joblib
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

st.set_page_config(page_title="Preprocessing · SteamML", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{--bg:#0d0f14;--surface:#13161e;--surface2:#1a1e2b;--border:#252a38;
      --accent:#5b8df6;--accent2:#f65b8d;--accent3:#5bf6c8;--accent4:#f6c85b;
      --text:#e8eaf2;--muted:#6b7280;
      --mono:'Space Mono',monospace;--sans:'DM Sans',sans-serif;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:var(--sans)!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
header[data-testid="stHeader"]{background:transparent!important;}
#MainMenu,footer{visibility:hidden;}
.ph{background:linear-gradient(135deg,#0d0f14,#111827,#0d1420);border:1px solid var(--border);
    border-left:4px solid var(--accent3);border-radius:12px;padding:28px 36px;margin-bottom:24px;}
.ph h1{font-size:24px;font-weight:600;margin:0 0 4px;color:var(--text);}
.ph p{font-size:12px;color:var(--muted);margin:0;}
.log-box{background:#080a0e;border:1px solid var(--border);border-radius:8px;padding:14px 18px;
    font-family:monospace;font-size:11px;color:var(--accent3);white-space:pre-wrap;
    max-height:400px;overflow-y:auto;line-height:1.8;}
.stButton>button{background:var(--accent3)!important;color:#0d0f14!important;border:none!important;
    border-radius:8px!important;font-family:var(--mono)!important;font-size:12px!important;
    letter-spacing:1px!important;padding:12px 28px!important;font-weight:700!important;width:100%!important;}
[data-testid="stMetric"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:10px!important;padding:14px!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:11px!important;}
[data-testid="stMetricValue"]{color:var(--text)!important;}
[data-testid="stTabs"] button{font-family:var(--mono)!important;font-size:11px!important;color:var(--muted)!important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--accent3)!important;border-bottom-color:var(--accent3)!important;}
.sec{display:flex;align-items:center;gap:10px;margin:20px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--border);}
.sec-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.sec-lbl{font-family:var(--mono);font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--text);}
</style>
""", unsafe_allow_html=True)

RAW_PATH = "./data/raw/train_data.csv"

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-family:monospace;font-size:18px;font-weight:700;color:#5b8df6;padding:8px 0 20px'>🎮 SteamML</div>", unsafe_allow_html=True)
    st.page_link("app.py",                      label="⬡  Dashboard")
    st.page_link("pages/preprocessing_page.py", label="⬡  Preprocessing & NLP")
    st.page_link("pages/model_page.py",         label="⬡  Models")
    st.markdown("---")
    done     = st.session_state.get("done", False)
    nlp_done = st.session_state.get("nlp_done", False)
    st.markdown(f"""<div style='font-size:12px;line-height:2'>
        <span style='color:{"#5bf6c8" if done else "#6b7280"}'>{"●" if done else "○"}</span>&nbsp;
        Preprocessing {"done ✓" if done else "idle"}<br>
        <span style='color:{"#5bf6c8" if nlp_done else "#6b7280"}'>{"●" if nlp_done else "○"}</span>&nbsp;
        NLP {"done ✓" if nlp_done else "idle"}
    </div>""", unsafe_allow_html=True)

st.markdown("""<div class="ph">
    <h1>⚙️  Preprocessing Pipeline</h1>
    <p>Runs the numeric preprocessing and NLP pipelines on train_data.csv — results appear live below.</p>
</div>""", unsafe_allow_html=True)

tab_pre, tab_nlp = st.tabs(["🔧  PREPROCESSING", "🔤  NLP / TF-IDF / LSA"])

# ── Shared helpers ─────────────────────────────────────────────────────
def dark_fig(nrows=1, ncols=1, figsize=(12, 4)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='#13161e')
    axl = np.array(axes).flatten() if hasattr(axes, '__len__') else [axes]
    for ax in axl:
        ax.set_facecolor('#1a1e2b')
        ax.tick_params(colors='#6b7280', labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('#252a38')
    return fig, (np.array(axes).flatten() if hasattr(axes, '__len__') else [axes])

def sec(label, color='#5bf6c8'):
    st.markdown(
        f'<div class="sec"><div class="sec-dot" style="background:{color}"></div>'
        f'<div class="sec-lbl">{label}</div></div>',
        unsafe_allow_html=True
    )

def render_log(lines):
    html = "".join(
        f'<div style=\'{"color:#6b7280" if k=="muted" else "color:#f65b8d" if k=="err" else "color:#5bf6c8"}\'>'
        f'{m.replace("<","&lt;").replace(">","&gt;")}</div>'
        for m, k in lines
    )
    st.markdown(f'<div class="log-box">{html}</div>', unsafe_allow_html=True)

def safe_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms

# ═══════════════════════════════════════════════════════
# TAB 1  ── PREPROCESSING
# ═══════════════════════════════════════════════════════
with tab_pre:
    cl, cr = st.columns([1, 1], gap="large")
    with cl:
        if os.path.exists(RAW_PATH):
            st.success(f"✓ `{RAW_PATH}` ready")
        else:
            st.error(f"Not found: `{RAW_PATH}`")
        run1 = st.button("▶  RUN PREPROCESSING", key="run1", disabled=not os.path.exists(RAW_PATH))
    with cr:
        st.markdown("""<div style='background:#13161e;border:1px solid #252a38;border-radius:10px;
            padding:16px 18px;font-size:12px;color:#6b7280;line-height:1.9'>
            Drop missing/constant → Bool 0/1 → Date features → Price features →
            Flags → SteamSpy log → Target log → Interaction features →
            Split 70/15/15 → Median fill → IQR capping → Isolation Forest →
            StandardScaler → Mutual Information → Save CSVs
        </div>""", unsafe_allow_html=True)

    if run1:
        log = []
        L = lambda m, k="ok": log.append((m, k))
        try:
            L("// Preprocessing started ─────────────────")
            df = pd.read_csv(RAW_PATH)
            L(f"  loaded  {df.shape[0]:,} rows × {df.shape[1]} cols")

            hm = df.isnull().mean()
            df.drop(columns=hm[hm > 0.5].index, inplace=True)
            L(f"  dropped {len(hm[hm>0.5])} high-missing cols", "muted")

            const = [c for c in df.columns if df[c].nunique() == 1]
            df.drop(columns=const, inplace=True)
            L(f"  dropped {len(const)} constant cols", "muted")

            bool_cols = [c for c in df.columns if df[c].dtype == bool]
            df[bool_cols] = df[bool_cols].astype(int)
            bvar = df[bool_cols].var().sort_values()
            low_var = bvar[bvar < 0.001].index
            df.drop(columns=low_var, inplace=True)
            df.drop(columns=['QueryID', 'ResponseID'], inplace=True, errors='ignore')
            L(f"  bool→int; dropped low-var: {list(low_var)}")

            df['ReleaseDate']   = pd.to_datetime(df['ReleaseDate'], errors='coerce')
            df['release_year']  = df['ReleaseDate'].dt.year.fillna(df['ReleaseDate'].dt.year.median())
            df['release_month'] = df['ReleaseDate'].dt.month.fillna(6)
            df['game_age_days'] = (pd.Timestamp.today() - df['ReleaseDate']).dt.days
            df['game_age_days'] = df['game_age_days'].fillna(df['game_age_days'].median())
            df.drop(columns=['ReleaseDate'], inplace=True)
            L("  date → release_year, release_month, game_age_days")

            df['discount_ratio']      = ((df['PriceInitial'] - df['PriceFinal']) / (df['PriceInitial'] + 1e-9)).clip(0, 1)
            df['is_effectively_free'] = ((df['PriceInitial'] == 0) | (df['IsFree'] == 1)).astype(int)
            L(f"  price — {df['is_effectively_free'].sum():,} free games")

            df['has_metacritic']     = (df['Metacritic'] > 0).astype(int)
            df['num_languages']      = df['SupportedLanguages'].fillna('').apply(lambda x: len([w for w in x.split() if len(w) > 2]))
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
            L("  flags engineered")

            for c in ['SteamSpyOwners', 'SteamSpyOwnersVariance', 'SteamSpyPlayersEstimate', 'SteamSpyPlayersVariance']:
                if c in df.columns:
                    df[f'{c}_log'] = np.log1p(df[c])
            df['target_log'] = np.log1p(df['RecommendationCount'])
            L(f"  log transforms done — target mean={df['target_log'].mean():.2f}")

            df['price_per_language']     = df['PriceFinal'] / (df['num_languages'] + 1)
            df['metacritic_x_age']       = df['has_metacritic'] * df['game_age_days']
            df['owners_per_achievement'] = df['SteamSpyOwners_log'] / (df['AchievementCount'] + 1)
            df['dlc_x_owners']           = np.log1p(df['DLCCount']) * df['SteamSpyOwners_log']
            df['movie_x_owners']         = df['MovieCount'] * df['SteamSpyOwners_log']
            L("  interaction features created")

            drop_text = [
                'QueryName', 'ResponseName', 'Website', 'SupportEmail', 'SupportURL',
                'LegalNotice', 'Reviews', 'SupportedLanguages', 'ShortDescrip',
                'DetailedDescrip', 'DRMNotice', 'ExtUserAcctNotice', 'PriceCurrency',
                'Background', 'HeaderImage', 'AboutText', 'PCMinReqsText', 'PCRecReqsText',
                'LinuxMinReqsText', 'LinuxRecReqsText', 'MacMinReqsText',
            ]
            df.drop(columns=[c for c in drop_text if c in df.columns], inplace=True)
            for c in df.select_dtypes(include='object').columns.tolist():
                df.drop(columns=c, inplace=True)
            L(f"  text dropped → {df.shape[1]} numeric cols remain")

            CONT = [c for c in [
                'RequiredAge', 'DemoCount', 'DeveloperCount', 'DLCCount', 'MovieCount', 'PackageCount',
                'PublisherCount', 'ScreenshotCount', 'SteamSpyOwners', 'SteamSpyOwnersVariance',
                'SteamSpyPlayersEstimate', 'SteamSpyPlayersVariance', 'AchievementCount',
                'AchievementHighlightedCount', 'PriceInitial', 'PriceFinal',
                'release_year', 'release_month', 'game_age_days', 'num_languages',
                'about_length', 'short_length', 'detail_length',
                'SteamSpyOwners_log', 'SteamSpyOwnersVariance_log',
                'SteamSpyPlayersEstimate_log', 'SteamSpyPlayersVariance_log',
                'price_per_language', 'metacritic_x_age', 'owners_per_achievement',
                'dlc_x_owners', 'movie_x_owners',
            ] if c in df.columns]

            X = df.drop(columns=['RecommendationCount', 'target_log'])
            y = df['target_log']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
            # X_train, X_val, y_train, y_val = train_test_split(Xt, yt, test_size=0.1765, random_state=42)
            # L(f"  split → train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

            num_cols = X_train.select_dtypes(include=np.number).columns
            med = X_train[num_cols].median()
            for s in [X_train , X_test]:
                s[num_cols] = s[num_cols].fillna(med)
            L(f"  median fill done — NaNs left: {X_train.isnull().sum().sum()}")

            cont_feat_cols = [c for c in CONT if c in X_train.columns]
            NO_IQR = ['RequiredAge', 'DemoCount', 'DeveloperCount', 'DLCCount', 'PackageCount', 'PublisherCount']
            X_train_raw = X_train.copy()
            total_clip = 0
            for col in cont_feat_cols:
                if col in NO_IQR:
                    for s in [X_train, X_test]:
                        s[col] = np.log1p(s[col])
                    continue
                Q1, Q3 = X_train[col].quantile(0.25), X_train[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:
                    continue
                lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                total_clip += ((X_train[col] < lo) | (X_train[col] > hi)).sum()
                for s in [X_train, X_test]:
                    s[col] = s[col].clip(lo, hi)
            L(f"  IQR capping — {total_clip:,} values clipped")

            iso = IsolationForest(contamination=0.05, random_state=42)
            mask = iso.fit_predict(X_train[cont_feat_cols]) == 1
            L(f"  Isolation Forest — kept={mask.sum():,}  removed={(~mask).sum():,}")

            X_train_precap = X_train.copy()
            scaler = StandardScaler()
            X_train[cont_feat_cols] = scaler.fit_transform(X_train[cont_feat_cols])
            # X_val[cont_feat_cols]   = scaler.transform(X_val[cont_feat_cols])
            X_test[cont_feat_cols]  = scaler.transform(X_test[cont_feat_cols])
            L(f"  StandardScaler applied to {len(cont_feat_cols)} cols")

            mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
            mi_df = pd.DataFrame({'feature': X_train.columns, 'MI': mi_scores}).sort_values('MI', ascending=False)
            L(f"  MI top: {mi_df.iloc[0]['feature']} = {mi_df.iloc[0]['MI']:.4f}")

            Xc = X_train_precap[cont_feat_cols].copy()
            Xc['target'] = y_train.values
            corr_s = Xc.corr()['target'].drop('target').abs().sort_values(ascending=False)

            os.makedirs('./data/processed', exist_ok=True)
            os.makedirs('./plots', exist_ok=True)
            tr_out = X_train.copy(); tr_out['target_log'] = y_train.values
            # v_out  = X_val.copy();   v_out['target_log']  = y_val.values
            te_out = X_test.copy();  te_out['target_log'] = y_test.values
            # tr_out.to_csv('./data/processed/train.csv', index=False)
            # # v_out.to_csv('./data/processed/val.csv',    index=False)
            # te_out.to_csv('./data/processed/test.csv',  index=False)

            L("// Done ✓ ─────────────────────────────────")
            st.session_state.update({
                "done": True, "X_train": X_train,  "X_test": X_test,
                "y_train": y_train, "y_test": y_test,
                "X_train_raw": X_train_raw, "cont_cols": cont_feat_cols,
                "corr_s": corr_s, "mi_df": mi_df,
                "iso_kept": mask.sum(), "iso_removed": (~mask).sum(),
                "y_raw": np.expm1(y_train.values), "y_log": y_train.values,
            })
        except Exception as e:
            import traceback
            L(f"ERROR: {e}", "err")
            L(traceback.format_exc(), "err")
        render_log(log)

    if st.session_state.get("done"):
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]
        corr_s  = st.session_state["corr_s"]
        mi_df   = st.session_state["mi_df"]
        cont_feat_cols = st.session_state["cont_cols"]

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train rows",  f"{len(X_train):,}")
        c2.metric("Features",    X_train.shape[1])
        c3.metric("ISO kept",    f"{st.session_state['iso_kept']:,}")
        c4.metric("ISO removed", f"{st.session_state['iso_removed']:,}")

        sec("Target Distribution (log-space)", "#5b8df6")
        fig, axs = dark_fig(1, 2, (13, 4))
        axs[0].hist(st.session_state["y_log"], bins=60, color='#5b8df6', alpha=0.85, edgecolor='none')
        axs[0].set_title("log(RecommendationCount+1)", color='#e8eaf2', fontsize=9, fontfamily='monospace')
        axs[1].hist(st.session_state["y_raw"], bins=60, color='#5bf6c8', alpha=0.85, edgecolor='none')
        axs[1].set_title("Original scale (clipped at 98th pct)", color='#e8eaf2', fontsize=9, fontfamily='monospace')
        axs[1].set_xlim(0, np.percentile(st.session_state["y_raw"], 98))
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        avail = [c for c in cont_feat_cols if c in X_train.columns]
        if avail:
            sec("Feature Explorer", "#f6c85b")
            sf = st.selectbox("Select feature", avail, key="feat_sel")
            vals = X_train[sf].dropna()
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Mean", f"{vals.mean():.3f}")
            e2.metric("Std",  f"{vals.std():.3f}")
            e3.metric("Min",  f"{vals.min():.3f}")
            e4.metric("Max",  f"{vals.max():.3f}")
            fig, axs = dark_fig(1, 2, (13, 4))
            axs[0].hist(vals, bins=60, color='#5b8df6', alpha=0.85, edgecolor='none')
            axs[0].axvline(vals.mean(),   color='#5bf6c8', lw=1.5, linestyle='--', label='mean')
            axs[0].axvline(vals.median(), color='#f6c85b', lw=1.2, linestyle=':',  label='median')
            axs[0].legend(fontsize=8, framealpha=0.4)
            axs[0].set_title(f"Distribution · {sf}", color='#e8eaf2', fontsize=9, fontfamily='monospace')
            axs[1].boxplot(
                vals, vert=False, patch_artist=True,
                medianprops=dict(color='#5bf6c8', lw=2),
                boxprops=dict(facecolor=(0.357, 0.553, 0.965, 0.3), color='#5b8df6'),
                whiskerprops=dict(color='#6b7280'), capprops=dict(color='#6b7280'),
                flierprops=dict(marker='.', color='#6b7280', markersize=3, alpha=0.3),
            )
            axs[1].set_title(f"Box Plot · {sf}", color='#e8eaf2', fontsize=9, fontfamily='monospace')
            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        sec("Feature Correlation with Target", "#5bf6c8")
        top15 = corr_s.head(15)
        fig, axs = dark_fig(1, 1, (12, 4))
        clrs = ['#5bf6c8' if i < 3 else '#5b8df6' if i < 8 else '#6b7280' for i in range(len(top15))]
        axs[0].barh(range(len(top15)), top15.values[::-1], color=clrs[::-1], alpha=0.85, height=0.65)
        axs[0].set_yticks(range(len(top15)))
        axs[0].set_yticklabels(top15.index[::-1], fontsize=8, color='#e8eaf2')
        axs[0].set_xlabel('|Pearson|', fontsize=9, color='#6b7280')
        axs[0].set_title('Top 15 Feature Correlations with log(RecommendationCount)',
                         color='#e8eaf2', fontsize=9, fontfamily='monospace')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        sec("Mutual Information — Top 20", "#7F77DD")
        t20 = mi_df.head(20)
        fig, axs = dark_fig(1, 1, (12, 4))
        axs[0].barh(range(len(t20)), t20['MI'].values[::-1], color='#7F77DD', alpha=0.85, height=0.65)
        axs[0].set_yticks(range(len(t20)))
        axs[0].set_yticklabels(t20['feature'].values[::-1], fontsize=8, color='#e8eaf2')
        axs[0].set_xlabel('MI Score', fontsize=9, color='#6b7280')
        axs[0].set_title('Mutual Information with Target',
                         color='#e8eaf2', fontsize=9, fontfamily='monospace')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        if avail:
            sec("Post-Scaling Stats (train)", "#5b8df6")
            desc = X_train[avail].describe().T[['mean', 'std', 'min', 'max']].round(4)
            desc.index = desc.index.astype(str)
            st.dataframe(desc, use_container_width=True, height=280)


# ═══════════════════════════════════════════════════════
# TAB 2  ── NLP  (per-field TF-IDF + SVD, matching predict_page)
# ═══════════════════════════════════════════════════════
with tab_nlp:
    cl2, cr2 = st.columns([1, 1], gap="large")
    with cl2:
        if os.path.exists(RAW_PATH):
            st.success(f"✓ `{RAW_PATH}` ready")
        else:
            st.error(f"Not found: `{RAW_PATH}`")
        run2 = st.button("▶  RUN NLP PIPELINE", key="run2", disabled=not os.path.exists(RAW_PATH))
    with cr2:
        st.markdown("""<div style='background:#13161e;border:1px solid #252a38;border-radius:10px;
            padding:16px 18px;font-size:12px;color:#6b7280;line-height:1.9'>
            Raw text stats → Sparsity filter (>50% empty dropped) →
            Clean (HTML/URL/stopwords/lemmatize) → TF-IDF <b>per field</b> →
            LSA / TruncatedSVD <b>15 components per field</b> → L2-normalize →
            Save nlp_features CSVs + <code>tfidf_vectorizers.pkl</code> + <code>svd_models.pkl</code>
        </div>""", unsafe_allow_html=True)

    if run2:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        for pkg in ['stopwords', 'wordnet', 'omw-1.4']:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

        log2 = []
        L2 = lambda m, k="ok": log2.append((m, k))
        try:
            L2("// NLP Pipeline started ─────────────────")
            df2 = pd.read_csv(RAW_PATH)

            # ── Field config (matches predict_page NLP_FIELD_MAP + TFIDF_CONFIG) ──
            TC_RAW = {
                'about'          : 'AboutText',
                'short'          : 'ShortDescrip',
                'detail'         : 'DetailedDescrip',
                'reviews'        : 'Reviews',
                'name'           : 'ResponseName',
                'PCMinReqsText'  : 'PCMinReqsText',
                'PCRecReqsText'  : 'PCRecReqsText',
                'LinuxMinReqsText': 'LinuxMinReqsText',
                'MacMinReqsText' : 'MacMinReqsText',
            }
            # keep only columns that exist in the CSV
            TC = {k: v for k, v in TC_RAW.items() if v in df2.columns}
            for col in TC.values():
                df2[col] = df2[col].fillna('')
            L2(f"  text cols found: {list(TC.values())}")

            # ── Step 1: raw stats + sparsity filter ───────────────────────────
            SPARSITY_THRESH = 0.5
            raw_stats = {}
            keys_to_drop = []
            for key, col in TC.items():
                vals = df2[col]
                empty_ratio = (vals.str.len() <= 10).mean()
                raw_stats[key] = {
                    'wc'         : vals.apply(lambda x: len(x.split())),
                    'has_content': (vals.str.len() > 10).sum(),
                    'empty_ratio': empty_ratio,
                }
                L2(f"  [{col}]  non-empty={raw_stats[key]['has_content']:,}  "
                   f"empty={empty_ratio:.1%}  avg_words={raw_stats[key]['wc'].mean():.0f}", "muted")
                if empty_ratio > SPARSITY_THRESH:
                    keys_to_drop.append(key)
                    L2(f"    → dropping (>{SPARSITY_THRESH:.0%} empty)", "muted")
            for key in keys_to_drop:
                TC.pop(key)
            L2(f"  kept fields after sparsity filter: {list(TC.keys())}")

            # ── Step 2: text cleaning ─────────────────────────────────────────
            lemmatizer = WordNetLemmatizer()
            STOP = set(stopwords.words('english'))
            HR = re.compile(r'<[^>]+>')
            UR = re.compile(r'http\S+|www\.\S+')
            PR = re.compile(r'[^a-zA-Z\s]')
            SR = re.compile(r'\s+')

            def clean(t):
                t = HR.sub(' ', t)
                t = UR.sub(' ', t)
                t = t.lower()
                t = PR.sub(' ', t)
                t = SR.sub(' ', t).strip()
                return ' '.join([lemmatizer.lemmatize(w) for w in t.split()
                                 if w not in STOP and len(w) > 2])

            cleaned = {}
            for key, col in TC.items():
                cleaned[key] = df2[col].apply(clean)
                avg = cleaned[key].apply(lambda x: len(x.split())).mean()
                L2(f"  cleaned [{key}] — avg tokens: {avg:.0f}", "muted")

            # ── Step 3: per-field TF-IDF config ──────────────────────────────
            TFIDF_CFG = {
                'about'          : dict(max_features=500, ngram_range=(1,2), min_df=3, max_df=0.95, sublinear_tf=True),
                'detail'         : dict(max_features=500, ngram_range=(1,2), min_df=3, max_df=0.95, sublinear_tf=True),
                'short'          : dict(max_features=300, ngram_range=(1,2), min_df=3, max_df=0.95, sublinear_tf=True),
                'reviews'        : dict(max_features=300, ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True),
                'name'           : dict(max_features=100, ngram_range=(1,1), min_df=2, max_df=0.90, sublinear_tf=True),
                'PCMinReqsText'  : dict(max_features=200, ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True),
                'PCRecReqsText'  : dict(max_features=200, ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True),
                'LinuxMinReqsText': dict(max_features=100, ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True),
                'MacMinReqsText' : dict(max_features=100, ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True),
            }
            # only keep configs for fields that survived sparsity filter
            TFIDF_CFG = {k: v for k, v in TFIDF_CFG.items() if k in cleaned}

            N_COMPONENTS = 15

            # train/val/test split indices (same seed as numeric preprocessing)
            idx_all = np.arange(len(df2))
            idx_t2, idx_te = train_test_split(idx_all, test_size=0.15, random_state=42)
            idx_tr, idx_v  = train_test_split(idx_t2,  test_size=0.1765, random_state=42)

            tfidf_vectorizers = {}
            svd_models        = {}
            tfidf_train_mats  = {}   # for plots
            lsa_train_parts, lsa_val_parts, lsa_test_parts = [], [], []

            for key, cfg in TFIDF_CFG.items():
                texts = cleaned[key].values

                # TF-IDF (fit on train only)
                vec   = TfidfVectorizer(**cfg)
                X_tr  = vec.fit_transform(texts[idx_tr])
                X_v   = vec.transform(texts[idx_v])
                X_te  = vec.transform(texts[idx_te])
                tfidf_vectorizers[key] = vec
                tfidf_train_mats[key]  = X_tr

                # LSA per field
                svd_f  = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
                lsa_tr = safe_normalize(svd_f.fit_transform(X_tr))
                lsa_v  = safe_normalize(svd_f.transform(X_v))
                lsa_te = safe_normalize(svd_f.transform(X_te))
                svd_models[key] = svd_f

                explained = svd_f.explained_variance_ratio_.cumsum()[-1] * 100
                L2(f"  [{key}] vocab={len(vec.vocabulary_):,}  "
                   f"lsa=({X_tr.shape[0]},{N_COMPONENTS})  var={explained:.1f}%", "muted")

                col_names = [f'lsa_{key}_{j}' for j in range(N_COMPONENTS)]
                lsa_train_parts.append(pd.DataFrame(lsa_tr, columns=col_names))
                lsa_val_parts.append(pd.DataFrame(lsa_v,   columns=col_names))
                lsa_test_parts.append(pd.DataFrame(lsa_te,  columns=col_names))

            # concatenate all per-field LSA blocks
            nlp_train = pd.concat(lsa_train_parts, axis=1)
            nlp_val   = pd.concat(lsa_val_parts,   axis=1)
            nlp_test  = pd.concat(lsa_test_parts,  axis=1)
            nlp_train.index = idx_tr
            nlp_val.index   = idx_v
            nlp_test.index  = idx_te

            total_nlp_feats = nlp_train.shape[1]
            L2(f"  total NLP features: {len(TFIDF_CFG)} fields × {N_COMPONENTS} = {total_nlp_feats}")

            # correlation with target
            tgt = np.log1p(df2['RecommendationCount'].iloc[idx_tr].values)
            tgt_s = pd.Series(tgt, index=nlp_train.index)
            corrs_nlp = nlp_train.corrwith(tgt_s).abs().sort_values(ascending=False)
            L2(f"  top NLP corr: {corrs_nlp.index[0]} = {corrs_nlp.iloc[0]:.4f}")

            # ── Save CSVs ─────────────────────────────────────────────────────
            # os.makedirs('./data/processed', exist_ok=True)
            # nlp_train.reset_index(drop=True).to_csv('./data/processed/nlp_features_train.csv', index=False)
            # nlp_val.reset_index(drop=True).to_csv('./data/processed/nlp_features_val.csv',     index=False)
            # nlp_test.reset_index(drop=True).to_csv('./data/processed/nlp_features_test.csv',   index=False)
            # L2("  saved NLP CSVs to ./data/processed/")

            # # ── Save models (used by predict_page) ────────────────────────────
            # os.makedirs('./models', exist_ok=True)
            # joblib.dump(tfidf_vectorizers, './models/tfidf_vectorizers.pkl')
            # joblib.dump(svd_models,        './models/svd_models.pkl')
            # L2("  saved tfidf_vectorizers.pkl + svd_models.pkl to ./models/")

            L2(f"  fields kept  : {list(TFIDF_CFG.keys())}")
            L2(f"  fields dropped (sparse): {keys_to_drop}")
            L2("// NLP Done ✓ ──────────────────────────")

            st.session_state.update({
                "nlp_done"        : True,
                "raw_stats"       : raw_stats,
                "cleaned"         : cleaned,
                "TC"              : TC,
                "TFIDF_CFG"       : TFIDF_CFG,
                "tfidf_vecs"      : tfidf_vectorizers,
                "svd_models"      : svd_models,
                "tfidf_train_mats": tfidf_train_mats,
                "nlp_train"       : nlp_train,
                "nlp_test"        : nlp_test,
                "corrs_nlp"       : corrs_nlp,
                "N_COMPONENTS"    : N_COMPONENTS,
                "keys_dropped"    : keys_to_drop,
            })
        except Exception as e:
            import traceback
            L2(f"ERROR: {e}", "err")
            L2(traceback.format_exc(), "err")
        render_log(log2)

    # ── NLP visualisations ─────────────────────────────────────────────
    if st.session_state.get("nlp_done"):
        rs      = st.session_state["raw_stats"]
        cl_     = st.session_state["cleaned"]
        TC      = st.session_state["TC"]
        CFG     = st.session_state["TFIDF_CFG"]
        vecs    = st.session_state["tfidf_vecs"]
        svd_m   = st.session_state["svd_models"]
        tmats   = st.session_state["tfidf_train_mats"]
        nlp_tr  = st.session_state["nlp_train"]
        nlp_te  = st.session_state["nlp_test"]
        cn      = st.session_state["corrs_nlp"]
        N       = st.session_state["N_COMPONENTS"]
        COLORS  = ['#4C8EDA', '#E8593C', '#1D9E75', '#7F77DD', '#EF9F27',
                   '#F4845F', '#56B4E9', '#CC79A7', '#D55E00']

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fields kept",      len(TC))
        c2.metric("Fields dropped",   len(st.session_state["keys_dropped"]))
        c3.metric("LSA comp/field",   N)
        c4.metric("Total NLP feats",  nlp_tr.shape[1])

        # ── Raw word count distributions ─────────────────────────────────
        sec("Raw Word Count — Per Field", "#5b8df6")
        n_fields = len(TC)
        fig, axs = dark_fig(1, n_fields, (max(5 * n_fields, 8), 4))
        for i, (key, col) in enumerate(TC.items()):
            wc = rs[key]['wc']
            axs[i].hist(wc.clip(0, wc.quantile(0.98)), bins=60,
                        color=COLORS[i % len(COLORS)], alpha=0.85, edgecolor='none')
            axs[i].axvline(wc.mean(), color='#e8eaf2', lw=1.2, linestyle='--',
                           label=f'mean={wc.mean():.0f}')
            axs[i].set_title(col, color='#e8eaf2', fontsize=8, fontfamily='monospace')
            axs[i].legend(fontsize=7, framealpha=0.4)
        plt.suptitle("Raw Text Word Counts", color='#e8eaf2', fontsize=10, fontfamily='monospace', y=1.02)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── Before vs after cleaning ──────────────────────────────────────
        sec("Text Cleaning — Before vs After", "#5bf6c8")
        sel_t = st.selectbox("Field", list(TC.keys()), key="txt_sel", format_func=lambda k: TC[k])
        fig, axs = dark_fig(1, 2, (13, 4))
        bwc = rs[sel_t]['wc']
        awc = cl_[sel_t].apply(lambda x: len(x.split()))
        cap = bwc.quantile(0.98)
        axs[0].hist(bwc.clip(0, cap), bins=60, color='#4C8EDA', alpha=0.85, edgecolor='none')
        axs[0].axvline(bwc.mean(), color='#5bf6c8', lw=1.5, linestyle='--', label=f'mean={bwc.mean():.0f}')
        axs[0].set_title(f"{TC[sel_t]} — Raw", color='#e8eaf2', fontsize=9, fontfamily='monospace')
        axs[0].legend(fontsize=8, framealpha=0.4)
        axs[1].hist(awc.clip(0, cap * 0.6), bins=60, color='#E8593C', alpha=0.85, edgecolor='none')
        axs[1].axvline(awc.mean(), color='#5bf6c8', lw=1.5, linestyle='--', label=f'mean={awc.mean():.0f}')
        axs[1].set_title(f"{TC[sel_t]} — Cleaned", color='#e8eaf2', fontsize=9, fontfamily='monospace')
        axs[1].legend(fontsize=8, framealpha=0.4)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── TF-IDF top terms per field ────────────────────────────────────
        sec("TF-IDF — Top Terms", "#f6c85b")
        keys_list = list(CFG.keys())
        sel_tf = st.selectbox("Field", keys_list, key="tfidf_sel", format_func=lambda k: TC[k])
        kidx   = keys_list.index(sel_tf)
        mat    = tmats[sel_tf]
        ms     = np.asarray(mat.mean(axis=0)).flatten()
        vocab  = vecs[sel_tf].get_feature_names_out()
        top40  = ms.argsort()[-40:][::-1]
        fig, axs = dark_fig(1, 1, (12, 6))
        axs[0].barh(range(40), ms[top40][::-1], color=COLORS[kidx % len(COLORS)], alpha=0.85, height=0.75)
        axs[0].set_yticks(range(40))
        axs[0].set_yticklabels(vocab[top40][::-1], fontsize=7, color='#e8eaf2')
        axs[0].set_title(f'Top 40 TF-IDF Terms — {TC[sel_tf]}',
                         color='#e8eaf2', fontsize=9, fontfamily='monospace')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── LSA explained variance per field ─────────────────────────────
        sec("LSA — Explained Variance per Field", "#7F77DD")
        field_vars = [(k, svd_m[k].explained_variance_ratio_.cumsum()[-1] * 100) for k in CFG]
        field_vars.sort(key=lambda x: x[1], reverse=True)
        fig, axs = dark_fig(1, 1, (12, 4))
        fkeys = [x[0] for x in field_vars]
        fvals = [x[1] for x in field_vars]
        bars  = axs[0].barh(range(len(fkeys)), fvals,
                             color=[COLORS[i % len(COLORS)] for i in range(len(fkeys))],
                             alpha=0.85, height=0.6)
        axs[0].set_yticks(range(len(fkeys)))
        axs[0].set_yticklabels(fkeys, fontsize=9, color='#e8eaf2')
        axs[0].axvline(80, color='#888780', lw=1, linestyle='--', label='80%')
        axs[0].legend(fontsize=8, framealpha=0.4)
        axs[0].set_xlabel(f'Cumulative Variance (%) — {N} components', fontsize=8, color='#6b7280')
        axs[0].set_title('LSA Variance Explained per Field',
                         color='#e8eaf2', fontsize=9, fontfamily='monospace')
        for bar, val in zip(bars, fvals):
            axs[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}%', va='center', fontsize=8, color='#e8eaf2')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── LSA top words per component ───────────────────────────────────
        sec("LSA — Top Words per Component (per field)", "#E8593C")
        sel_field_lsa = st.selectbox("Field", keys_list, key="lsa_field_sel",
                                     format_func=lambda k: TC[k])
        comp_i = st.slider("Component", 1, N, 1, key="comp_sl") - 1
        svd_f  = svd_m[sel_field_lsa]
        vocab_f = vecs[sel_field_lsa].get_feature_names_out()
        loading = svd_f.components_[comp_i]
        top_idx = np.concatenate([loading.argsort()[-15:][::-1], loading.argsort()[:5]])
        tw, ts  = vocab_f[top_idx], loading[top_idx]
        cb      = ['#E8593C' if s > 0 else '#4C8EDA' for s in ts]
        fig, axs = dark_fig(1, 1, (12, 5))
        axs[0].barh(range(len(top_idx)), ts[::-1], color=cb[::-1], alpha=0.85, height=0.75)
        axs[0].set_yticks(range(len(top_idx)))
        axs[0].set_yticklabels(tw[::-1], fontsize=8, color='#e8eaf2')
        axs[0].axvline(0, color='#6b7280', lw=0.8)
        axs[0].set_title(f'{TC[sel_field_lsa]} — LSA Component {comp_i + 1}  '
                         f'(red=positive, blue=negative)',
                         color='#e8eaf2', fontsize=9, fontfamily='monospace')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── NLP–target correlation ────────────────────────────────────────
        sec("NLP Features — Correlation with Target", "#5bf6c8")
        t20n = cn.head(20)
        fig, axs = dark_fig(1, 1, (12, 4))
        # colour by field
        bar_colors = []
        for feat in t20n.index[::-1]:
            fld = feat.replace('lsa_', '').rsplit('_', 1)[0]
            fi  = keys_list.index(fld) if fld in keys_list else 0
            bar_colors.append(COLORS[fi % len(COLORS)])
        axs[0].barh(range(len(t20n)), t20n.values[::-1], color=bar_colors, alpha=0.85, height=0.65)
        axs[0].set_yticks(range(len(t20n)))
        axs[0].set_yticklabels(t20n.index[::-1], fontsize=8, color='#e8eaf2')
        axs[0].set_xlabel('|Pearson|', fontsize=9, color='#6b7280')
        axs[0].set_title('Top 20 NLP Features — Correlation with log(RecommendationCount)  '
                         '(colour = field)',
                         color='#e8eaf2', fontsize=9, fontfamily='monospace')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── LSA train vs test distributions ──────────────────────────────
        sec("LSA Feature Distributions — Train vs Test", "#5b8df6")
        sample_cols = []
        for key in CFG:
            sample_cols += [f'lsa_{key}_0', f'lsa_{key}_1']
        sample_cols = [c for c in sample_cols if c in nlp_tr.columns][:10]
        if sample_cols:
            n_sc = len(sample_cols)
            ncols_sc = min(5, n_sc)
            nrows_sc = int(np.ceil(n_sc / ncols_sc))
            fig, axs = dark_fig(nrows_sc, ncols_sc, (ncols_sc * 3.5, nrows_sc * 3.5))
            for i, col in enumerate(sample_cols):
                axs[i].hist(nlp_tr[col], bins=50, color='#4C8EDA', alpha=0.65,
                            label='Train', edgecolor='none')
                axs[i].hist(nlp_te[col], bins=50, color='#E8593C', alpha=0.65,
                            label='Test',  edgecolor='none')
                axs[i].set_title(col, fontsize=7.5, color='#e8eaf2', fontfamily='monospace')
                axs[i].tick_params(labelsize=6.5)
                if i == 0:
                    axs[i].legend(fontsize=7, framealpha=0.4)
            for j in range(n_sc, len(axs)):
                axs[j].set_visible(False)
            plt.suptitle("LSA Distributions — Train vs Test (first 2 components per field)",
                         color='#e8eaf2', fontsize=9, fontfamily='monospace', y=1.02)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()