import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="Steam Game Recommender · ML Pipeline",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --bg: #0d0f14; --surface: #13161e; --surface2: #1a1e2b;
    --border: #252a38; --accent: #5b8df6; --accent2: #f65b8d;
    --accent3: #5bf6c8; --text: #e8eaf2; --muted: #6b7280;
    --mono: 'Space Mono', monospace; --sans: 'DM Sans', sans-serif;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important; color: var(--text) !important;
    font-family: var(--sans) !important;
}
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }

.hero { background: linear-gradient(135deg, #0d0f14 0%, #131a2e 50%, #0d1420 100%);
    border: 1px solid var(--border); border-radius: 16px; padding: 48px 52px; margin-bottom: 32px; }
.hero-eyebrow { font-family: var(--mono); font-size: 11px; letter-spacing: 3px; color: var(--accent); text-transform: uppercase; margin-bottom: 14px; }
.hero-title { font-size: 42px; font-weight: 600; line-height: 1.15; color: var(--text); margin-bottom: 16px; }
.hero-title span { color: var(--accent); }
.hero-sub { font-size: 15px; color: var(--muted); max-width: 560px; line-height: 1.7; font-weight: 300; }
.hero-badge { display: inline-block; background: rgba(91,141,246,0.12); border: 1px solid rgba(91,141,246,0.3);
    color: var(--accent); font-family: var(--mono); font-size: 11px; padding: 4px 12px; border-radius: 20px; margin-top: 20px; margin-right: 8px; }
.hero-badge.green { background: rgba(91,246,200,0.08); border-color: rgba(91,246,200,0.25); color: var(--accent3); }
.hero-badge.pink  { background: rgba(246,91,141,0.08); border-color: rgba(246,91,141,0.25); color: var(--accent2); }

.stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin-bottom: 32px; }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 22px 24px; position: relative; overflow: hidden; }
.stat-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }
.stat-card.blue::before  { background: var(--accent); }
.stat-card.green::before { background: var(--accent3); }
.stat-card.pink::before  { background: var(--accent2); }
.stat-card.yellow::before{ background: #f6c85b; }
.stat-label { font-family: var(--mono); font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; }
.stat-value { font-size: 32px; font-weight: 600; color: var(--text); line-height: 1; }
.stat-sub { font-size: 12px; color: var(--muted); margin-top: 6px; font-weight: 300; }

.section-header { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid var(--border); }
.section-title { font-family: var(--mono); font-size: 12px; letter-spacing: 2px; text-transform: uppercase; color: var(--text); }
.section-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); flex-shrink: 0; }
.section-dot.green { background: var(--accent3); }
.section-dot.pink  { background: var(--accent2); }

.info-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 24px; height: 100%; }
.info-card h4 { font-family: var(--mono); font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: var(--accent); margin-bottom: 16px; }
.info-card p { font-size: 13px; color: var(--muted); line-height: 1.7; margin: 0; }
.pill-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.pill { background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; padding: 5px 12px; font-family: var(--mono); font-size: 10px; color: var(--muted); }
.pill.active { border-color: var(--accent); color: var(--accent); background: rgba(91,141,246,0.08); }
.pill.green  { border-color: var(--accent3); color: var(--accent3); background: rgba(91,246,200,0.06); }
.pill.pink   { border-color: var(--accent2); color: var(--accent2); background: rgba(246,91,141,0.06); }

.stButton > button { background: var(--accent) !important; color: #fff !important; border: none !important;
    border-radius: 8px !important; font-family: var(--mono) !important; font-size: 12px !important;
    letter-spacing: 1px !important; padding: 10px 24px !important; }
[data-testid="stMetric"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; padding: 16px !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 10px !important; overflow: hidden !important; }
[data-testid="stTabs"] button { font-family: var(--mono) !important; font-size: 11px !important; letter-spacing: 1px !important; color: var(--muted) !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }
.sidebar-label { font-family: var(--mono); font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); padding: 6px 0; }
.status-row { display: flex; align-items: center; gap: 8px; margin: 8px 0; }
.dot-live { width:8px;height:8px;border-radius:50%;background:var(--accent3);box-shadow:0 0 6px var(--accent3); }
.dot-idle { width:8px;height:8px;border-radius:50%;background:var(--muted); }
.status-text { font-size: 12px; color: var(--muted); }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
PROCESSED_PATH = "./data/processed/train.csv"
NLP_PATH       = "./data/processed/nlp_features_train.csv"
RAW_PATH       = "./data/raw/train_data.csv"

@st.cache_data
def load_data():
    if os.path.exists(PROCESSED_PATH):
        df = pd.read_csv(PROCESSED_PATH)
        if os.path.exists(NLP_PATH):
            nlp = pd.read_csv(NLP_PATH)
            shared = [c for c in nlp.columns if c not in df.columns]
            if shared or len(nlp) == len(df):
                try:
                    df = pd.concat([df.reset_index(drop=True), nlp[shared].reset_index(drop=True)], axis=1)
                except Exception:
                    pass
        return df, "processed"
    elif os.path.exists(RAW_PATH):
        return pd.read_csv(RAW_PATH), "raw"
    return None, None

df, source = load_data()
if df is not None:
    st.session_state["df"] = df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px 0;'>
        <div style='font-family:monospace; font-size:18px; font-weight:700; color:#5b8df6;'>🎮 SteamML</div>
        <div style='font-size:11px; color:#6b7280; margin-top:4px; font-family:monospace;'>Recommendation Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Navigation</div>', unsafe_allow_html=True)

    st.page_link("app.py",                       label="⬡  Dashboard")
    st.page_link("pages/preprocessing_page.py", label="⬡  Preprocessing")
    st.page_link("pages/model_page.py",         label="⬡  Models")
    st.page_link("pages/Model_page.py", label="⚙️ Train Pipeline (Upload)")
    st.page_link("pages/predict_page.py", label="🎯 Inference (Test Upload)")
    st.markdown("---")
    st.markdown('<div class="sidebar-label">Data Source</div>', unsafe_allow_html=True)
    if source == "processed":
        st.markdown('<div class="status-row"><div class="dot-live"></div><div class="status-text">Loaded: data/processed/train.csv</div></div>', unsafe_allow_html=True)
        if os.path.exists(NLP_PATH):
            st.markdown('<div class="status-row"><div class="dot-live"></div><div class="status-text">NLP features merged</div></div>', unsafe_allow_html=True)
    elif source == "raw":
        st.markdown('<div class="status-row"><div class="dot-live"></div><div class="status-text">Loaded: data/raw/train_data.csv</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-row"><div class="dot-idle"></div><div class="status-text">No data found</div></div>', unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">// Steam Game Analytics · ML Pipeline</div>
    <div class="hero-title">Predict Game<br><span>Recommendations</span></div>
    <div class="hero-sub">End-to-end machine learning pipeline for predicting Steam game recommendation counts.</div>
    <div>
        <span class="hero-badge">v1.0.0</span>
        <span class="hero-badge green">sklearn</span>
        <span class="hero-badge pink">~40k games</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Stat Cards ────────────────────────────────────────────────────────────────
rows_val = f"{df.shape[0]:,}" if df is not None else "—"
cols_val = str(df.shape[1])  if df is not None else "—"

st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card blue">
        <div class="stat-label">Total Rows</div>
        <div class="stat-value">{rows_val}</div>
        <div class="stat-sub">training samples</div>
    </div>
    <div class="stat-card green">
        <div class="stat-label">Features</div>
        <div class="stat-value">{cols_val}</div>
        <div class="stat-sub">after preprocessing</div>
    </div>
    <div class="stat-card pink">
        <div class="stat-label">Target</div>
        <div class="stat-value">log(RC)</div>
        <div class="stat-sub">RecommendationCount</div>
    </div>
    <div class="stat-card yellow">
        <div class="stat-label">Source</div>
        <div class="stat-value">{source or "—"}</div>
        <div class="stat-sub">data/{'processed' if source=='processed' else 'raw'}/</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊  OVERVIEW", "📈  EXPLORE", "🗂  FEATURE MAP"])

# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    if df is None:
        st.error("No data found. Make sure data/processed/train.csv or data/raw/train_data.csv exists.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows",    f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing", f"{df.isnull().mean().mean()*100:.1f}%")

        st.markdown("##### Preview (first 5 rows)")
        st.dataframe(df.head(), use_container_width=True, height=200)

        st.markdown("##### Column Types")
        type_df = df.dtypes.value_counts().reset_index()
        type_df.columns = ["dtype", "count"]
        st.dataframe(type_df, use_container_width=True, height=150)

# ── Tab 2: Explore ────────────────────────────────────────────────────────────
with tab2:
    if df is None:
        st.warning("No data loaded.")
    else:
        import matplotlib.pyplot as plt

        st.markdown("""
        <div class="section-header">
            <div class="section-dot pink"></div>
            <div class="section-title">Distribution Explorer</div>
        </div>
        """, unsafe_allow_html=True)

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        col_sel  = st.selectbox("Select column", num_cols, key="explore_col")

        if col_sel:
            col_data = df[col_sel].dropna()
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Mean",   f"{col_data.mean():.2f}")
            e2.metric("Median", f"{col_data.median():.2f}")
            e3.metric("Std",    f"{col_data.std():.2f}")
            e4.metric("Nulls",  f"{df[col_sel].isnull().sum():,}")

            fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), facecolor='#13161e')
            for ax in axes:
                ax.set_facecolor('#1a1e2b')
                ax.tick_params(colors='#6b7280', labelsize=8)
                for sp in ax.spines.values(): sp.set_color('#252a38')

            axes[0].hist(col_data.clip(col_data.quantile(0.01), col_data.quantile(0.99)),
                         bins=60, color='#5b8df6', alpha=0.85, edgecolor='none')
            axes[0].set_title(f"Distribution · {col_sel}", color='#e8eaf2', fontsize=10, fontfamily='monospace')

            # ── FIX: use tuple for RGBA, not CSS string ──
            axes[1].boxplot(col_data, vert=False, patch_artist=True,
                            medianprops=dict(color='#5bf6c8', linewidth=2),
                            boxprops=dict(facecolor=(0.357, 0.553, 0.965, 0.3), color='#5b8df6'),
                            whiskerprops=dict(color='#6b7280'),
                            capprops=dict(color='#6b7280'),
                            flierprops=dict(marker='.', color='#6b7280', markersize=3, alpha=0.4))
            axes[1].set_title(f"Box Plot · {col_sel}", color='#e8eaf2', fontsize=10, fontfamily='monospace')

            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown("---")
        st.markdown("##### Missing Values")
        miss = df.isnull().mean().sort_values(ascending=False).head(20)
        miss = miss[miss > 0]
        if len(miss):
            fig2, ax2 = plt.subplots(figsize=(12, 3), facecolor='#13161e')
            ax2.set_facecolor('#1a1e2b')
            ax2.barh(miss.index, miss.values * 100, color='#f65b8d', alpha=0.85, height=0.6)
            ax2.tick_params(colors='#6b7280', labelsize=8)
            ax2.set_xlabel("% Missing", color='#6b7280', fontsize=9)
            ax2.set_title("Top Missing Columns", color='#e8eaf2', fontsize=10, fontfamily='monospace')
            for sp in ax2.spines.values(): sp.set_color('#252a38')
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()
        else:
            st.success("No missing values found!")

# ── Tab 3: Feature Map ────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="section-header">
        <div class="section-dot"></div>
        <div class="section-title">Feature Engineering Map</div>
    </div>
    """, unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3, gap="large")
    with fc1:
        st.markdown("""
        <div class="info-card">
            <h4>🔵  Continuous · Scaled</h4>
            <p>IQR-capped then StandardScaler applied.</p>
            <div class="pill-grid">
                <span class="pill active">SteamSpyOwners_log</span>
                <span class="pill active">AchievementCount</span>
                <span class="pill active">PriceInitial</span>
                <span class="pill active">DLCCount</span>
                <span class="pill active">game_age_days</span>
                <span class="pill active">about_length</span>
                <span class="pill active">release_year</span>
                <span class="pill active">num_languages</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with fc2:
        st.markdown("""
        <div class="info-card">
            <h4>🟢  Binary Flags</h4>
            <p>0/1 indicators derived from raw fields.</p>
            <div class="pill-grid">
                <span class="pill green">is_effectively_free</span>
                <span class="pill green">has_metacritic</span>
                <span class="pill green">has_website</span>
                <span class="pill green">has_drm</span>
                <span class="pill green">GenreIsIndie</span>
                <span class="pill green">GenreIsAction</span>
                <span class="pill green">CategoryCoop</span>
                <span class="pill green">PlatformLinux</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with fc3:
        st.markdown("""
        <div class="info-card">
            <h4>🔴  Dropped · Text / ID</h4>
            <p>Text columns used for flag derivation, then dropped.</p>
            <div class="pill-grid">
                <span class="pill pink">AboutText</span>
                <span class="pill pink">QueryName</span>
                <span class="pill pink">ShortDescrip</span>
                <span class="pill pink">LegalNotice</span>
                <span class="pill pink">Website</span>
                <span class="pill pink">Reviews</span>
                <span class="pill pink">HeaderImage</span>
                <span class="pill pink">SupportedLanguages</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style='margin-top:48px; padding:20px 0; border-top:1px solid #252a38;
     display:flex; justify-content:space-between; align-items:center;'>
    <span style='font-family:monospace; font-size:11px; color:#6b7280;'>SteamML Pipeline · built with Streamlit</span>
    <span style='font-family:monospace; font-size:11px; color:#5b8df6;'>navigate → Preprocessing ↗</span>
</div>
""", unsafe_allow_html=True)