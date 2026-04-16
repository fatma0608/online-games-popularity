# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os, warnings
# warnings.filterwarnings('ignore')
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# st.set_page_config(
#     page_title="Preprocessing · SteamML",
#     page_icon="⚙️",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
# :root {
#     --bg: #0d0f14; --surface: #13161e; --surface2: #1a1e2b;
#     --border: #252a38; --accent: #5b8df6; --accent2: #f65b8d;
#     --accent3: #5bf6c8; --accent4: #f6c85b; --text: #e8eaf2; --muted: #6b7280;
#     --mono: 'Space Mono', monospace; --sans: 'DM Sans', sans-serif;
# }
# html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; color: var(--text) !important; font-family: var(--sans) !important; }
# [data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
# [data-testid="stSidebar"] * { color: var(--text) !important; }
# header[data-testid="stHeader"] { background: transparent !important; }
# #MainMenu, footer { visibility: hidden; }

# .page-header { background: linear-gradient(135deg, #0d0f14 0%, #111827 60%, #0d1420 100%);
#     border: 1px solid var(--border); border-left: 4px solid var(--accent3);
#     border-radius: 12px; padding: 32px 40px; margin-bottom: 28px; }
# .page-header h1 { font-size: 28px; font-weight: 600; margin: 0 0 6px 0; color: var(--text); }
# .page-header p { font-size: 13px; color: var(--muted); margin: 0; font-weight: 300; }

# .step-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 12px; }
# .step-num { font-family: var(--mono); font-size: 10px; color: var(--accent); letter-spacing: 2px; margin-bottom: 4px; }
# .step-title { font-size: 14px; font-weight: 600; color: var(--text); margin-bottom: 4px; }
# .step-desc { font-size: 12px; color: var(--muted); line-height: 1.6; }

# .section-header { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
# .section-title { font-family: var(--mono); font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: var(--text); }
# .section-dot { width:8px;height:8px;border-radius:50%;background:var(--accent3);flex-shrink:0; }
# .section-dot.blue { background: var(--accent); }
# .section-dot.pink { background: var(--accent2); }

# .log-box { background: #0a0c10; border: 1px solid var(--border); border-radius: 8px; padding: 16px 20px;
#     font-family: monospace; font-size: 12px; color: var(--accent3); white-space: pre-wrap;
#     max-height: 340px; overflow-y: auto; line-height: 1.7; }

# .stButton > button { background: var(--accent3) !important; color: #0d0f14 !important; border: none !important;
#     border-radius: 8px !important; font-family: var(--mono) !important; font-size: 12px !important;
#     letter-spacing: 1px !important; padding: 12px 32px !important; font-weight: 700 !important; width: 100% !important; }
# [data-testid="stMetric"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; padding: 14px !important; }
# [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size:11px !important; }
# [data-testid="stMetricValue"] { color: var(--text) !important; }
# [data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 10px !important; overflow: hidden !important; }
# [data-testid="stTabs"] button { font-family: var(--mono) !important; font-size: 11px !important; color: var(--muted) !important; }
# [data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent3) !important; border-bottom-color: var(--accent3) !important; }
# .sidebar-label { font-family: var(--mono); font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); padding: 6px 0; }
# .status-row { display:flex; align-items:center; gap:8px; margin:8px 0; }
# .dot-live { width:8px;height:8px;border-radius:50%;background:var(--accent3);box-shadow:0 0 6px var(--accent3); }
# .dot-done { width:8px;height:8px;border-radius:50%;background:var(--accent); }
# .dot-idle { width:8px;height:8px;border-radius:50%;background:var(--muted); }
# .status-text { font-size:12px; color:var(--muted); }
# </style>
# """, unsafe_allow_html=True)

# # ── Paths ─────────────────────────────────────────────────────────────────────
# RAW_PATH  = "./data/raw/train_data.csv"
# PROC_PATH = "./data/processed/train.csv"
# NLP_PATH  = "./data/processed/nlp_features_train.csv"

# # ── Sidebar ───────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("""
#     <div style='padding: 8px 0 24px 0;'>
#         <div style='font-family:monospace; font-size:18px; font-weight:700; color:#5b8df6;'>🎮 SteamML</div>
#         <div style='font-size:11px; color:#6b7280; margin-top:4px; font-family:monospace;'>Recommendation Engine</div>
#     </div>
#     """, unsafe_allow_html=True)
#     st.page_link("app.py",                      label="⬡  Dashboard")
#     st.page_link("pages/preprocessing_page.py", label="⬡  Preprocessing")

#     st.markdown("---")
#     pipeline_done = st.session_state.get("pipeline_done", False)
#     st.markdown(f"""
#     <div class="status-row"><div class="dot-live"></div><div class="status-text">Raw data · ready</div></div>
#     <div class="status-row"><div class="{'dot-done' if pipeline_done else 'dot-idle'}"></div>
#         <div class="status-text">Preprocessing · {'done' if pipeline_done else 'idle'}</div></div>
#     """, unsafe_allow_html=True)

#     st.markdown("---")
#     st.markdown('<div class="sidebar-label">Settings</div>', unsafe_allow_html=True)
#     test_size  = st.slider("Test size",  0.10, 0.30, 0.15, 0.05)
#     val_frac   = st.slider("Val fraction", 0.10, 0.30, 0.1765, 0.01)
#     iqr_factor = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)
#     seed       = st.number_input("Random seed", value=42, step=1)
#     do_scale   = st.checkbox("Apply StandardScaler", value=True)

# # ── Page header ───────────────────────────────────────────────────────────────
# st.markdown("""
# <div class="page-header">
#     <h1>⚙️  Preprocessing Pipeline</h1>
#     <p>Runs on data/raw/train_data.csv — cleans, engineers features, splits and scales.</p>
# </div>
# """, unsafe_allow_html=True)

# tab_run, tab_results = st.tabs(["🚀  RUN PIPELINE", "📊  RESULTS & PLOTS"])

# # ═══════════════════════════════════════════════════════════════
# # TAB 1 — RUN
# # ═══════════════════════════════════════════════════════════════
# with tab_run:
#     col_steps, col_run = st.columns([2, 1], gap="large")

#     with col_steps:
#         st.markdown("""<div class="section-header"><div class="section-dot blue"></div>
#         <div class="section-title">Pipeline Steps</div></div>""", unsafe_allow_html=True)

#         for num, title, desc in [
#             ("01", "Load",               f"Read {RAW_PATH}"),
#             ("02", "Drop High-Missing",  "Drop columns with >50% nulls"),
#             ("03", "Drop Constant Cols", "Remove zero-variance columns"),
#             ("04", "Drop Low-Var Bools", "Remove bool cols with variance < 0.001"),
#             ("05", "Date Engineering",   "Parse ReleaseDate → release_year, release_month, game_age_days"),
#             ("06", "Price Features",     "discount_ratio, is_effectively_free"),
#             ("07", "Text / Flag Features","has_metacritic, num_languages, about_length, has_pc_min_reqs, has_drm, etc."),
#             ("08", "Drop Text Columns",  "Remove all remaining string/object columns"),
#             ("09", "Split",              f"Train / Val / Test  ({int((1-test_size)*(1-val_frac)*100)}% / {int((1-test_size)*val_frac*100)}% / {int(test_size*100)}%)"),
#             ("10", "Median Fill",        "Fill NaNs using train-split medians"),
#             ("11", "IQR Capping",        f"Fit on train (×{iqr_factor}), clip all splits"),
#             ("12", "Standard Scale",     "Fit on train, transform all splits" if do_scale else "Skipped"),
#         ]:
#             st.markdown(f"""<div class="step-card">
#                 <div class="step-num">STEP {num}</div>
#                 <div class="step-title">{title}</div>
#                 <div class="step-desc">{desc}</div>
#             </div>""", unsafe_allow_html=True)

#     with col_run:
#         st.markdown("""<div class="section-header"><div class="section-dot"></div>
#         <div class="section-title">Run</div></div>""", unsafe_allow_html=True)

#         if not os.path.exists(RAW_PATH):
#             st.error(f"Raw data not found at `{RAW_PATH}`")
#         else:
#             st.success(f"✓ Found `{RAW_PATH}`")
#             run_btn = st.button("▶  RUN PREPROCESSING")

#             if run_btn:
#                 log_lines = []
#                 def log(msg, kind=""):
#                     log_lines.append((msg, kind))

#                 log("// Pipeline started")
#                 try:
#                     # ── STEP 1: Load ──────────────────────────────────────
#                     df = pd.read_csv(RAW_PATH)
#                     log(f"  loaded {df.shape[0]:,} rows × {df.shape[1]} cols")

#                     # ── STEP 2: Drop high-missing cols (>50%) ─────────────
#                     high_missing = df.isnull().mean()
#                     drop_cols = high_missing[high_missing > 0.5].index
#                     df.drop(columns=drop_cols, inplace=True)
#                     log(f"  dropped {len(drop_cols)} high-missing cols (>50%)", "muted")

#                     # ── STEP 3: Drop constant columns ─────────────────────
#                     constant_cols = [col for col in df.columns if df[col].nunique() == 1]
#                     df.drop(columns=constant_cols, inplace=True)
#                     log(f"  dropped {len(constant_cols)} constant cols: {constant_cols}", "muted")

#                     # ── STEP 4: Drop low-variance bool cols ───────────────
#                     bool_cols = [c for c in df.columns if df[c].dtype == bool]
#                     df[bool_cols] = df[bool_cols].astype(int)
#                     bool_variance = df[bool_cols].var().sort_values()
#                     low_var_bool = bool_variance[bool_variance < 0.001].index
#                     df.drop(columns=low_var_bool, inplace=True)
#                     log(f"  dropped {len(low_var_bool)} low-variance bool cols", "muted")

#                     # Drop ID cols
#                     df.drop(columns=['QueryID', 'ResponseID'], inplace=True, errors='ignore')
#                     log("  dropped QueryID, ResponseID", "muted")

#                     # ── STEP 5: Date engineering ──────────────────────────
#                     df['ReleaseDate'] = pd.to_datetime(df['ReleaseDate'], errors='coerce')
#                     df['release_year']  = df['ReleaseDate'].dt.year
#                     df['release_month'] = df['ReleaseDate'].dt.month
#                     df['game_age_days'] = (pd.Timestamp.today() - df['ReleaseDate']).dt.days
#                     df['release_year']  = df['release_year'].fillna(df['release_year'].median())
#                     df['release_month'] = df['release_month'].fillna(6)
#                     df['game_age_days'] = df['game_age_days'].fillna(df['game_age_days'].median())
#                     df.drop(columns=['ReleaseDate'], inplace=True)
#                     log("  date features engineered (release_year, release_month, game_age_days)")

#                     # ── STEP 6: Price features ────────────────────────────
#                     if 'PriceInitial' in df.columns and 'PriceFinal' in df.columns:
#                         df['discount_ratio'] = (
#                             (df['PriceInitial'] - df['PriceFinal']) /
#                             (df['PriceInitial'] + 1e-9)
#                         ).clip(0, 1)
#                     if 'PriceInitial' in df.columns and 'IsFree' in df.columns:
#                         df['is_effectively_free'] = (
#                             (df['PriceInitial'] == 0) | (df['IsFree'] == 1)
#                         ).astype(int)

#                     # ── STEP 7: Text / flag features ──────────────────────
#                     if 'Metacritic' in df.columns:
#                         df['has_metacritic'] = (df['Metacritic'] > 0).astype(int)

#                     # num_languages: count tokens with len > 2 (matches script exactly)
#                     if 'SupportedLanguages' in df.columns:
#                         df['num_languages'] = df['SupportedLanguages'].fillna('').apply(
#                             lambda x: len([w for w in x.split(' ') if len(w) > 2])
#                         )

#                     # Presence / notna flags
#                     if 'Website' in df.columns:
#                         df['has_website'] = df['Website'].notna().astype(int)
#                     if 'SupportEmail' in df.columns:
#                         df['has_support_email'] = df['SupportEmail'].notna().astype(int)
#                     if 'SupportURL' in df.columns:
#                         df['has_support_url'] = df['SupportURL'].notna().astype(int)

#                     # Length-based flags (len > 1)
#                     for flag, src in [
#                         ('has_legal_notice',  'LegalNotice'),
#                         ('has_drm',           'DRMNotice'),
#                         ('has_ext_account',   'ExtUserAcctNotice'),
#                     ]:
#                         if src in df.columns:
#                             df[flag] = df[src].fillna('').apply(lambda x: 1 if len(x.strip()) > 1 else 0)

#                     # Length-based flag (len > 5)
#                     if 'Reviews' in df.columns:
#                         df['has_reviews_text'] = df['Reviews'].fillna('').apply(
#                             lambda x: 1 if len(x.strip()) > 5 else 0
#                         )

#                     # Text length features
#                     for txt_col, new_col in [
#                         ('AboutText',       'about_length'),
#                         ('ShortDescrip',    'short_length'),
#                         ('DetailedDescrip', 'detail_length'),
#                     ]:
#                         if txt_col in df.columns:
#                             df[new_col] = df[txt_col].fillna('').apply(len)

#                     # PC / Linux / Mac requirements flags (len > 5)
#                     for flag, src in [
#                         ('has_pc_min_reqs',    'PCMinReqsText'),
#                         ('has_pc_rec_reqs',    'PCRecReqsText'),
#                         ('has_linux_min_reqs', 'LinuxMinReqsText'),
#                         ('has_mac_min_reqs',   'MacMinReqsText'),
#                     ]:
#                         if src in df.columns:
#                             df[flag] = df[src].fillna('').apply(lambda x: 1 if len(x.strip()) > 5 else 0)

#                     log(f"  flag/text features engineered → {df.shape[1]} cols total")

#                     # ── STEP 8: Drop all text / string columns ────────────
#                     drop_text_cols = [
#                         'QueryName', 'ResponseName', 'Website', 'SupportEmail', 'SupportURL',
#                         'LegalNotice', 'Reviews', 'SupportedLanguages', 'ShortDescrip',
#                         'DetailedDescrip', 'DRMNotice', 'ExtUserAcctNotice', 'PriceCurrency',
#                         'Background', 'HeaderImage',
#                         'AboutText', 'PCMinReqsText', 'PCRecReqsText',
#                         'LinuxMinReqsText', 'MacMinReqsText',
#                     ]
#                     df.drop(columns=[c for c in drop_text_cols if c in df.columns], inplace=True)

#                     # Drop any remaining object columns
#                     remaining_text = df.select_dtypes(include='object').columns.tolist()
#                     if remaining_text:
#                         df.drop(columns=remaining_text, inplace=True)
#                         log(f"  also dropped remaining object cols: {remaining_text}", "muted")

#                     log(f"  after text drop → {df.shape[1]} numeric cols", "muted")

#                     # Optionally merge NLP features
#                     if os.path.exists(NLP_PATH):
#                         nlp = pd.read_csv(NLP_PATH)
#                         shared = [c for c in nlp.columns if c not in df.columns]
#                         if shared and len(nlp) == len(df):
#                             df = pd.concat([df.reset_index(drop=True), nlp[shared].reset_index(drop=True)], axis=1)
#                             log(f"  merged {len(shared)} NLP features → {df.shape[1]} total cols")

#                     # ── STEP 9: Split ──────────────────────────────────────
#                     # Target is raw RecommendationCount (consistent with script)
#                     X = df.drop(columns=['RecommendationCount'])
#                     y = df['RecommendationCount']

#                     X_temp, X_test, y_temp, y_test = train_test_split(
#                         X, y, test_size=test_size, random_state=int(seed)
#                     )
#                     X_train, X_val, y_train, y_val = train_test_split(
#                         X_temp, y_temp, test_size=val_frac, random_state=int(seed)
#                     )
#                     log(f"  split → train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

#                     # ── STEP 10: Median fill (fit on train only) ──────────
#                     num_cols_all = X_train.select_dtypes(include=np.number).columns
#                     train_medians = X_train[num_cols_all].median()
#                     X_train[num_cols_all] = X_train[num_cols_all].fillna(train_medians)
#                     X_val[num_cols_all]   = X_val[num_cols_all].fillna(train_medians)
#                     X_test[num_cols_all]  = X_test[num_cols_all].fillna(train_medians)
#                     remaining_nans = (
#                         X_train.isnull().sum().sum() +
#                         X_val.isnull().sum().sum() +
#                         X_test.isnull().sum().sum()
#                     )
#                     log(f"  median fill complete — remaining NaNs: {remaining_nans}")

#                     # ── Identify continuous cols (same logic as script) ────
#                     CONTINUOUS_COLS = [
#                         'RequiredAge', 'DemoCount', 'DeveloperCount', 'DLCCount',
#                         'MovieCount', 'PackageCount', 'RecommendationCount', 'PublisherCount',
#                         'ScreenshotCount',
#                         'SteamSpyOwners', 'SteamSpyOwnersVariance',
#                         'SteamSpyPlayersEstimate', 'SteamSpyPlayersVariance',
#                         'AchievementCount', 'AchievementHighlightedCount',
#                         'PriceInitial', 'PriceFinal',
#                         'release_year', 'release_month', 'game_age_days',
#                         'num_languages',
#                         'about_length', 'short_length', 'detail_length',
#                         'SteamSpyOwners_log', 'SteamSpyOwnersVariance_log',
#                         'SteamSpyPlayersEstimate_log', 'SteamSpyPlayersVariance_log',
#                         'target_log',
#                     ]
#                     # Only keep cols that actually exist in X_train
#                     cont_feat_cols = [c for c in CONTINUOUS_COLS if c in X_train.columns]

#                     # ── STEP 11: IQR Capping (fit on train) ───────────────
#                     total_clipped = 0
#                     for col in cont_feat_cols:
#                         Q1 = X_train[col].quantile(0.25)
#                         Q3 = X_train[col].quantile(0.75)
#                         IQR = Q3 - Q1
#                         lo  = Q1 - iqr_factor * IQR
#                         hi  = Q3 + iqr_factor * IQR
#                         n   = ((X_train[col] < lo) | (X_train[col] > hi)).sum()
#                         total_clipped += n
#                         X_train[col] = X_train[col].clip(lo, hi)
#                         X_val[col]   = X_val[col].clip(lo, hi)
#                         X_test[col]  = X_test[col].clip(lo, hi)
#                     log(f"  IQR capping (×{iqr_factor}): {total_clipped} values clipped across {len(cont_feat_cols)} cols")

#                     # ── STEP 12: Standard Scaling (fit on train) ──────────
#                     if do_scale and cont_feat_cols:
#                         scaler = StandardScaler()
#                         X_train[cont_feat_cols] = scaler.fit_transform(X_train[cont_feat_cols])
#                         X_val[cont_feat_cols]   = scaler.transform(X_val[cont_feat_cols])
#                         X_test[cont_feat_cols]  = scaler.transform(X_test[cont_feat_cols])
#                         log(f"  StandardScaler applied to {len(cont_feat_cols)} continuous cols")
#                     elif not do_scale:
#                         log("  StandardScaler skipped (disabled in settings)", "muted")

#                     log("// Pipeline complete ✓")

#                     st.session_state.update({
#                         "X_train": X_train, "X_val": X_val, "X_test": X_test,
#                         "y_train": y_train, "y_val": y_val, "y_test": y_test,
#                         "cont_cols": cont_feat_cols,
#                         "pipeline_done": True,
#                     })

#                 except Exception as e:
#                     import traceback
#                     log(f"ERROR: {e}", "error")
#                     log(traceback.format_exc(), "error")

#                 # Render log
#                 log_html = ""
#                 for line, kind in log_lines:
#                     css = {"muted": "color:#6b7280", "error": "color:#f65b8d"}.get(kind, "color:#5bf6c8")
#                     safe = line.replace("<", "&lt;").replace(">", "&gt;")
#                     log_html += f'<div style="{css}">{safe}</div>'
#                 st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

#                 if st.session_state.get("pipeline_done"):
#                     c1, c2, c3 = st.columns(3)
#                     c1.metric("Train", f"{len(st.session_state['X_train']):,}")
#                     c2.metric("Val",   f"{len(st.session_state['X_val']):,}")
#                     c3.metric("Test",  f"{len(st.session_state['X_test']):,}")

# # ═══════════════════════════════════════════════════════════════
# # TAB 2 — RESULTS
# # ═══════════════════════════════════════════════════════════════
# with tab_results:
#     if not st.session_state.get("pipeline_done"):
#         st.info("Run the pipeline first.")
#     else:
#         X_train   = st.session_state["X_train"]
#         y_train   = st.session_state["y_train"]
#         cont_cols = st.session_state["cont_cols"]

#         def dark_fig(nrows=1, ncols=1, figsize=(12, 4)):
#             fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='#13161e')
#             ax_list = axes.flatten() if hasattr(axes, 'flatten') else [axes]
#             for ax in ax_list:
#                 ax.set_facecolor('#1a1e2b')
#                 ax.tick_params(colors='#6b7280', labelsize=8)
#                 for sp in ax.spines.values():
#                     sp.set_color('#252a38')
#             return fig, axes

#         # Overview metrics
#         m1, m2, m3, m4 = st.columns(4)
#         m1.metric("Train rows",  f"{len(X_train):,}")
#         m2.metric("Features",    X_train.shape[1])
#         m3.metric("Cont. cols",  len(cont_cols))
#         m4.metric("Val rows",    f"{len(st.session_state['X_val']):,}")

#         st.markdown("---")

#         # Feature distribution
#         st.markdown("""<div class="section-header"><div class="section-dot blue"></div>
#         <div class="section-title">Feature Distribution</div></div>""", unsafe_allow_html=True)

#         avail = [c for c in cont_cols if c in X_train.columns]
#         sel   = st.selectbox("Select feature", avail)

#         if sel:
#             fig, axes = dark_fig(1, 2, (12, 3.5))
#             ax1, ax2  = axes.flatten()
#             vals = X_train[sel].dropna()

#             ax1.hist(vals, bins=60, color='#5b8df6', alpha=0.85, edgecolor='none')
#             ax1.axvline(vals.mean(),   color='#5bf6c8', lw=1.5, linestyle='--', label='mean')
#             ax1.axvline(vals.median(), color='#f6c85b', lw=1.2, linestyle=':',  label='median')
#             ax1.legend(fontsize=8, framealpha=0.4)
#             ax1.set_title(f"Distribution · {sel}", color='#e8eaf2', fontsize=9, fontfamily='monospace')

#             ax2.boxplot(vals, vert=False, patch_artist=True,
#                         medianprops=dict(color='#5bf6c8', lw=2),
#                         boxprops=dict(facecolor=(0.357, 0.553, 0.965, 0.3), color='#5b8df6'),
#                         whiskerprops=dict(color='#6b7280'),
#                         capprops=dict(color='#6b7280'),
#                         flierprops=dict(marker='.', color='#6b7280', markersize=3, alpha=0.3))
#             ax2.set_title(f"Box Plot · {sel}", color='#e8eaf2', fontsize=9, fontfamily='monospace')

#             plt.tight_layout(pad=1.5)
#             st.pyplot(fig, use_container_width=True)
#             plt.close()

#         # Target distribution (raw RecommendationCount)
#         st.markdown("---")
#         st.markdown("""<div class="section-header"><div class="section-dot pink"></div>
#         <div class="section-title">Target Distribution</div></div>""", unsafe_allow_html=True)

#         fig2, axes2 = dark_fig(1, 2, (12, 3.5))
#         ax_a, ax_b  = axes2.flatten()

#         raw_vals = y_train.values
#         ax_a.hist(raw_vals, bins=80, color='#f65b8d', alpha=0.85, edgecolor='none')
#         ax_a.axvline(np.mean(raw_vals), color='#5bf6c8', lw=1.5, linestyle='--', label='mean')
#         ax_a.legend(fontsize=8, framealpha=0.4)
#         ax_a.set_title("RecommendationCount (raw)", color='#e8eaf2', fontsize=9, fontfamily='monospace')

#         log_vals = np.log1p(raw_vals)
#         ax_b.hist(log_vals, bins=60, color='#f6c85b', alpha=0.85, edgecolor='none')
#         ax_b.axvline(np.mean(log_vals), color='#5bf6c8', lw=1.5, linestyle='--', label='mean')
#         ax_b.legend(fontsize=8, framealpha=0.4)
#         ax_b.set_title("log1p(RecommendationCount)", color='#e8eaf2', fontsize=9, fontfamily='monospace')

#         plt.tight_layout(pad=1.5)
#         st.pyplot(fig2, use_container_width=True)
#         plt.close()

#         # Correlation heatmap (top 15 vs target)
#         st.markdown("---")
#         st.markdown("""<div class="section-header"><div class="section-dot"></div>
#         <div class="section-title">Top 15 Feature Correlations with Target</div></div>""", unsafe_allow_html=True)

#         combined = X_train[cont_cols].copy()
#         combined['RecommendationCount'] = y_train.values
#         corrs    = combined.corr()['RecommendationCount'].drop('RecommendationCount').abs().nlargest(15)
#         top_cols = corrs.index.tolist() + ['RecommendationCount']
#         corr_mat = combined[top_cols].corr()

#         fig3, ax3 = plt.subplots(figsize=(12, 5), facecolor='#13161e')
#         ax3.set_facecolor('#13161e')
#         im = ax3.imshow(corr_mat.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
#         ax3.set_xticks(range(len(top_cols))); ax3.set_yticks(range(len(top_cols)))
#         ax3.set_xticklabels(top_cols, rotation=45, ha='right', fontsize=7, color='#6b7280')
#         ax3.set_yticklabels(top_cols, fontsize=7, color='#6b7280')
#         for sp in ax3.spines.values(): sp.set_color('#252a38')
#         for i in range(len(top_cols)):
#             for j in range(len(top_cols)):
#                 v = corr_mat.values[i, j]
#                 ax3.text(j, i, f"{v:.2f}", ha='center', va='center',
#                          fontsize=6, color='white' if abs(v) > 0.5 else '#6b7280')
#         plt.colorbar(im, ax=ax3, fraction=0.025, pad=0.04)
#         ax3.set_title("Correlation Matrix", color='#e8eaf2', fontsize=10, fontfamily='monospace', pad=12)
#         plt.tight_layout()
#         st.pyplot(fig3, use_container_width=True)
#         plt.close()

#         # Stats table
#         st.markdown("---")
#         st.markdown("""<div class="section-header"><div class="section-dot blue"></div>
#         <div class="section-title">Post-Scaling Stats (train)</div></div>""", unsafe_allow_html=True)
#         if cont_cols:
#             avail_cont = [c for c in cont_cols if c in X_train.columns]
#             st.dataframe(
#                 X_train[avail_cont].describe().T[['mean', 'std', 'min', 'max']].round(4),
#                 use_container_width=True, height=320
#             )