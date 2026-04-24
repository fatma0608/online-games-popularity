import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, warnings, joblib, traceback
warnings.filterwarnings('ignore')

from sklearn.linear_model      import Ridge
from sklearn.tree              import DecisionTreeRegressor
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection   import RandomizedSearchCV
from sklearn.metrics           import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from scipy.stats               import loguniform, randint, uniform

st.set_page_config(page_title="Models · SteamML", page_icon="🤖", layout="wide")

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
.ph{background:linear-gradient(135deg,#0d0f14,#111827,#0d1420);border:1px solid var(--border);
    border-left:4px solid var(--accent);border-radius:12px;padding:28px 36px;margin-bottom:24px;}
.ph h1{font-size:24px;font-weight:600;margin:0 0 4px;color:var(--text);}
.ph p{font-size:12px;color:var(--muted);margin:0;}
.log-box{background:#080a0e;border:1px solid var(--border);border-radius:8px;padding:14px 18px;
    font-family:monospace;font-size:11px;color:var(--accent3);white-space:pre-wrap;
    max-height:380px;overflow-y:auto;line-height:1.8;}
.stButton>button{background:var(--accent)!important;color:#0d0f14!important;border:none!important;
    border-radius:8px!important;font-family:var(--mono)!important;font-size:12px!important;
    letter-spacing:1px!important;padding:12px 28px!important;font-weight:700!important;width:100%!important;}
[data-testid="stMetric"]{background:var(--surface)!important;border:1px solid var(--border)!important;
    border-radius:10px!important;padding:14px!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:11px!important;}
[data-testid="stMetricValue"]{color:var(--text)!important;}
[data-testid="stTabs"] button{font-family:var(--mono)!important;font-size:11px!important;color:var(--muted)!important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--accent)!important;border-bottom-color:var(--accent)!important;}
.sec{display:flex;align-items:center;gap:10px;margin:20px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--border);}
.sec-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.sec-lbl{font-family:var(--mono);font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--text);}
.model-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
    padding:18px 22px;margin-bottom:12px;}
.model-card h3{margin:0 0 6px;font-size:14px;font-family:var(--mono);color:var(--text);}
.model-card p{margin:0;font-size:11px;color:var(--muted);line-height:1.7;}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:10px;
    font-family:var(--mono);font-weight:700;margin-right:6px;}
.predict-box{background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:22px 26px;margin-top:18px;}
.predict-result{background:linear-gradient(135deg,#0d0f14,#111827);border:1px solid var(--accent3);
    border-radius:10px;padding:20px;text-align:center;margin-top:16px;}
.predict-result .val{font-size:42px;font-weight:700;font-family:var(--mono);color:var(--accent3);}
.predict-result .sub{font-size:11px;color:var(--muted);margin-top:4px;}
.status-ok{color:#5bf6c8;font-family:monospace;font-size:11px;}
.status-no{color:#6b7280;font-family:monospace;font-size:11px;}
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] select,
div[data-testid="stTextInput"] input{
    background:var(--surface2)!important;color:var(--text)!important;
    border:1px solid var(--border)!important;border-radius:6px!important;}
.stDataFrame{background:var(--surface)!important;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════
TARGET     = 'target_log'
MODELS_DIR = './models'
DATA_SEL   = './data/selected'

MODEL_FILES = {
    'Ridge'            : 'ridge_tuned.pkl',
    'Decision Tree'    : 'decision_tree_tuned.pkl',
    'Random Forest'    : 'random_forest_tuned.pkl',
    'Gradient Boosting': 'gradient_boosting_tuned.pkl',
}

MODEL_COLORS = {
    'Ridge'            : '#4C8EDA',
    'Decision Tree'    : '#8D6E63',
    'Random Forest'    : '#1D9E75',
    'Gradient Boosting': '#E8593C',
}

MODEL_DESC = {
    'Ridge': (
        "Linear model with L2 regularisation. Fast, interpretable, strong baseline. "
        "Handles multicollinearity well but cannot capture non-linear patterns.",
        ["alpha", "fit_intercept", "solver"]
    ),
    'Decision Tree': (
        "Single tree; interpretable and fast. High variance without regularisation. "
        "Useful as a baseline and as the building block for ensembles.",
        ["max_depth", "min_samples_leaf", "min_samples_split", "max_features", "criterion"]
    ),
    'Random Forest': (
        "Ensemble of decorrelated decision trees. Naturally captures non-linearities "
        "and interactions. Low variance via bagging; reliable feature importances.",
        ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features"]
    ),
    'Gradient Boosting': (
        "Sequential boosting; each tree corrects the residuals of the previous one. "
        "High accuracy but slower to train. Prone to overfit without regularisation.",
        ["n_estimators", "learning_rate", "max_depth", "subsample", "min_samples_leaf"]
    ),
    
}

# ═══════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════
def dark_fig(nrows=1, ncols=1, figsize=(12, 4)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='#13161e')
    axl = np.array(axes).flatten() if hasattr(axes, '__len__') else [axes]
    for ax in axl:
        ax.set_facecolor('#1a1e2b')
        ax.tick_params(colors='#6b7280', labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('#252a38')
    return fig, (np.array(axes).flatten() if hasattr(axes, '__len__') else [axes])


def sec(label, color='#5b8df6'):
    st.markdown(
        f'<div class="sec">'
        f'<div class="sec-dot" style="background:{color}"></div>'
        f'<div class="sec-lbl">{label}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def render_log(lines):
    html = "".join(
        '<div style=\'{c}\'>{m}</div>'.format(
            c='color:#6b7280' if k == 'muted' else 'color:#f65b8d' if k == 'err' else 'color:#5bf6c8',
            m=m.replace('<', '&lt;').replace('>', '&gt;')
        )
        for m, k in lines
    )
    st.markdown(f'<div class="log-box">{html}</div>', unsafe_allow_html=True)


def model_exists(name):
    return os.path.exists(os.path.join(MODELS_DIR, MODEL_FILES[name]))


def load_model(name):
    return joblib.load(os.path.join(MODELS_DIR, MODEL_FILES[name]))


def data_ready():
    """Check for at least train + test selected CSVs (val is optional)."""
    return (
        os.path.exists(os.path.join(DATA_SEL, 'train_selected.csv')) and
        os.path.exists(os.path.join(DATA_SEL, 'test_selected.csv'))
    )


def get_model_feature_names(model):
    """Extract feature names a model was trained on. Returns list or None."""
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    if hasattr(model, 'named_steps'):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    if hasattr(model, 'booster_') and hasattr(model.booster_, 'feature_name'):
        names = model.booster_.feature_name()
        if names:
            return names
    if hasattr(model, 'feature_names_'):
        return list(model.feature_names_)
    return None


def align_features(model, X):
    """Reindex X columns to match what the model was trained on."""
    expected = get_model_feature_names(model)
    if expected is None:
        return X
    X = X.copy()
    for col in expected:
        if col not in X.columns:
            X[col] = 0.0
    return X[expected]


def compute_metrics(model, X_tr, y_tr, X_te, y_te):
    X_tr = align_features(model, X_tr)
    X_te = align_features(model, X_te)
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


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='font-family:monospace;font-size:18px;font-weight:700;"
        "color:#5b8df6;padding:8px 0 20px'>🎮 SteamML</div>",
        unsafe_allow_html=True
    )
    st.page_link("app.py",                       label="⬡  Dashboard")
    st.page_link("pages/preprocessing_page.py",  label="⬡  Preprocessing & NLP")
    st.page_link("pages/model_page.py",          label="⬡  Models")
    st.markdown("---")

    fs_done    = data_ready()
    any_model  = any(model_exists(n) for n in MODEL_FILES)
    all_models = all(model_exists(n) for n in MODEL_FILES)

    fs_color = '#5bf6c8' if fs_done else '#6b7280'
    fs_dot   = '●' if fs_done else '○'
    fs_label = 'Feature selection done ✓' if fs_done else 'Feature selection not run'

    if all_models:
        m_color, m_dot, m_label = '#5bf6c8', '●', 'Models all saved ✓'
    elif any_model:
        m_color, m_dot, m_label = '#f6c85b', '◑', 'Models partial'
    else:
        m_color, m_dot, m_label = '#6b7280', '○', 'Models not trained'

    st.markdown(
        f"<div style='font-size:12px;line-height:2.2'>"
        f"<span style='color:{fs_color}'>{fs_dot}</span>&nbsp;{fs_label}<br>"
        f"<span style='color:{m_color}'>{m_dot}</span>&nbsp;{m_label}"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:10px;color:#6b7280;font-family:monospace'>SAVED MODELS</div>",
        unsafe_allow_html=True
    )
    for name in MODEL_FILES:
        exists    = model_exists(name)
        col_saved = MODEL_COLORS[name] if exists else '#3a3f52'
        dot       = '●' if exists else '○'
        txt_color = '#e8eaf2' if exists else '#6b7280'
        st.markdown(
            f"<div style='font-size:11px;margin:4px 0'>"
            f"<span style='color:{col_saved}'>{dot}</span>"
            f"&nbsp;<span style='color:{txt_color}'>{name}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

# ═══════════════════════════════════════════════════════
# PAGE HEADER
# ═══════════════════════════════════════════════════════
st.markdown("""<div class="ph">
    <h1>🤖  Models &amp; Feature Selection</h1>
    <p>Feature selection pipeline · Train / load saved models · Compare performance · Predict new games</p>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════
tab_fs, tab_train, tab_compare, tab_predict = st.tabs([
    "🔍  FEATURE SELECTION",
    "⚙️  TRAIN / LOAD MODELS",
    "📊  MODEL COMPARISON",
    "🎯  PREDICT NEW GAME",
])

# ───────────────────────────────────────────────────────
# TAB 1 ── FEATURE SELECTION
# ───────────────────────────────────────────────────────
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
        unsafe_allow_html=True
    )

    # Check prerequisites: at minimum need numeric processed train + test
    prereq_ok = (
        os.path.exists('./data/processed/train.csv') and
        os.path.exists('./data/processed/test.csv')
    )
    nlp_avail = (
        os.path.exists('./data/processed/nlp_features_train.csv') and
        os.path.exists('./data/processed/nlp_features_test.csv')
    )
    if not prereq_ok:
        st.warning("⚠️  Run the **Preprocessing** pipeline first to generate processed CSVs.")

    cl, cr = st.columns([1, 1], gap="large")
    with cl:
        if data_ready():
            st.success("✓ Selected features already saved — click to re-run and overwrite.")
        else:
            st.info("ℹ️  Selected features not found — run the pipeline below.")
        run_fs = st.button("▶  RUN FEATURE SELECTION", key="run_fs", disabled=not prereq_ok)

    with cr:
        st.markdown(
            "<div style='background:#13161e;border:1px solid #252a38;border-radius:10px;"
            "padding:16px 18px;font-size:12px;color:#6b7280;line-height:2'>"
            "① Merge processed + NLP CSVs (if available)<br>"
            "② Dominant-value filter  (≥95%)<br>"
            "③ VarianceThreshold  (var &lt; 0.01)<br>"
            "④ High inter-feature corr  (|r| &gt; 0.90)<br>"
            "⑤ Low target-corr  (|r| &lt; 0.01)<br>"
            "⑥ RF SelectFromModel  (threshold = median)<br>"
            "⑦ Save  →  ./data/selected/"
            "</div>",
            unsafe_allow_html=True
        )

    if run_fs:
        log = []
        L = lambda m, k="ok": log.append((m, k))
        try:
            L("// Feature Selection started ─────────────")

            def merge_split(num_df, nlp_df=None):
                num_df = num_df.reset_index(drop=True)
                if nlp_df is not None:
                    nlp_df = nlp_df.reset_index(drop=True)
                    min_len = min(len(num_df), len(nlp_df))
                    num_df = num_df.iloc[:min_len]
                    nlp_df = nlp_df.iloc[:min_len]
                    merged = pd.concat([num_df, nlp_df], axis=1)
                else:
                    merged = num_df
                return merged.loc[:, ~merged.columns.duplicated()]

            train_num = pd.read_csv('./data/processed/train.csv')
            test_num  = pd.read_csv('./data/processed/test.csv')

            if nlp_avail:
                train_nlp = pd.read_csv('./data/processed/nlp_features_train.csv')
                test_nlp  = pd.read_csv('./data/processed/nlp_features_test.csv')
                train_df  = merge_split(train_num, train_nlp)
                test_df   = merge_split(test_num,  test_nlp)
                L(f"  merged numeric + NLP  train={train_df.shape}  test={test_df.shape}")
            else:
                train_df = merge_split(train_num)
                test_df  = merge_split(test_num)
                L("  NLP features not found — using numeric only", "muted")
                L(f"  train={train_df.shape}  test={test_df.shape}")

            val_path = './data/processed/val.csv'
            if os.path.exists(val_path):
                val_num = pd.read_csv(val_path)
                if nlp_avail and os.path.exists('./data/processed/nlp_features_val.csv'):
                    val_nlp = pd.read_csv('./data/processed/nlp_features_val.csv')
                    val_df  = merge_split(val_num, val_nlp)
                else:
                    val_df = merge_split(val_num)
                has_val = True
            else:
                val_df  = None
                has_val = False

            X_train = train_df.drop(columns=[TARGET]); y_train = train_df[TARGET]
            X_test  = test_df.drop(columns=[TARGET]);  y_test  = test_df[TARGET]
            if has_val:
                X_val = val_df.drop(columns=[TARGET]); y_val = val_df[TARGET]
            initial = X_train.shape[1]

            # 3a: Dominant-value
            dom_drop = [
                c for c in X_train.columns
                if X_train[c].value_counts(normalize=True, dropna=False).max() >= 0.95
            ]
            sets_to_filter = [X_train, X_test] + ([X_val] if has_val else [])
            for df_ in sets_to_filter:
                df_.drop(columns=dom_drop, inplace=True)
            L(f"  [3a] dominant-value  → dropped {len(dom_drop):3d}  remaining: {X_train.shape[1]}", "muted")

            # 3b: VarianceThreshold
            cols_bvt = X_train.columns.tolist()
            vt = VarianceThreshold(threshold=0.01)
            Xtr_a  = vt.fit_transform(X_train)
            Xte_a  = vt.transform(X_test)
            sel_vt = [c for c, k in zip(cols_bvt, vt.get_support()) if k]
            drp_vt = [c for c, k in zip(cols_bvt, vt.get_support()) if not k]
            X_train = pd.DataFrame(Xtr_a, columns=sel_vt)
            X_test  = pd.DataFrame(Xte_a, columns=sel_vt)
            if has_val:
                X_val = pd.DataFrame(vt.transform(X_val), columns=sel_vt)
            L(f"  [3b] VarianceThreshold → dropped {len(drp_vt):3d}  remaining: {X_train.shape[1]}", "muted")

            # 3c: High inter-feature correlation
            corr_mat = X_train.corr().abs()
            upper    = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
            tgt_corr = X_train.corrwith(y_train).abs()
            corr_drop = []
            for col in upper.columns:
                for partner in upper.index[upper[col] > 0.90].tolist():
                    if col not in corr_drop and partner not in corr_drop:
                        drop_col = col if tgt_corr.get(col, 0) < tgt_corr.get(partner, 0) else partner
                        corr_drop.append(drop_col)
            drop_sets = [X_train, X_test] + ([X_val] if has_val else [])
            for df_ in drop_sets:
                df_.drop(columns=corr_drop, inplace=True, errors='ignore')
            L(f"  [3c] high corr       → dropped {len(corr_drop):3d}  remaining: {X_train.shape[1]}", "muted")

            # 3d: Low target-correlation
            low_series = X_train.corrwith(y_train).abs()
            low_drop   = low_series[low_series < 0.01].index.tolist()
            drop_sets2 = [X_train, X_test] + ([X_val] if has_val else [])
            for df_ in drop_sets2:
                df_.drop(columns=low_drop, inplace=True)
            L(f"  [3d] low target-corr → dropped {len(low_drop):3d}  remaining: {X_train.shape[1]}", "muted")

            # 3e: RF SelectFromModel
            L("  [3e] fitting RandomForest for importance …")
            rf_sel = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_sel.fit(X_train, y_train)
            all_feats = X_train.columns.tolist()
            imp_df = pd.DataFrame({
                'feature'   : all_feats,
                'importance': rf_sel.feature_importances_,
            }).sort_values('importance', ascending=False)
            selector = SelectFromModel(rf_sel, threshold="median", prefit=True)
            Xtr_s  = selector.transform(X_train)
            Xte_s  = selector.transform(X_test)
            sel_rf = [c for c, k in zip(all_feats, selector.get_support()) if k]
            X_train = pd.DataFrame(Xtr_s, columns=sel_rf)
            X_test  = pd.DataFrame(Xte_s, columns=sel_rf)
            if has_val:
                X_val = pd.DataFrame(selector.transform(X_val), columns=sel_rf)
            L(f"  [3e] SelectFromModel → kept {len(sel_rf)}  dropped {len(all_feats) - len(sel_rf)}", "muted")

            # Save
            os.makedirs('./data/selected', exist_ok=True)
            out_train = X_train.copy(); out_train[TARGET] = y_train.values
            out_test  = X_test.copy();  out_test[TARGET]  = y_test.values
            out_train.to_csv('./data/selected/train_selected.csv', index=False)
            out_test.to_csv('./data/selected/test_selected.csv',   index=False)
            if has_val:
                out_val = X_val.copy(); out_val[TARGET] = y_val.values
                out_val.to_csv('./data/selected/val_selected.csv', index=False)

            L("  saved to ./data/selected/")
            L(f"  initial={initial}  final={X_train.shape[1]}  removed={initial - X_train.shape[1]}")
            L("// Feature Selection Done ✓ ──────────────")

            st.session_state['fs_imp_df']    = imp_df
            st.session_state['fs_sel_feats'] = sel_rf
            st.session_state['fs_initial']   = initial
            st.session_state['fs_final']     = X_train.shape[1]
            st.session_state['fs_done_run']  = True

        except Exception as e:
            L(f"ERROR: {e}", "err")
            L(traceback.format_exc(), "err")
        render_log(log)
        st.rerun()

    # Show results if data is ready
    if data_ready():
        train_sel = pd.read_csv('./data/selected/train_selected.csv')
        test_sel  = pd.read_csv('./data/selected/test_selected.csv')
        feat_cols = [c for c in train_sel.columns if c != TARGET]

        sec("Selection Summary", "#5bf6c8")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Features", len(feat_cols))
        c2.metric("Train Rows",     f"{len(train_sel):,}")
        c3.metric("Test Rows",      f"{len(test_sel):,}")
        c4.metric("Target",         "target_log (log1p)")

        sec("Feature List", "#5b8df6")
        feat_display = pd.DataFrame({
            'Feature': feat_cols,
            'Type':    ['NLP/LSA' if 'lsa' in f else 'Numeric' for f in feat_cols],
        })
        st.dataframe(feat_display, use_container_width=True, height=280)

        imp_df = st.session_state.get('fs_imp_df')
        if imp_df is not None:
            sec("RandomForest Feature Importances — Top 30", "#f6c85b")
            top30 = imp_df.head(30)
            fig, axs = dark_fig(1, 1, (14, 7))
            clrs = [
                MODEL_COLORS['Random Forest'] if 'lsa' not in f else '#7F77DD'
                for f in top30['feature']
            ]
            axs[0].barh(range(len(top30)), top30['importance'].values[::-1],
                        color=clrs[::-1], alpha=0.85, height=0.75)
            axs[0].set_yticks(range(len(top30)))
            axs[0].set_yticklabels(top30['feature'].values[::-1], fontsize=8, color='#e8eaf2')
            axs[0].set_title('Top 30 Feature Importances (green=numeric, purple=NLP/LSA)',
                             color='#e8eaf2', fontsize=9, fontfamily='monospace')
            axs[0].set_xlabel('Importance', fontsize=8, color='#6b7280')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        sec("Feature Correlation with Target", "#f65b8d")
        X_disp   = train_sel[feat_cols]
        y_disp   = train_sel[TARGET]
        corr_tgt = X_disp.corrwith(y_disp).abs().sort_values(ascending=False).head(25)
        fig, axs = dark_fig(1, 1, (14, 5))
        clrs2 = [
            '#5bf6c8' if v > 0.3 else '#5b8df6' if v > 0.1 else '#6b7280'
            for v in corr_tgt.values
        ]
        axs[0].barh(range(len(corr_tgt)), corr_tgt.values[::-1],
                    color=clrs2[::-1], alpha=0.85, height=0.65)
        axs[0].set_yticks(range(len(corr_tgt)))
        axs[0].set_yticklabels(corr_tgt.index[::-1], fontsize=8, color='#e8eaf2')
        axs[0].set_xlabel('|Pearson r|', fontsize=8, color='#6b7280')
        axs[0].set_title('Top 25 Features — Absolute Correlation with target_log',
                         color='#e8eaf2', fontsize=9, fontfamily='monospace')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ───────────────────────────────────────────────────────
# TAB 2 ── TRAIN / LOAD MODELS
# ───────────────────────────────────────────────────────
with tab_train:
    if not data_ready():
        st.warning("⚠️  Run **Feature Selection** first.")
        st.stop()

    @st.cache_data
    def load_data():
        train_df = pd.read_csv('./data/selected/train_selected.csv')
        test_df  = pd.read_csv('./data/selected/test_selected.csv')
        X_train  = train_df.drop(columns=[TARGET]); y_train = train_df[TARGET]
        X_test   = test_df.drop(columns=[TARGET]);  y_test  = test_df[TARGET]

        val_path = './data/selected/val_selected.csv'
        if os.path.exists(val_path):
            val_df = pd.read_csv(val_path)
            X_val  = val_df.drop(columns=[TARGET]); y_val = val_df[TARGET]
            X_tv   = pd.concat([X_train, X_val]).reset_index(drop=True)
            y_tv   = pd.concat([y_train, y_val]).reset_index(drop=True)
        else:
            X_tv, y_tv = X_train.reset_index(drop=True), y_train.reset_index(drop=True)

        return X_train, y_train, X_test, y_test, X_tv, y_tv

    X_train, y_train, X_test, y_test, X_tv, y_tv = load_data()

    st.markdown(
        f"<div style='background:#13161e;border:1px solid #252a38;border-radius:10px;"
        f"padding:14px 20px;font-size:12px;color:#6b7280;margin-bottom:18px'>"
        f"Data loaded · train+val = <b style='color:#e8eaf2'>{len(X_tv):,}</b> rows · "
        f"test = <b style='color:#e8eaf2'>{len(X_test):,}</b> rows · "
        f"features = <b style='color:#e8eaf2'>{X_train.shape[1]}</b>"
        f"</div>",
        unsafe_allow_html=True
    )

    for name, (desc, hp_list) in MODEL_DESC.items():
        col_hex = MODEL_COLORS[name]
        exists  = model_exists(name)
        label   = f"{'✅' if exists else '⬜'}  {name}  {'— model saved ✓' if exists else '— not trained yet'}"

        with st.expander(label, expanded=False):
            cl, cr = st.columns([3, 1], gap="large")
            with cl:
                badges = ''.join(
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
                    unsafe_allow_html=True
                )
            with cr:
                btn_label = f"🔄  {'Re-train' if exists else 'Train'} {name}"
                btn_train = st.button(btn_label, key=f"train_{name}")
                if exists:
                    st.markdown("<span class='status-ok'>● saved</span>", unsafe_allow_html=True)
                    try:
                        m = load_model(name)
                        params = list(m.get_params().items())[:6]
                        param_html = '<br>'.join(
                            f"{k}: <span style='color:#5bf6c8'>{v}</span>"
                            for k, v in params
                        )
                        st.markdown(
                            f"<div style='font-size:10px;color:#6b7280;margin-top:6px'>"
                            f"<b style='color:#e8eaf2'>Best params:</b><br>{param_html}"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    except Exception:
                        pass
                else:
                    st.markdown("<span class='status-no'>○ not trained</span>", unsafe_allow_html=True)

            if btn_train:
                log = []
                L = lambda m, k="ok": log.append((m, k))
                with st.spinner(f"Training {name} …"):
                    try:
                        os.makedirs(MODELS_DIR, exist_ok=True)
                        L(f"// Training {name} ─────────────────────")

                        if name == 'Ridge':
                            param_dist = {
                                'alpha'         : loguniform(1e-3, 1e4),
                                'fit_intercept' : [True, False],
                                'solver'        : ['auto', 'svd', 'cholesky', 'lsqr', 'saga'],
                            }
                            base = Ridge()

                        elif name == 'Decision Tree':
                            param_dist = {
                                'max_depth'         : [3, 4, 5, 6, 8, 10, 15, None],
                                'min_samples_split' : randint(2, 40),
                                'min_samples_leaf'  : randint(1, 30),
                                'max_features'      : ['sqrt', 'log2', 0.5, 0.7, None],
                                'criterion'         : ['squared_error', 'friedman_mse', 'absolute_error'],
                            }
                            base = DecisionTreeRegressor(random_state=42)

                        elif name == 'Random Forest':
                            param_dist = {
                                'n_estimators'     : randint(100, 600),
                                'max_depth'        : [None, 5, 10, 15, 20, 30],
                                'min_samples_leaf' : randint(1, 30),
                                'min_samples_split': randint(2, 20),
                                'max_features'     : ['sqrt', 'log2', 0.3, 0.5, 0.7],
                            }
                            base = RandomForestRegressor(random_state=42, n_jobs=-1)

                        else:
                            param_dist = {
                                'n_estimators'    : randint(100, 600),
                                'learning_rate'   : loguniform(0.01, 0.3),
                                'max_depth'       : randint(2, 8),
                                'subsample'       : uniform(0.6, 0.4),
                                'min_samples_leaf': randint(5, 40),
                                'max_features'    : ['sqrt', 'log2', 0.5, 0.7, None],
                            }
                            base = GradientBoostingRegressor(random_state=42)

                        
                        search = RandomizedSearchCV(
                            base, param_dist,
                            n_iter=40, cv=5, scoring='r2',
                            n_jobs=-1, random_state=42, verbose=0
                        )
                        search.fit(X_tv, y_tv)
                        best  = search.best_estimator_
                        cv_r2 = search.best_score_
                        L(f"  CV R²  = {cv_r2:.4f}")
                        L(f"  params = {search.best_params_}", "muted")

                        X_test_aligned = align_features(best, X_test)
                        p_te    = best.predict(X_test_aligned)
                        r2_te   = r2_score(y_test, p_te)
                        rmse_te = np.sqrt(mean_squared_error(y_test, p_te))
                        L(f"  test R² = {r2_te:.4f}  RMSE = {rmse_te:.4f}")

                        joblib.dump(best, os.path.join(MODELS_DIR, MODEL_FILES[name]))
                        L(f"  saved → {MODEL_FILES[name]}")
                        L("// Done ✓ ──────────────────────────────")

                    except Exception as e:
                        L(f"ERROR: {e}", "err")
                        L(traceback.format_exc(), "err")
                render_log(log)
                st.rerun()


# ───────────────────────────────────────────────────────
# TAB 3 ── MODEL COMPARISON
# ───────────────────────────────────────────────────────
with tab_compare:
    saved_models = {n: n for n in MODEL_FILES if model_exists(n)}

    if not saved_models:
        st.warning("⚠️  No saved models found. Train at least one model first.")
    elif not data_ready():
        st.warning("⚠️  Feature selection data not found.")
    else:
        @st.cache_data
        def load_test_data():
            tr  = pd.read_csv('./data/selected/train_selected.csv')
            te  = pd.read_csv('./data/selected/test_selected.csv')
            Xtr = tr.drop(columns=[TARGET]); ytr = tr[TARGET]
            Xte = te.drop(columns=[TARGET]); yte = te[TARGET]

            val_path = './data/selected/val_selected.csv'
            if os.path.exists(val_path):
                va  = pd.read_csv(val_path)
                Xva = va.drop(columns=[TARGET]); yva = va[TARGET]
                Xtv = pd.concat([Xtr, Xva]).reset_index(drop=True)
                ytv = pd.concat([ytr, yva]).reset_index(drop=True)
            else:
                Xtv, ytv = Xtr.reset_index(drop=True), ytr.reset_index(drop=True)

            return Xtv, ytv, Xte, yte

        X_tv_c, y_tv_c, X_test_c, y_test_c = load_test_data()

        results = {}
        skipped = []
        for name in saved_models:
            try:
                m = load_model(name)
                tr_m, te_m, p_tr, p_te = compute_metrics(m, X_tv_c, y_tv_c, X_test_c, y_test_c)
                results[name] = dict(train=tr_m, test=te_m, p_tr=p_tr, p_te=p_te, model=m)
            except Exception as e:
                skipped.append((name, str(e)))

        if skipped:
            for sname, serr in skipped:
                st.warning(f"⚠️  Skipped **{sname}**: {serr}")

        if not results:
            st.error("No models could be evaluated. Please re-train your models.")
        else:
            sec("Performance Summary — Test Set", "#5bf6c8")
            rows = []
            for name, res in results.items():
                te = res['test']
                tr = res['train']
                rows.append({
                    'Model'      : name,
                    'R² Test'    : round(te['r2'],        4),
                    'R² Train'   : round(tr['r2'],        4),
                    'Gap'        : round(tr['r2'] - te['r2'], 4),
                    'RMSE (log)' : round(te['rmse'],      4),
                    'MAE (log)'  : round(te['mae'],       4),
                    'RMSE (orig)': f"{te['rmse_orig']:,.0f}",
                    'MAE (orig)' : f"{te['mae_orig']:,.0f}",
                })
            df_res = pd.DataFrame(rows).sort_values('R² Test', ascending=False)
            st.dataframe(df_res, use_container_width=True, hide_index=True)

            best_name  = df_res.iloc[0]['Model']
            best_r2    = df_res.iloc[0]['R² Test']
            best_color = MODEL_COLORS.get(best_name, '#5bf6c8')
            st.markdown(
                f"<div style='margin-top:8px;font-size:12px;font-family:monospace'>"
                f"🏆 Best model: "
                f"<span style='color:{best_color};font-weight:700'>{best_name}</span>"
                f" · test R² = <span style='color:#5bf6c8'>{best_r2}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

            sec("Metric Comparison — Bar Charts", "#f6c85b")
            names  = list(results.keys())
            colors = [MODEL_COLORS.get(n, '#888') for n in names]

            fig, axs = dark_fig(1, 3, (18, 5))
            for ax, (metric, label) in zip(axs, [
                ('r2',   'R² (↑ better)'),
                ('rmse', 'RMSE log (↓ better)'),
                ('mae',  'MAE log (↓ better)'),
            ]):
                vals = [results[n]['test'][metric] for n in names]
                bars = ax.bar(names, vals, color=colors, alpha=0.85,
                              edgecolor='#0d0f14', linewidth=0.5, width=0.55)
                for bar, v in zip(bars, vals):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.01,
                        f'{v:.4f}', ha='center', va='bottom',
                        fontsize=7, color='#e8eaf2', rotation=45
                    )
                ax.set_title(label, color='#e8eaf2', fontsize=9, fontfamily='monospace')
                ax.tick_params(labelsize=7, colors='#6b7280')
                ax.set_xticklabels(names, rotation=30, ha='right', fontsize=7)
                ax.set_ylim(0, max(vals) * 1.22)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            sec("Overfitting Check — Train vs Test R²", "#f65b8d")
            x = np.arange(len(names))
            w = 0.35
            tr_r2s = [results[n]['train']['r2'] for n in names]
            te_r2s = [results[n]['test']['r2']  for n in names]
            fig, axs = dark_fig(1, 1, (14, 5))
            b1 = axs[0].bar(x - w/2, tr_r2s, w, label='Train R²',
                            color='#4C8EDA', alpha=0.85, edgecolor='#0d0f14')
            b2 = axs[0].bar(x + w/2, te_r2s, w, label='Test R²',
                            color='#E8593C', alpha=0.85, edgecolor='#0d0f14')
            for bar, v in zip(list(b1) + list(b2), tr_r2s + te_r2s):
                axs[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom',
                    fontsize=7, color='#e8eaf2', rotation=45
                )
            axs[0].set_xticks(x)
            axs[0].set_xticklabels(names, fontsize=8, color='#e8eaf2', rotation=20, ha='right')
            axs[0].set_ylim(0, 1.10)
            axs[0].set_ylabel('R²', fontsize=9, color='#6b7280')
            axs[0].set_title('Train vs Test R²  (gap > 0.10 = overfitting)',
                             color='#e8eaf2', fontsize=9, fontfamily='monospace')
            axs[0].legend(fontsize=9, framealpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            sec("Actual vs Predicted — Test Set", "#5b8df6")
            n_models = len(results)
            fig, axs = dark_fig(1, n_models, (5 * n_models, 5))
            for ax, (name, res) in zip(axs, results.items()):
                col_h = MODEL_COLORS.get(name, '#888')
                ax.scatter(y_test_c, res['p_te'], alpha=0.2, s=7, color=col_h, edgecolors='none')
                lo = min(float(y_test_c.min()), float(res['p_te'].min()))
                hi = max(float(y_test_c.max()), float(res['p_te'].max()))
                ax.plot([lo, hi], [lo, hi], color='#e8eaf2', lw=1.2, linestyle='--')
                ax.set_title(
                    f"{name}\nR²={res['test']['r2']:.4f}",
                    color='#e8eaf2', fontsize=9, fontfamily='monospace'
                )
                ax.set_xlabel('Actual (log)',    fontsize=8, color='#6b7280')
                ax.set_ylabel('Predicted (log)', fontsize=8, color='#6b7280')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            sec("Residual Distribution — Test Set", "#7F77DD")
            fig, axs = dark_fig(1, n_models, (5 * n_models, 4))
            for ax, (name, res) in zip(axs, results.items()):
                col_h = MODEL_COLORS.get(name, '#888')
                resid = np.array(y_test_c) - res['p_te']
                ax.hist(resid, bins=60, color=col_h, alpha=0.85, edgecolor='none')
                ax.axvline(0,            color='#e8eaf2', lw=1.2, linestyle='--', label='zero')
                ax.axvline(resid.mean(), color='#5bf6c8', lw=1.0, linestyle='-',
                           label=f'mean={resid.mean():.3f}')
                ax.set_title(name, color='#e8eaf2', fontsize=9, fontfamily='monospace')
                ax.set_xlabel('Residual', fontsize=8, color='#6b7280')
                ax.legend(fontsize=7, framealpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Feature importances (tree-based)
            tree_results = {
                n: r for n, r in results.items()
                if n in ('Random Forest', 'Gradient Boosting')
                and hasattr(r['model'], 'feature_importances_')
            }
            if tree_results:
                sec("Feature Importances — Top 20 (tree-based models)", "#1D9E75")
                n_tm = len(tree_results)
                fig, axs = dark_fig(1, n_tm, (6 * n_tm, 7))
                axs_list = list(axs) if n_tm > 1 else [axs[0]]
                for ax, (name, res) in zip(axs_list, tree_results.items()):
                    col_h = MODEL_COLORS.get(name, '#888')
                    model_feats = get_model_feature_names(res['model'])
                    if model_feats is None:
                        model_feats = [
                            c for c in pd.read_csv('./data/selected/train_selected.csv').columns
                            if c != TARGET
                        ]
                    imp   = pd.Series(res['model'].feature_importances_, index=model_feats)
                    top   = imp.nlargest(20)
                    bar_clrs = [col_h if 'lsa' not in f else '#7F77DD' for f in top.index]
                    ax.barh(range(20), top.values[::-1], color=bar_clrs[::-1], alpha=0.85, height=0.75)
                    ax.set_yticks(range(20))
                    ax.set_yticklabels(top.index[::-1], fontsize=7.5, color='#e8eaf2')
                    ax.set_title(f'{name}\n(purple=NLP/LSA)', color='#e8eaf2', fontsize=9, fontfamily='monospace')
                    ax.set_xlabel('Importance', fontsize=8, color='#6b7280')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # Linear model coefficients (Ridge only now)
            linear_results = {
                n: r for n, r in results.items()
                if n in ('Ridge',) and hasattr(r['model'], 'coef_')
            }
            if linear_results:
                sec("Linear Model Coefficients — Top 20 by |coef|", "#AB47BC")
                n_lm = len(linear_results)
                fig, axs = dark_fig(1, n_lm, (7 * n_lm, 6))
                axs_list = list(axs) if n_lm > 1 else [axs[0]]
                for ax, (name, res) in zip(axs_list, linear_results.items()):
                    col_h = MODEL_COLORS.get(name, '#888')
                    model_feats = get_model_feature_names(res['model'])
                    if model_feats is None:
                        model_feats = [
                            c for c in pd.read_csv('./data/selected/train_selected.csv').columns
                            if c != TARGET
                        ]
                    coef      = pd.Series(res['model'].coef_, index=model_feats)
                    top20     = coef.abs().nlargest(20)
                    top20_vals = coef[top20.index]
                    bar_clrs  = [col_h if v > 0 else '#888780' for v in top20_vals.values[::-1]]
                    ax.barh(range(20), top20_vals.values[::-1], color=bar_clrs, alpha=0.85, height=0.75)
                    ax.set_yticks(range(20))
                    ax.set_yticklabels(top20.index[::-1], fontsize=7.5, color='#e8eaf2')
                    ax.axvline(0, color='#e8eaf2', lw=0.8, linestyle='--')
                    ax.set_title(f'{name}\n(grey = negative coef)', color='#e8eaf2', fontsize=9, fontfamily='monospace')
                    ax.set_xlabel('Coefficient value', fontsize=8, color='#6b7280')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()


# ───────────────────────────────────────────────────────
# TAB 4 ── PREDICT NEW GAME
# ───────────────────────────────────────────────────────
with tab_predict:
    saved_pred = {n: n for n in MODEL_FILES if model_exists(n)}

    if not saved_pred:
        st.warning("⚠️  No saved models found. Train at least one model first.")
    elif not data_ready():
        st.warning("⚠️  Feature selection data not found.")
    else:
        sel_model = st.selectbox(
            "Select model for prediction",
            list(saved_pred.keys()),
            key="pred_model_sel"
        )

        pred_model       = load_model(sel_model)
        model_feat_names = get_model_feature_names(pred_model)

        current_feat_cols = [
            c for c in pd.read_csv('./data/selected/train_selected.csv').columns
            if c != TARGET
        ]
        feat_cols = model_feat_names if model_feat_names is not None else current_feat_cols

        train_ref_df = pd.read_csv('./data/selected/train_selected.csv')
        train_ref_df = train_ref_df[[c for c in train_ref_df.columns if c != TARGET]]
        col_stats    = train_ref_df.describe().T

        st.markdown(
            "<div class='predict-box'>"
            "<div style='font-family:monospace;font-size:13px;color:#5b8df6;"
            "font-weight:700;margin-bottom:16px'>⌨  INPUT GAME FEATURES</div>",
            unsafe_allow_html=True
        )

        if model_feat_names is not None and set(model_feat_names) != set(current_feat_cols):
            st.info(
                f"ℹ️  This model was trained on **{len(model_feat_names)} features** "
                f"which differ from the current selected set ({len(current_feat_cols)} features). "
                f"Re-training will sync them."
            )

        st.markdown("---")

        numeric_feats = [f for f in feat_cols if 'lsa' not in f.lower()]
        lsa_feats     = [f for f in feat_cols if 'lsa'     in f.lower()]
        input_vals    = {}

        if numeric_feats:
            st.markdown(
                "<div style='font-size:11px;color:#f6c85b;font-family:monospace;"
                "margin-bottom:10px'>NUMERIC FEATURES</div>",
                unsafe_allow_html=True
            )
            n_cols = 3
            rows   = [numeric_feats[i:i + n_cols] for i in range(0, len(numeric_feats), n_cols)]
            for row in rows:
                cols = st.columns(len(row))
                for col_w, feat in zip(cols, row):
                    if feat in col_stats.index:
                        mean_v = float(col_stats.loc[feat, 'mean'])
                        min_v  = float(col_stats.loc[feat, 'min'])
                        max_v  = float(col_stats.loc[feat, 'max'])
                    else:
                        mean_v, min_v, max_v = 0.0, 0.0, 1e6
                    input_vals[feat] = col_w.number_input(
                        feat, value=round(mean_v, 4),
                        min_value=round(min_v, 4),
                        max_value=round(max_v, 4),
                        key=f"inp_{feat}", format="%.4f"
                    )

        if lsa_feats:
            with st.expander(
                f"LSA / NLP features ({len(lsa_feats)} components) — defaults to 0.0",
                expanded=False
            ):
                st.markdown(
                    "<div style='font-size:11px;color:#7F77DD;font-family:monospace;"
                    "margin-bottom:8px'>Leave at 0.0 if you don't have NLP embeddings</div>",
                    unsafe_allow_html=True
                )
                lsa_cols_ui = st.columns(5)
                for i, feat in enumerate(lsa_feats):
                    input_vals[feat] = lsa_cols_ui[i % 5].number_input(
                        feat, value=0.0, min_value=-1.0, max_value=1.0,
                        key=f"inp_{feat}", format="%.4f"
                    )

        st.markdown("</div>", unsafe_allow_html=True)

        _, btn_col, _ = st.columns([2, 2, 2])
        with btn_col:
            do_predict = st.button("🎯  PREDICT RECOMMENDATIONS", key="do_predict")

        if do_predict:
            try:
                X_new = pd.DataFrame([input_vals])
                X_new = align_features(pred_model, X_new)

                log_pred  = float(pred_model.predict(X_new)[0])
                orig_pred = float(np.expm1(log_pred))

                train_ref_aligned = align_features(pred_model, train_ref_df)
                train_preds = pred_model.predict(train_ref_aligned)
                pct = float((train_preds < log_pred).mean() * 100)

                st.markdown(
                    f"<div class='predict-result'>"
                    f"<div style='font-size:11px;color:#6b7280;font-family:monospace;"
                    f"margin-bottom:8px'>{sel_model}</div>"
                    f"<div class='val'>{orig_pred:,.0f}</div>"
                    f"<div class='sub'>predicted recommendation count</div>"
                    f"<div style='margin-top:14px;display:flex;justify-content:center;gap:40px'>"
                    f"<div style='text-align:center'>"
                    f"<div style='font-size:20px;font-family:monospace;color:#5b8df6'>{log_pred:.4f}</div>"
                    f"<div style='font-size:10px;color:#6b7280'>log-space value</div>"
                    f"</div>"
                    f"<div style='text-align:center'>"
                    f"<div style='font-size:20px;font-family:monospace;color:#f6c85b'>{pct:.1f}%</div>"
                    f"<div style='font-size:10px;color:#6b7280'>percentile vs train</div>"
                    f"</div>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                st.markdown("---")
                sec("Prediction Context", "#5bf6c8")
                train_orig = np.expm1(train_preds)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Prediction",   f"{orig_pred:,.0f}")
                c2.metric("Train Mean",   f"{float(train_orig.mean()):,.0f}")
                c3.metric("Train Median", f"{float(np.median(train_orig)):,.0f}")
                c4.metric("Percentile",   f"{pct:.1f}%")

                fig, axs = dark_fig(1, 1, (12, 4))
                axs[0].hist(train_preds, bins=60, color='#4C8EDA',
                            alpha=0.7, edgecolor='none', label='Train predictions')
                axs[0].axvline(log_pred, color='#5bf6c8', lw=2.0, linestyle='--',
                               label=f'Your game = {log_pred:.3f}')
                axs[0].set_title('Your game vs training distribution (log-space)',
                                 color='#e8eaf2', fontsize=9, fontfamily='monospace')
                axs[0].set_xlabel('Predicted log(recommendations)', fontsize=8, color='#6b7280')
                axs[0].legend(fontsize=8, framealpha=0.3)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.code(traceback.format_exc())
