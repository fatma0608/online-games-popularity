import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, os, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import scipy.sparse as sp

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

with st.sidebar:
    st.markdown("<div style='font-family:monospace;font-size:18px;font-weight:700;color:#5b8df6;padding:8px 0 20px'>🎮 SteamML</div>", unsafe_allow_html=True)
    st.page_link("app.py",                      label="⬡  Dashboard")
    st.page_link("pages/preprocessing_page.py", label="⬡  Preprocessing")
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
    <p>Runs both scripts on train_data.csv — results appear live below the run button.</p>
</div>""", unsafe_allow_html=True)

tab_pre, tab_nlp = st.tabs(["🔧  PREPROCESSING", "🔤  NLP / TF-IDF / LSA"])

def dark_fig(nrows=1, ncols=1, figsize=(12, 4)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='#13161e')
    axl = np.array(axes).flatten() if hasattr(axes, 'flatten') else [axes]
    for ax in axl:
        ax.set_facecolor('#1a1e2b')
        ax.tick_params(colors='#6b7280', labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#252a38')
    return fig, (np.array(axes).flatten() if hasattr(axes, 'flatten') else [axes])

def sec(label, color='#5bf6c8'):
    st.markdown(f'<div class="sec"><div class="sec-dot" style="background:{color}"></div>'
                f'<div class="sec-lbl">{label}</div></div>', unsafe_allow_html=True)

def render_log(lines):
    html = "".join(
        f'<div style=\'{"color:#6b7280" if k=="muted" else "color:#f65b8d" if k=="err" else "color:#5bf6c8"}\'>'
        f'{m.replace("<","&lt;").replace(">","&gt;")}</div>'
        for m,k in lines)
    st.markdown(f'<div class="log-box">{html}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# TAB 1  ── PREPROCESSING
# ═══════════════════════════════════════════════════════
with tab_pre:
    cl, cr = st.columns([1,1], gap="large")
    with cl:
        if os.path.exists(RAW_PATH): st.success(f"✓ `{RAW_PATH}` ready")
        else:                         st.error(f"Not found: `{RAW_PATH}`")
        run1 = st.button("▶  RUN PREPROCESSING", key="run1")
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
        L = lambda m,k="ok": log.append((m,k))
        try:
            L("// Preprocessing started ─────────────────")
            df = pd.read_csv(RAW_PATH)
            L(f"  loaded  {df.shape[0]:,} rows × {df.shape[1]} cols")

            hm = df.isnull().mean()
            df.drop(columns=hm[hm>0.5].index, inplace=True)
            L(f"  dropped {len(hm[hm>0.5])} high-missing cols","muted")

            const = [c for c in df.columns if df[c].nunique()==1]
            df.drop(columns=const, inplace=True)
            L(f"  dropped {len(const)} constant cols","muted")

            bool_cols = [c for c in df.columns if df[c].dtype==bool]
            df[bool_cols] = df[bool_cols].astype(int)
            bvar = df[bool_cols].var().sort_values()
            low_var = bvar[bvar<0.001].index
            df.drop(columns=low_var, inplace=True)
            df.drop(columns=['QueryID','ResponseID'], inplace=True, errors='ignore')
            L(f"  bool→int; dropped low-var: {list(low_var)}")

            df['ReleaseDate']  = pd.to_datetime(df['ReleaseDate'], errors='coerce')
            df['release_year'] = df['ReleaseDate'].dt.year.fillna(df['ReleaseDate'].dt.year.median())
            df['release_month']= df['ReleaseDate'].dt.month.fillna(6)
            df['game_age_days']= (pd.Timestamp.today()-df['ReleaseDate']).dt.days
            df['game_age_days']= df['game_age_days'].fillna(df['game_age_days'].median())
            df.drop(columns=['ReleaseDate'], inplace=True)
            L("  date → release_year, release_month, game_age_days")

            df['discount_ratio']     = ((df['PriceInitial']-df['PriceFinal'])/(df['PriceInitial']+1e-9)).clip(0,1)
            df['is_effectively_free']= ((df['PriceInitial']==0)|(df['IsFree']==1)).astype(int)
            L(f"  price — {df['is_effectively_free'].sum():,} free games")

            df['has_metacritic']    = (df['Metacritic']>0).astype(int)
            df['num_languages']     = df['SupportedLanguages'].fillna('').apply(lambda x: len([w for w in x.split() if len(w)>2]))
            df['has_website']       = df['Website'].notna().astype(int)
            df['has_support_email'] = df['SupportEmail'].notna().astype(int)
            df['has_support_url']   = df['SupportURL'].notna().astype(int)
            df['has_legal_notice']  = df['LegalNotice'].fillna('').apply(lambda x: 1 if len(x.strip())>1 else 0)
            df['has_reviews_text']  = df['Reviews'].fillna('').apply(lambda x: 1 if len(x.strip())>5 else 0)
            df['about_length']      = df['AboutText'].fillna('').apply(len)
            df['short_length']      = df['ShortDescrip'].fillna('').apply(len)
            df['detail_length']     = df['DetailedDescrip'].fillna('').apply(len)
            df['has_pc_min_reqs']   = df['PCMinReqsText'].fillna('').apply(lambda x: 1 if len(x.strip())>5 else 0)
            df['has_pc_rec_reqs']   = df['PCRecReqsText'].fillna('').apply(lambda x: 1 if len(x.strip())>5 else 0)
            df['has_linux_min_reqs']= df['LinuxMinReqsText'].fillna('').apply(lambda x: 1 if len(x.strip())>5 else 0)
            df['has_mac_min_reqs']  = df['MacMinReqsText'].fillna('').apply(lambda x: 1 if len(x.strip())>5 else 0)
            df['has_drm']           = df['DRMNotice'].fillna('').apply(lambda x: 1 if len(x.strip())>1 else 0)
            df['has_ext_account']   = df['ExtUserAcctNotice'].fillna('').apply(lambda x: 1 if len(x.strip())>1 else 0)
            L("  flags engineered")

            for c in ['SteamSpyOwners','SteamSpyOwnersVariance','SteamSpyPlayersEstimate','SteamSpyPlayersVariance']:
                df[f'{c}_log'] = np.log1p(df[c])
            df['target_log'] = np.log1p(df['RecommendationCount'])
            L(f"  log transforms done — target mean={df['target_log'].mean():.2f}")

            df['price_per_language']    = df['PriceFinal']/(df['num_languages']+1)
            df['metacritic_x_age']      = df['has_metacritic']*df['game_age_days']
            df['owners_per_achievement']= df['SteamSpyOwners_log']/(df['AchievementCount']+1)
            df['dlc_x_owners']          = np.log1p(df['DLCCount'])*df['SteamSpyOwners_log']
            df['movie_x_owners']        = df['MovieCount']*df['SteamSpyOwners_log']
            L("  interaction features created")

            drop_text = ['QueryName','ResponseName','Website','SupportEmail','SupportURL','LegalNotice',
                         'Reviews','SupportedLanguages','ShortDescrip','DetailedDescrip','DRMNotice',
                         'ExtUserAcctNotice','PriceCurrency','Background','HeaderImage','AboutText',
                         'PCMinReqsText','PCRecReqsText','LinuxMinReqsText','LinuxRecReqsText','MacMinReqsText']
            df.drop(columns=[c for c in drop_text if c in df.columns], inplace=True)
            for c in df.select_dtypes(include='object').columns: df.drop(columns=c, inplace=True)
            L(f"  text dropped → {df.shape[1]} numeric cols remain")

            CONT = [c for c in ['RequiredAge','DemoCount','DeveloperCount','DLCCount','MovieCount','PackageCount',
                'PublisherCount','ScreenshotCount','SteamSpyOwners','SteamSpyOwnersVariance',
                'SteamSpyPlayersEstimate','SteamSpyPlayersVariance','AchievementCount','AchievementHighlightedCount',
                'PriceInitial','PriceFinal','release_year','release_month','game_age_days','num_languages',
                'about_length','short_length','detail_length','SteamSpyOwners_log','SteamSpyOwnersVariance_log',
                'SteamSpyPlayersEstimate_log','SteamSpyPlayersVariance_log',
                'price_per_language','metacritic_x_age','owners_per_achievement','dlc_x_owners','movie_x_owners',
                ] if c in df.columns]

            X = df.drop(columns=['RecommendationCount','target_log'])
            y = df['target_log']
            Xt, X_test, yt, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(Xt, yt, test_size=0.1765, random_state=42)
            L(f"  split → train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

            num_cols = X_train.select_dtypes(include=np.number).columns
            med = X_train[num_cols].median()
            for s in [X_train,X_val,X_test]: s[num_cols] = s[num_cols].fillna(med)
            L(f"  median fill done — NaNs left: {X_train.isnull().sum().sum()}")

            cont_feat_cols = [c for c in CONT if c in X_train.columns]
            NO_IQR = ['RequiredAge','DemoCount','DeveloperCount','DLCCount','PackageCount','PublisherCount']
            X_train_raw = X_train.copy()
            total_clip = 0
            for col in cont_feat_cols:
                if col in NO_IQR:
                    for s in [X_train,X_val,X_test]: s[col] = np.log1p(s[col])
                    continue
                Q1,Q3 = X_train[col].quantile(0.25), X_train[col].quantile(0.75)
                IQR = Q3-Q1
                if IQR==0: continue
                lo,hi = Q1-1.5*IQR, Q3+1.5*IQR
                total_clip += ((X_train[col]<lo)|(X_train[col]>hi)).sum()
                for s in [X_train,X_val,X_test]: s[col] = s[col].clip(lo,hi)
            L(f"  IQR capping — {total_clip:,} values clipped")

            iso = IsolationForest(contamination=0.05, random_state=42)
            mask = iso.fit_predict(X_train[cont_feat_cols])==1
            L(f"  Isolation Forest — kept={mask.sum():,}  removed={(~mask).sum():,}")

            X_train_precap = X_train.copy()
            scaler = StandardScaler()
            X_train[cont_feat_cols] = scaler.fit_transform(X_train[cont_feat_cols])
            X_val[cont_feat_cols]   = scaler.transform(X_val[cont_feat_cols])
            X_test[cont_feat_cols]  = scaler.transform(X_test[cont_feat_cols])
            L(f"  StandardScaler applied to {len(cont_feat_cols)} cols")

            mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
            mi_df = pd.DataFrame({'feature':X_train.columns,'MI':mi_scores}).sort_values('MI',ascending=False)
            L(f"  MI top: {mi_df.iloc[0]['feature']} = {mi_df.iloc[0]['MI']:.4f}")

            Xc = X_train_precap[cont_feat_cols].copy(); Xc['target']=y_train.values
            corr_s = Xc.corr()['target'].drop('target').abs().sort_values(ascending=False)

            os.makedirs('./data/processed', exist_ok=True)
            os.makedirs('./plots', exist_ok=True)
            tr_out = X_train.copy(); tr_out['target_log']=y_train.values
            v_out  = X_val.copy();   v_out['target_log'] =y_val.values
            te_out = X_test.copy();  te_out['target_log']=y_test.values
            # tr_out.to_csv('./data/processed/train.csv', index=False)
            # v_out.to_csv('./data/processed/val.csv',    index=False)
            # te_out.to_csv('./data/processed/test.csv',  index=False)

            L("// Done ✓ ─────────────────────────────────")
            st.session_state.update({"done":True,"X_train":X_train,"X_val":X_val,"X_test":X_test,
                "y_train":y_train,"y_val":y_val,"y_test":y_test,"X_train_raw":X_train_raw,
                "cont_cols":cont_feat_cols,"corr_s":corr_s,"mi_df":mi_df,
                "iso_kept":mask.sum(),"iso_removed":(~mask).sum(),
                "y_raw":np.expm1(y_train.values),"y_log":y_train.values})
        except Exception as e:
            import traceback; L(f"ERROR: {e}","err"); L(traceback.format_exc(),"err")
        render_log(log)

    if st.session_state.get("done"):
        X_train   = st.session_state["X_train"]
        X_raw     = st.session_state["X_train_raw"]
        y_train   = st.session_state["y_train"]
        cont_cols = st.session_state["cont_cols"]
        corr_s    = st.session_state["corr_s"]
        mi_df     = st.session_state["mi_df"]

        st.markdown("---")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Train",   f"{len(X_train):,}")
        c2.metric("Val",     f"{len(st.session_state['X_val']):,}")
        c3.metric("Test",    f"{len(st.session_state['X_test']):,}")
        c4.metric("Features", X_train.shape[1])
        c5,c6,c7,c8 = st.columns(4)
        c5.metric("ISO kept",    f"{st.session_state['iso_kept']:,}")
        c6.metric("ISO removed", f"{st.session_state['iso_removed']:,}")
        c7.metric("Cont. cols",  len(cont_cols))
        c8.metric("Top corr.",   f"{corr_s.iloc[0]:.3f}")

        sec("Target — Before vs After log1p","#f65b8d")
        fig,axs = dark_fig(1,2,(13,4))
        rv,lv = st.session_state["y_raw"], st.session_state["y_log"]
        axs[0].hist(rv.clip(0,np.percentile(rv,98)),bins=70,color='#f65b8d',alpha=0.85,edgecolor='none')
        axs[0].axvline(rv.mean(),color='#5bf6c8',lw=1.5,linestyle='--',label=f'mean={rv.mean():.0f}')
        axs[0].set_title("RecommendationCount (original)",color='#e8eaf2',fontsize=9,fontfamily='monospace')
        axs[0].legend(fontsize=8,framealpha=0.4)
        axs[1].hist(lv,bins=60,color='#f6c85b',alpha=0.85,edgecolor='none')
        axs[1].axvline(lv.mean(),color='#5bf6c8',lw=1.5,linestyle='--',label=f'mean={lv.mean():.2f}')
        axs[1].set_title("log1p(RecommendationCount)",color='#e8eaf2',fontsize=9,fontfamily='monospace')
        axs[1].legend(fontsize=8,framealpha=0.4)
        plt.tight_layout(pad=1.5); st.pyplot(fig,use_container_width=True); plt.close()

        sec("Outlier Capping — Before vs After IQR","#f6c85b")
        opts = [c for c in ['SteamSpyOwners','SteamSpyPlayersEstimate','AchievementCount','PriceInitial','game_age_days','about_length','MovieCount','DLCCount'] if c in X_raw.columns and c in X_train.columns]
        sel_iqr = st.selectbox("Column",opts,key="iqr_sel")
        fig,axs = dark_fig(1,2,(13,4))
        b,a = X_raw[sel_iqr].dropna(), X_train[sel_iqr].dropna()
        axs[0].hist(b.clip(b.quantile(0.01),b.quantile(0.99)),bins=60,color='#4C8EDA',alpha=0.85,edgecolor='none')
        axs[0].axvline(b.mean(),color='#5bf6c8',lw=1.5,linestyle='--',label=f'mean={b.mean():.1f}')
        axs[0].set_title(f"{sel_iqr} — Before",color='#e8eaf2',fontsize=9,fontfamily='monospace')
        axs[0].legend(fontsize=8,framealpha=0.4)
        axs[1].hist(a,bins=60,color='#E8593C',alpha=0.85,edgecolor='none')
        axs[1].axvline(a.mean(),color='#5bf6c8',lw=1.5,linestyle='--',label=f'mean={a.mean():.1f}')
        axs[1].set_title(f"{sel_iqr} — After IQR",color='#e8eaf2',fontsize=9,fontfamily='monospace')
        axs[1].legend(fontsize=8,framealpha=0.4)
        plt.tight_layout(pad=1.5); st.pyplot(fig,use_container_width=True); plt.close()

        sec("Feature Distribution Explorer","#5b8df6")
        avail = [c for c in cont_cols if c in X_train.columns]
        sf = st.selectbox("Feature",avail,key="feat_sel")
        vals = X_train[sf].dropna()
        e1,e2,e3,e4 = st.columns(4)
        e1.metric("Mean",f"{vals.mean():.3f}"); e2.metric("Std",f"{vals.std():.3f}")
        e3.metric("Min",f"{vals.min():.3f}");   e4.metric("Max",f"{vals.max():.3f}")
        fig,axs = dark_fig(1,2,(13,4))
        axs[0].hist(vals,bins=60,color='#5b8df6',alpha=0.85,edgecolor='none')
        axs[0].axvline(vals.mean(),color='#5bf6c8',lw=1.5,linestyle='--',label='mean')
        axs[0].axvline(vals.median(),color='#f6c85b',lw=1.2,linestyle=':',label='median')
        axs[0].legend(fontsize=8,framealpha=0.4)
        axs[0].set_title(f"Distribution · {sf}",color='#e8eaf2',fontsize=9,fontfamily='monospace')
        axs[1].boxplot(vals,vert=False,patch_artist=True,
            medianprops=dict(color='#5bf6c8',lw=2),
            boxprops=dict(facecolor=(0.357,0.553,0.965,0.3),color='#5b8df6'),
            whiskerprops=dict(color='#6b7280'),capprops=dict(color='#6b7280'),
            flierprops=dict(marker='.',color='#6b7280',markersize=3,alpha=0.3))
        axs[1].set_title(f"Box Plot · {sf}",color='#e8eaf2',fontsize=9,fontfamily='monospace')
        plt.tight_layout(pad=1.5); st.pyplot(fig,use_container_width=True); plt.close()

        sec("Feature Correlation with Target","#5bf6c8")
        top15 = corr_s.head(15)
        fig,axs = dark_fig(1,1,(12,4))
        clrs = ['#5bf6c8' if i<3 else '#5b8df6' if i<8 else '#6b7280' for i in range(len(top15))]
        axs[0].barh(range(len(top15)),top15.values[::-1],color=clrs[::-1],alpha=0.85,height=0.65)
        axs[0].set_yticks(range(len(top15))); axs[0].set_yticklabels(top15.index[::-1],fontsize=8,color='#e8eaf2')
        axs[0].set_xlabel('|Pearson|',fontsize=9,color='#6b7280')
        axs[0].set_title('Top 15 Feature Correlations with log(RecommendationCount)',color='#e8eaf2',fontsize=9,fontfamily='monospace')
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

        sec("Mutual Information — Top 20","#7F77DD")
        t20 = mi_df.head(20)
        fig,axs = dark_fig(1,1,(12,4))
        axs[0].barh(range(len(t20)),t20['MI'].values[::-1],color='#7F77DD',alpha=0.85,height=0.65)
        axs[0].set_yticks(range(len(t20))); axs[0].set_yticklabels(t20['feature'].values[::-1],fontsize=8,color='#e8eaf2')
        axs[0].set_xlabel('MI Score',fontsize=9,color='#6b7280')
        axs[0].set_title('Mutual Information with Target',color='#e8eaf2',fontsize=9,fontfamily='monospace')
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

        sec("Post-Scaling Stats (train)","#5b8df6")
        st.dataframe(X_train[avail].describe().T[['mean','std','min','max']].round(4),use_container_width=True,height=280)

# ═══════════════════════════════════════════════════════
# TAB 2  ── NLP
# ═══════════════════════════════════════════════════════
with tab_nlp:
    cl2,cr2 = st.columns([1,1],gap="large")
    with cl2:
        if os.path.exists(RAW_PATH): st.success(f"✓ `{RAW_PATH}` ready")
        else:                         st.error(f"Not found: `{RAW_PATH}`")
        run2 = st.button("▶  RUN NLP PIPELINE", key="run2")
    with cr2:
        st.markdown("""<div style='background:#13161e;border:1px solid #252a38;border-radius:10px;
            padding:16px 18px;font-size:12px;color:#6b7280;line-height:1.9'>
            Raw text stats → Clean (HTML/URL/stopwords/lemmatize) →
            TF-IDF per field (5 fields) → LSA / TruncatedSVD (50 components) →
            NLP-target correlation → Save nlp_features CSVs
        </div>""", unsafe_allow_html=True)

    if run2:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        for pkg in ['stopwords','wordnet','omw-1.4']:
            try: nltk.download(pkg, quiet=True)
            except: pass

        log2 = []; L2 = lambda m,k="ok": log2.append((m,k))
        try:
            L2("// NLP Pipeline started ─────────────────")
            df2 = pd.read_csv(RAW_PATH)
            TC = {'about':'AboutText','short':'ShortDescrip','detail':'DetailedDescrip','reviews':'Reviews','name':'ResponseName'}
            TC = {k:v for k,v in TC.items() if v in df2.columns}
            for col in TC.values(): df2[col] = df2[col].fillna('')
            text_df = df2[list(TC.values())].copy()
            L2(f"  text cols: {list(TC.values())}")

            raw_stats = {}
            for key,col in TC.items():
                vals = text_df[col]
                raw_stats[key] = {'wc': vals.apply(lambda x: len(x.split())),
                                  'has': (vals.str.len()>10).sum()}
            L2(f"  raw stats — AboutText avg words: {raw_stats['about']['wc'].mean():.0f}")

            lemmatizer = WordNetLemmatizer()
            STOP = set(stopwords.words('english'))
            STOP.update({'game','games','play','player','players','feature','features','include','includes',
                         'new','get','also','available','download','update','version','support','use',
                         'using','system','may','will','can'})
            HR = re.compile(r'<[^>]+>'); UR = re.compile(r'http\S+|www\.\S+')
            PR = re.compile(r'[^a-zA-Z\s]'); SR = re.compile(r'\s+')
            def clean(t):
                t = HR.sub(' ',t); t = UR.sub(' ',t); t = t.lower()
                t = PR.sub(' ',t); t = SR.sub(' ',t).strip()
                return ' '.join([lemmatizer.lemmatize(w) for w in t.split() if w not in STOP and len(w)>2])
            cleaned = {}
            for key,col in TC.items():
                cleaned[key] = text_df[col].apply(clean)
                avg = cleaned[key].apply(lambda x: len(x.split())).mean()
                L2(f"  cleaned {col} — avg tokens: {avg:.0f}","muted")

            CFG = {'about': dict(max_features=200,ngram_range=(1,2),min_df=3,max_df=0.95,sublinear_tf=True),
                   'detail':dict(max_features=300,ngram_range=(1,2),min_df=3,max_df=0.95,sublinear_tf=True),
                   'short': dict(max_features=100,ngram_range=(1,2),min_df=3,max_df=0.95,sublinear_tf=True),
                   'reviews':dict(max_features=100,ngram_range=(1,2),min_df=2,max_df=0.95,sublinear_tf=True),
                   'name':  dict(max_features=50, ngram_range=(1,1),min_df=2,max_df=0.90,sublinear_tf=True)}
            CFG = {k:v for k,v in CFG.items() if k in cleaned}
            idx_all = np.arange(len(df2))
            idx_t2,idx_te = train_test_split(idx_all,test_size=0.15,random_state=42)
            idx_tr,idx_v  = train_test_split(idx_t2, test_size=0.1765,random_state=42)

            parts_tr,parts_v,parts_te,vecs,fnames_all = [],[],[],{},[]
            for key,cfg in CFG.items():
                texts = cleaned[key].values
                vec = TfidfVectorizer(**cfg)
                tr=vec.fit_transform(texts[idx_tr]); v=vec.transform(texts[idx_v]); te=vec.transform(texts[idx_te])
                vecs[key]=vec; fnames_all.append(vec.get_feature_names_out())
                parts_tr.append(tr); parts_v.append(v); parts_te.append(te)
                L2(f"  TF-IDF [{key}] vocab={len(vec.vocabulary_):,} shape={tr.shape}","muted")
            tfidf_tr = sp.hstack(parts_tr,format='csr')
            L2(f"  combined TF-IDF: {tfidf_tr.shape}")

            N=50; svd=TruncatedSVD(n_components=N,random_state=42)
            lsa_tr=normalize(svd.fit_transform(tfidf_tr))
            lsa_v =normalize(svd.transform(sp.hstack(parts_v, format='csr')))
            lsa_te=normalize(svd.transform(sp.hstack(parts_te,format='csr')))
            cum_var=svd.explained_variance_ratio_.cumsum()[-1]*100
            L2(f"  LSA {N} components — cum variance: {cum_var:.1f}%")

            lsa_cols=[f'lsa_{i}' for i in range(N)]
            lsa_tr_df=pd.DataFrame(lsa_tr,columns=lsa_cols,index=idx_tr)
            lsa_v_df =pd.DataFrame(lsa_v, columns=lsa_cols,index=idx_v)
            lsa_te_df=pd.DataFrame(lsa_te,columns=lsa_cols,index=idx_te)
            lsa_all=pd.concat([lsa_tr_df,lsa_v_df,lsa_te_df]).sort_index()
            nlp_train=lsa_all.loc[idx_tr].reset_index(drop=True)
            nlp_val  =lsa_all.loc[idx_v].reset_index(drop=True)
            nlp_test =lsa_all.loc[idx_te].reset_index(drop=True)

            tgt = np.log1p(df2['RecommendationCount']).iloc[idx_tr].reset_index(drop=True)
            corrs_nlp = nlp_train.corrwith(tgt).abs().sort_values(ascending=False)
            L2(f"  top NLP corr: {corrs_nlp.index[0]} = {corrs_nlp.iloc[0]:.4f}")

            os.makedirs('./data/processed',exist_ok=True)
            # nlp_train.to_csv('./data/processed/nlp_features_train.csv',index=False)
            # nlp_val.to_csv('./data/processed/nlp_features_val.csv',    index=False)
            # nlp_test.to_csv('./data/processed/nlp_features_test.csv',  index=False)
            L2("// NLP Done ✓ ──────────────────────────")
            st.session_state.update({"nlp_done":True,"raw_stats":raw_stats,"cleaned":cleaned,
                "parts_tr":parts_tr,"vecs":vecs,"fnames_all":fnames_all,"TC":TC,"CFG":CFG,
                "svd":svd,"nlp_train":nlp_train,"nlp_test":nlp_test,"corrs_nlp":corrs_nlp,
                "cum_var":cum_var,"N":N,"all_fn":np.concatenate(fnames_all)})
        except Exception as e:
            import traceback; L2(f"ERROR: {e}","err"); L2(traceback.format_exc(),"err")
        render_log(log2)

    if st.session_state.get("nlp_done"):
        rs    = st.session_state["raw_stats"]
        cl_   = st.session_state["cleaned"]
        TC    = st.session_state["TC"]
        vecs  = st.session_state["vecs"]
        pts   = st.session_state["parts_tr"]
        fna   = st.session_state["fnames_all"]
        CFG   = st.session_state["CFG"]
        svd   = st.session_state["svd"]
        nlp_tr= st.session_state["nlp_train"]
        nlp_te= st.session_state["nlp_test"]
        cn    = st.session_state["corrs_nlp"]
        N     = st.session_state["N"]
        afn   = st.session_state["all_fn"]
        COLORS= ['#4C8EDA','#E8593C','#1D9E75','#7F77DD','#EF9F27']

        st.markdown("---")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Text fields",   len(TC))
        c2.metric("TF-IDF feats",  sum(len(vecs[k].vocabulary_) for k in vecs))
        c3.metric("LSA components",N)
        c4.metric("Cum. variance", f"{st.session_state['cum_var']:.1f}%")

        sec("Raw Word Count — Per Field","#5b8df6")
        fig,axs = dark_fig(1,len(TC),(5*len(TC),4))
        for i,(key,col) in enumerate(TC.items()):
            wc=rs[key]['wc']
            axs[i].hist(wc.clip(0,wc.quantile(0.98)),bins=60,color=COLORS[i%5],alpha=0.85,edgecolor='none')
            axs[i].axvline(wc.mean(),color='#2C2C2A',lw=1.2,linestyle='--',label=f'mean={wc.mean():.0f}')
            axs[i].set_title(col,color='#e8eaf2',fontsize=8,fontfamily='monospace')
            axs[i].legend(fontsize=7,framealpha=0.4)
        plt.suptitle("Raw Text Word Counts",color='#e8eaf2',fontsize=10,fontfamily='monospace',y=1.02)
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

        sec("Text Cleaning — Before vs After","#5bf6c8")
        sel_t = st.selectbox("Field",list(TC.keys()),key="txt_sel",format_func=lambda k: TC[k])
        fig,axs = dark_fig(1,2,(13,4))
        bwc=rs[sel_t]['wc']; awc=cl_[sel_t].apply(lambda x: len(x.split()))
        cap=bwc.quantile(0.98)
        axs[0].hist(bwc.clip(0,cap),bins=60,color='#4C8EDA',alpha=0.85,edgecolor='none')
        axs[0].axvline(bwc.mean(),color='#5bf6c8',lw=1.5,linestyle='--',label=f'mean={bwc.mean():.0f}')
        axs[0].set_title(f"{TC[sel_t]} — Raw",color='#e8eaf2',fontsize=9,fontfamily='monospace')
        axs[0].legend(fontsize=8,framealpha=0.4)
        axs[1].hist(awc.clip(0,cap*0.6),bins=60,color='#E8593C',alpha=0.85,edgecolor='none')
        axs[1].axvline(awc.mean(),color='#5bf6c8',lw=1.5,linestyle='--',label=f'mean={awc.mean():.0f}')
        axs[1].set_title(f"{TC[sel_t]} — Cleaned",color='#e8eaf2',fontsize=9,fontfamily='monospace')
        axs[1].legend(fontsize=8,framealpha=0.4)
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

        sec("TF-IDF — Top Terms","#f6c85b")
        keys_t = list(CFG.keys())
        sel_tf = st.selectbox("Field",keys_t,key="tfidf_sel",format_func=lambda k: TC[k])
        kidx = keys_t.index(sel_tf)
        mat = pts[kidx]; ms = np.asarray(mat.mean(axis=0)).flatten()
        vocab = vecs[sel_tf].get_feature_names_out()
        top40 = ms.argsort()[-40:][::-1]
        fig,axs = dark_fig(1,1,(12,6))
        axs[0].barh(range(40),ms[top40][::-1],color=COLORS[kidx%5],alpha=0.85,height=0.75)
        axs[0].set_yticks(range(40)); axs[0].set_yticklabels(vocab[top40][::-1],fontsize=7,color='#e8eaf2')
        axs[0].set_title(f'Top 40 TF-IDF Terms — {TC[sel_tf]}',color='#e8eaf2',fontsize=9,fontfamily='monospace')
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

        sec("LSA — Explained Variance","#7F77DD")
        fig,axs = dark_fig(1,2,(13,4))
        axs[0].plot(range(1,N+1),svd.explained_variance_ratio_*100,color='#4C8EDA',lw=1.8,marker='o',markersize=3)
        axs[0].fill_between(range(1,N+1),svd.explained_variance_ratio_*100,alpha=0.2,color='#4C8EDA')
        axs[0].set_title('Scree Plot',color='#e8eaf2',fontsize=9,fontfamily='monospace')
        axs[0].set_xlabel('Component',fontsize=8,color='#6b7280'); axs[0].set_ylabel('Variance %',fontsize=8,color='#6b7280')
        cum=svd.explained_variance_ratio_.cumsum()
        axs[1].plot(range(1,N+1),cum*100,color='#E8593C',lw=1.8)
        axs[1].fill_between(range(1,N+1),cum*100,alpha=0.2,color='#E8593C')
        axs[1].axhline(80,color='#6b7280',lw=1,linestyle='--',label='80%')
        axs[1].axhline(90,color='#444441',lw=1,linestyle='--',label='90%')
        axs[1].legend(fontsize=8,framealpha=0.4)
        axs[1].set_title('Cumulative Variance',color='#e8eaf2',fontsize=9,fontfamily='monospace')
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

        sec("LSA — Top Words per Component","#E8593C")
        comp_i = st.slider("Component",1,min(6,N),1,key="comp_sl")-1
        fig,axs = dark_fig(1,1,(12,5))
        loading = svd.components_[comp_i]
        top_idx = np.concatenate([loading.argsort()[-15:][::-1], loading.argsort()[:5]])
        tw,ts = afn[top_idx], loading[top_idx]
        cb = ['#E8593C' if s>0 else '#4C8EDA' for s in ts]
        axs[0].barh(range(len(top_idx)),ts[::-1],color=cb[::-1],alpha=0.85,height=0.75)
        axs[0].set_yticks(range(len(top_idx))); axs[0].set_yticklabels(tw[::-1],fontsize=8,color='#e8eaf2')
        axs[0].axvline(0,color='#6b7280',lw=0.8)
        axs[0].set_title(f'LSA Component {comp_i+1}',color='#e8eaf2',fontsize=9,fontfamily='monospace')
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

        sec("NLP Features — Correlation with Target","#5bf6c8")
        t20n = cn.head(20)
        fig,axs = dark_fig(1,1,(12,4))
        axs[0].barh(range(len(t20n)),t20n.values[::-1],color='#7F77DD',alpha=0.85,height=0.65)
        axs[0].set_yticks(range(len(t20n))); axs[0].set_yticklabels(t20n.index[::-1],fontsize=8,color='#e8eaf2')
        axs[0].set_xlabel('|Pearson|',fontsize=9,color='#6b7280')
        axs[0].set_title('Top 20 NLP Features — Correlation with log(RecommendationCount)',color='#e8eaf2',fontsize=9,fontfamily='monospace')
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()