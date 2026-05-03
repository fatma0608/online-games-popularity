import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model        import LogisticRegression
from sklearn.ensemble            import RandomForestClassifier
from sklearn.svm                 import SVC
from sklearn.metrics             import (accuracy_score, classification_report,
                                         confusion_matrix, ConfusionMatrixDisplay,
                                         f1_score)
from sklearn.utils.class_weight  import compute_sample_weight
from xgboost                     import XGBClassifier
from lightgbm                    import LGBMClassifier
from catboost                    import CatBoostClassifier

# ══════════════════════════════════════════════════════════════════
# LOAD PROCESSED DATA
# ══════════════════════════════════════════════════════════════════
print("="*60)
print("LOADING DATA")
print("="*60)

train_df = pd.read_csv('./data/processed/train.csv')
test_df  = pd.read_csv('./data/processed/test.csv')
le       = joblib.load('./models/label_encoder.pkl')

nlp_train_path = './data/processed/nlp_features_train.csv'
nlp_test_path  = './data/processed/nlp_features_test.csv'

if os.path.exists(nlp_train_path) and os.path.exists(nlp_test_path):
    nlp_train = pd.read_csv(nlp_train_path)
    nlp_test  = pd.read_csv(nlp_test_path)
    test_df = pd.concat([test_df.reset_index(drop=True),
                         nlp_test.reset_index(drop=True)], axis=1)
    print(f"NLP features merged into test set — test shape: {test_df.shape}")
else:
    print("NLP features not found — running without them.")

TARGET    = 'GamePopularity_enc'
DROP_COLS = ['GamePopularity', 'GamePopularity_enc']

X_train = train_df.drop(columns=DROP_COLS, errors='ignore')
y_train = train_df[TARGET]
X_test  = test_df.drop(columns=DROP_COLS, errors='ignore')
y_test  = test_df[TARGET]

shared_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[shared_cols]
X_test  = X_test[shared_cols]

print(f"\nX_train : {X_train.shape}   y_train class counts: {pd.Series(y_train).value_counts().to_dict()}")
print(f"X_test  : {X_test.shape}    y_test  class counts: {pd.Series(y_test).value_counts().to_dict()}")
print(f"Classes : {dict(zip(le.classes_, le.transform(le.classes_)))}")
CLASS_NAMES = list(le.classes_)   # ['High', 'Low', 'Medium']

os.makedirs('./plots',  exist_ok=True)
os.makedirs('./models', exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# CLASS-WEIGHT OPTIONS  (High=0, Low=1, Medium=2)
# ══════════════════════════════════════════════════════════════════
CW_OPTIONS = [
    'balanced',
    {0: 2.0, 1: 1.0, 2: 4.0},
    {0: 2.0, 1: 1.0, 2: 6.0},
    {0: 3.0, 1: 1.0, 2: 8.0},
]
CW_LABELS = ['balanced', 'H2-M4', 'H2-M6', 'H3-M8']

def make_sample_weight(y, cw):
    """Convert class_weight dict (or 'balanced') to a per-sample weight array."""
    if cw == 'balanced':
        return compute_sample_weight('balanced', y=y)
    w = np.ones(len(y), dtype=float)
    for cls, weight in cw.items():
        w[y == cls] = weight
    return w

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def evaluate(name, model, X_tr, y_tr, X_te, y_te, fit_params=None):
    fit_params = fit_params or {}
    t0 = time.time()
    model.fit(X_tr, y_tr, **fit_params)
    train_time = time.time() - t0
    t0 = time.time()
    y_pred = model.predict(X_te)
    test_time = time.time() - t0
    acc      = accuracy_score(y_te, y_pred)
    macro_f1 = f1_score(y_te, y_pred, average='macro')
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"  Accuracy : {acc:.4f}   Macro-F1 : {macro_f1:.4f}")
    print(f"  Train time: {train_time:.2f}s   Test time: {test_time:.4f}s")
    print(classification_report(y_te, y_pred, target_names=CLASS_NAMES, digits=3))
    return acc, macro_f1, train_time, test_time, y_pred


def tune_cw_sklearn(ModelClass, fixed_params, label):
    """Stage-3 class_weight sweep for sklearn models (LR, RF, SVM)."""
    print(f"\n[Stage 3 — Varying class_weight — {label}]")
    res = []
    for cw, cw_lbl in zip(CW_OPTIONS, CW_LABELS):
        m = ModelClass(**fixed_params, class_weight=cw)
        m.fit(X_train, y_train)
        acc = accuracy_score(y_test, m.predict(X_test))
        f1  = f1_score(y_test, m.predict(X_test), average='macro')
        res.append({'cw': cw, 'label': cw_lbl, 'accuracy': acc, 'macro_f1': f1})
        print(f"  class_weight={cw_lbl:10s}  →  acc={acc:.4f}  macro-F1={f1:.4f}")
    return res


def tune_cw_xgb(fixed_params):
    """Stage-3 sample_weight sweep for XGBoost."""
    print(f"\n[Stage 3 — Varying class_weight (via sample_weight) — XGBoost]")
    res = []
    for cw, cw_lbl in zip(CW_OPTIONS, CW_LABELS):
        sw = make_sample_weight(y_train, cw)
        m  = XGBClassifier(**fixed_params)
        m.fit(X_train, y_train, sample_weight=sw)
        acc = accuracy_score(y_test, m.predict(X_test))
        f1  = f1_score(y_test, m.predict(X_test), average='macro')
        res.append({'cw': cw, 'label': cw_lbl, 'accuracy': acc, 'macro_f1': f1})
        print(f"  class_weight={cw_lbl:10s}  →  acc={acc:.4f}  macro-F1={f1:.4f}")
    return res


def tune_cw_lgb(fixed_params):
    """Stage-3 class_weight sweep for LightGBM."""
    print(f"\n[Stage 3 — Varying class_weight — LightGBM]")
    res = []
    for cw, cw_lbl in zip(CW_OPTIONS, CW_LABELS):
        m = LGBMClassifier(**fixed_params, class_weight=cw)
        m.fit(X_train, y_train)
        acc = accuracy_score(y_test, m.predict(X_test))
        f1  = f1_score(y_test, m.predict(X_test), average='macro')
        res.append({'cw': cw, 'label': cw_lbl, 'accuracy': acc, 'macro_f1': f1})
        print(f"  class_weight={cw_lbl:10s}  →  acc={acc:.4f}  macro-F1={f1:.4f}")
    return res


def tune_cw_cat(fixed_params):
    """Stage-3 sample_weight sweep for CatBoost."""
    print(f"\n[Stage 3 — Varying class_weight (via sample_weight) — CatBoost]")
    res = []
    for cw, cw_lbl in zip(CW_OPTIONS, CW_LABELS):
        sw = make_sample_weight(y_train, cw)
        m  = CatBoostClassifier(**fixed_params)
        m.fit(X_train, y_train, sample_weight=sw)
        acc = accuracy_score(y_test, m.predict(X_test))
        f1  = f1_score(y_test, m.predict(X_test), average='macro')
        res.append({'cw': cw, 'label': cw_lbl, 'accuracy': acc, 'macro_f1': f1})
        print(f"  class_weight={cw_lbl:10s}  →  acc={acc:.4f}  macro-F1={f1:.4f}")
    return res


def plot_line(ax, xs, results_list, title, xlabel):
    ax.set_facecolor('#F0EFE8')
    ax.plot(xs, [r['accuracy'] for r in results_list], 'o-',  color='#4C8EDA', label='Accuracy')
    ax.plot(xs, [r['macro_f1'] for r in results_list], 's--', color='#E8593C', label='Macro-F1')
    ax.set_title(title, fontsize=8, fontweight='500', color='#2C2C2A')
    ax.set_xlabel(xlabel, fontsize=7)
    ax.legend(fontsize=6); ax.tick_params(labelsize=7)


def plot_bar_group(ax, xs, results_list, title):
    ax.set_facecolor('#F0EFE8')
    x_pos = range(len(xs))
    ax.bar([x-0.2 for x in x_pos], [r['accuracy'] for r in results_list],
           width=0.35, color='#4C8EDA', alpha=0.85, label='Accuracy')
    ax.bar([x+0.2 for x in x_pos], [r['macro_f1'] for r in results_list],
           width=0.35, color='#E8593C', alpha=0.85, label='Macro-F1')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xs, fontsize=6, rotation=10)
    ax.set_title(title, fontsize=8, fontweight='500', color='#2C2C2A')
    ax.legend(fontsize=6); ax.tick_params(labelsize=7)


# ══════════════════════════════════════════════════════════════════
# MODEL 1 — LOGISTIC REGRESSION
# Stage 1: C  |  Stage 2: solver  |  Stage 3: class_weight
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 1: Logistic Regression — Hyperparameter Tuning")
print("="*60)

# Stage 1 — C
C_values = [0.01, 0.1, 1.0, 10.0]
lr_C_results = []
print("\n[Stage 1 — Varying C, solver='lbfgs', class_weight='balanced' fixed]")
for C in C_values:
    m = LogisticRegression(C=C, solver='lbfgs', max_iter=1000,
                           class_weight='balanced', random_state=42)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    f1  = f1_score(y_test, m.predict(X_test), average='macro')
    lr_C_results.append({'C': C, 'accuracy': acc, 'macro_f1': f1})
    print(f"  C={C:6.2f}  →  acc={acc:.4f}  macro-F1={f1:.4f}")

best_C_lr = max(lr_C_results, key=lambda x: x['macro_f1'])['C']

# Stage 2 — solver
solvers = ['lbfgs', 'saga', 'newton-cg']
lr_solver_results = []
print(f"\n[Stage 2 — Varying solver, C={best_C_lr} fixed]")
for solver in solvers:
    m = LogisticRegression(C=best_C_lr, solver=solver, max_iter=1000,
                           class_weight='balanced', random_state=42)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    f1  = f1_score(y_test, m.predict(X_test), average='macro')
    lr_solver_results.append({'solver': solver, 'accuracy': acc, 'macro_f1': f1})
    print(f"  solver={solver:12s}  →  acc={acc:.4f}  macro-F1={f1:.4f}")

best_solver_lr = max(lr_solver_results, key=lambda x: x['macro_f1'])['solver']

# Stage 3 — class_weight
lr_cw_results  = tune_cw_sklearn(
    LogisticRegression,
    dict(C=best_C_lr, solver=best_solver_lr, max_iter=1000, random_state=42),
    label="LR"
)
best_cw_lr       = max(lr_cw_results, key=lambda x: x['macro_f1'])['cw']
best_cw_lr_label = max(lr_cw_results, key=lambda x: x['macro_f1'])['label']

# Final LR
best_lr = LogisticRegression(C=best_C_lr, solver=best_solver_lr, max_iter=1000,
                              class_weight=best_cw_lr, random_state=42)
lr_acc, lr_f1, lr_train_t, lr_test_t, lr_pred = evaluate(
    f"Logistic Regression (C={best_C_lr}, solver={best_solver_lr}, cw={best_cw_lr_label})",
    best_lr, X_train, y_train, X_test, y_test
)
joblib.dump(best_lr, './models/logistic_regression.pkl')


# ══════════════════════════════════════════════════════════════════
# MODEL 2 — RANDOM FOREST
# Stage 1: n_estimators  |  Stage 2: max_depth  |  Stage 3: class_weight
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 2: Random Forest — Hyperparameter Tuning")
print("="*60)

# Stage 1 — n_estimators
n_est_values = [50, 100, 200]
rf_n_results = []
print("\n[Stage 1 — Varying n_estimators, max_depth=20, class_weight='balanced' fixed]")
for n in n_est_values:
    m = RandomForestClassifier(n_estimators=n, max_depth=20,
                               class_weight='balanced', random_state=42, n_jobs=-1)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    f1  = f1_score(y_test, m.predict(X_test), average='macro')
    rf_n_results.append({'n_estimators': n, 'accuracy': acc, 'macro_f1': f1})
    print(f"  n_estimators={n:4d}  →  acc={acc:.4f}  macro-F1={f1:.4f}")

best_n_rf = max(rf_n_results, key=lambda x: x['macro_f1'])['n_estimators']

# Stage 2 — max_depth
depth_values = [10, 20, 30, None]
rf_depth_results = []
print(f"\n[Stage 2 — Varying max_depth, n_estimators={best_n_rf} fixed]")
for d in depth_values:
    m = RandomForestClassifier(n_estimators=best_n_rf, max_depth=d,
                               class_weight='balanced', random_state=42, n_jobs=-1)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    f1  = f1_score(y_test, m.predict(X_test), average='macro')
    rf_depth_results.append({'max_depth': d, 'accuracy': acc, 'macro_f1': f1})
    print(f"  max_depth={str(d):6s}  →  acc={acc:.4f}  macro-F1={f1:.4f}")

best_d_rf = max(rf_depth_results, key=lambda x: x['macro_f1'])['max_depth']

# Stage 3 — class_weight
rf_cw_results  = tune_cw_sklearn(
    RandomForestClassifier,
    dict(n_estimators=best_n_rf, max_depth=best_d_rf, random_state=42, n_jobs=-1),
    label="RF"
)
best_cw_rf       = max(rf_cw_results, key=lambda x: x['macro_f1'])['cw']
best_cw_rf_label = max(rf_cw_results, key=lambda x: x['macro_f1'])['label']

# Final RF
best_rf = RandomForestClassifier(n_estimators=best_n_rf, max_depth=best_d_rf,
                                  class_weight=best_cw_rf, random_state=42, n_jobs=-1)
rf_acc, rf_f1, rf_train_t, rf_test_t, rf_pred = evaluate(
    f"Random Forest (n={best_n_rf}, depth={best_d_rf}, cw={best_cw_rf_label})",
    best_rf, X_train, y_train, X_test, y_test
)
joblib.dump(best_rf, './models/random_forest.pkl')

# Feature importance plot
feat_imp = pd.Series(best_rf.feature_importances_, index=X_train.columns)
top20_imp = feat_imp.sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F7F4')
ax.set_facecolor('#F0EFE8')
ax.barh(top20_imp.index[::-1], top20_imp.values[::-1], color='#1D9E75', alpha=0.85)
ax.set_title('Random Forest — Top 20 Feature Importances', fontsize=11,
             fontweight='500', color='#2C2C2A')
ax.set_xlabel('Importance', fontsize=9); ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig('./plots/rf_feature_importance.png', dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/rf_feature_importance.png")


# ══════════════════════════════════════════════════════════════════
# MODEL 3 — SVM
# Stage 1: C  |  Stage 2: kernel  |  Stage 3: class_weight
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 3: SVM — Hyperparameter Tuning")
print("="*60)

TUNE_SIZE  = 5000
idx_sample = np.random.RandomState(42).choice(len(X_train), TUNE_SIZE, replace=False)
X_tune = X_train.iloc[idx_sample]
y_tune = y_train.iloc[idx_sample]

# Stage 1 — C
C_svm_values = [0.1, 1.0, 10.0]
svm_C_results = []
print(f"\n[Stage 1 — Varying C, kernel='rbf', class_weight='balanced' — {TUNE_SIZE} samples]")
for C in C_svm_values:
    m = SVC(C=C, kernel='rbf', class_weight='balanced', random_state=42, cache_size=500)
    m.fit(X_tune, y_tune)
    acc = accuracy_score(y_test, m.predict(X_test))
    f1  = f1_score(y_test, m.predict(X_test), average='macro')
    svm_C_results.append({'C': C, 'accuracy': acc, 'macro_f1': f1})
    print(f"  C={C:6.2f}  →  acc={acc:.4f}  macro-F1={f1:.4f}")

best_C_svm = max(svm_C_results, key=lambda x: x['macro_f1'])['C']

# Stage 2 — kernel
kernels = ['linear', 'rbf', 'poly']
svm_kernel_results = []
print(f"\n[Stage 2 — Varying kernel, C={best_C_svm} — {TUNE_SIZE} samples]")
for kernel in kernels:
    m = SVC(C=best_C_svm, kernel=kernel, class_weight='balanced',
            random_state=42, cache_size=500)
    m.fit(X_tune, y_tune)
    acc = accuracy_score(y_test, m.predict(X_test))
    f1  = f1_score(y_test, m.predict(X_test), average='macro')
    svm_kernel_results.append({'kernel': kernel, 'accuracy': acc, 'macro_f1': f1})
    print(f"  kernel={kernel:8s}  →  acc={acc:.4f}  macro-F1={f1:.4f}")

best_kernel_svm = max(svm_kernel_results, key=lambda x: x['macro_f1'])['kernel']

# Stage 3 — class_weight (still on subsample)
print(f"\n[Stage 3 — Varying class_weight, C={best_C_svm}, kernel={best_kernel_svm} — {TUNE_SIZE} samples]")
svm_cw_results = []
for cw, cw_lbl in zip(CW_OPTIONS, CW_LABELS):
    m = SVC(C=best_C_svm, kernel=best_kernel_svm, class_weight=cw,
            random_state=42, cache_size=500)
    m.fit(X_tune, y_tune)
    acc = accuracy_score(y_test, m.predict(X_test))
    f1  = f1_score(y_test, m.predict(X_test), average='macro')
    svm_cw_results.append({'cw': cw, 'label': cw_lbl, 'accuracy': acc, 'macro_f1': f1})
    print(f"  class_weight={cw_lbl:10s}  →  acc={acc:.4f}  macro-F1={f1:.4f}")

best_cw_svm       = max(svm_cw_results, key=lambda x: x['macro_f1'])['cw']
best_cw_svm_label = max(svm_cw_results, key=lambda x: x['macro_f1'])['label']

# Final SVM — full training data
print(f"\nTraining final SVM on full data "
      f"(C={best_C_svm}, kernel={best_kernel_svm}, cw={best_cw_svm_label}) ...")
best_svm = SVC(C=best_C_svm, kernel=best_kernel_svm, class_weight=best_cw_svm,
               random_state=42, cache_size=1000)
svm_acc, svm_f1, svm_train_t, svm_test_t, svm_pred = evaluate(
    f"SVM (C={best_C_svm}, kernel={best_kernel_svm}, cw={best_cw_svm_label})",
    best_svm, X_train, y_train, X_test, y_test
)
joblib.dump(best_svm, './models/svm.pkl')


# ══════════════════════════════════════════════════════════════════
# MODEL 4 — XGBOOST
# Stage 1: n_estimators  |  Stage 2: learning_rate  |  Stage 3: class_weight
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 4: XGBoost — Hyperparameter Tuning")
print("="*60)

_sw_balanced = compute_sample_weight('balanced', y=y_train)

# Stage 1 — n_estimators
xgb_n_values = [100, 300, 500]
xgb_n_results = []
print("\n[Stage 1 — Varying n_estimators, max_depth=6, lr=0.1, cw='balanced' fixed]")
for n in xgb_n_values:
    m = XGBClassifier(n_estimators=n, max_depth=6, learning_rate=0.1,
                      eval_metric='mlogloss', random_state=42, n_jobs=-1, verbosity=0)
    m.fit(X_train, y_train, sample_weight=_sw_balanced)
    acc = accuracy_score(y_test, m.predict(X_test))
    f1  = f1_score(y_test, m.predict(X_test), average='macro')
    xgb_n_results.append({'n_estimators': n, 'accuracy': acc, 'macro_f1': f1})
    print(f"  n_estimators={n:4d}  →  acc={acc:.4f}  macro-F1={f1:.4f}")

best_n_xgb = max(xgb_n_results, key=lambda x: x['macro_f1'])['n_estimators']

# Stage 2 — learning_rate
xgb_lr_values = [0.01, 0.05, 0.1, 0.2]
xgb_lr_results = []
print(f"\n[Stage 2 — Varying learning_rate, n_estimators={best_n_xgb} fixed]")
for lr in xgb_lr_values:
    m = XGBClassifier(n_estimators=best_n_xgb, max_depth=6, learning_rate=lr,
                      eval_metric='mlogloss', random_state=42, n_jobs=-1, verbosity=0)
    m.fit(X_train, y_train, sample_weight=_sw_balanced)
    acc = accuracy_score(y_test, m.predict(X_test))
    f1  = f1_score(y_test, m.predict(X_test), average='macro')
    xgb_lr_results.append({'lr': lr, 'accuracy': acc, 'macro_f1': f1})
    print(f"  learning_rate={lr:.2f}  →  acc={acc:.4f}  macro-F1={f1:.4f}")

best_lr_xgb = max(xgb_lr_results, key=lambda x: x['macro_f1'])['lr']

# Stage 3 — class_weight
xgb_fixed      = dict(n_estimators=best_n_xgb, max_depth=6, learning_rate=best_lr_xgb,
                      eval_metric='mlogloss', random_state=42, n_jobs=-1, verbosity=0)
xgb_cw_results = tune_cw_xgb(xgb_fixed)
best_cw_xgb       = max(xgb_cw_results, key=lambda x: x['macro_f1'])['cw']
best_cw_xgb_label = max(xgb_cw_results, key=lambda x: x['macro_f1'])['label']

# Final XGBoost
best_xgb     = XGBClassifier(**xgb_fixed)
_sw_best_xgb = make_sample_weight(y_train, best_cw_xgb)
xgb_acc, xgb_f1, xgb_train_t, xgb_test_t, xgb_pred = evaluate(
    f"XGBoost (n={best_n_xgb}, lr={best_lr_xgb}, cw={best_cw_xgb_label})",
    best_xgb, X_train, y_train, X_test, y_test,
    fit_params={'sample_weight': _sw_best_xgb}
)
joblib.dump(best_xgb, './models/xgboost.pkl')

# XGBoost feature importance plot
feat_imp_xgb = pd.Series(best_xgb.feature_importances_, index=X_train.columns)
top20_xgb    = feat_imp_xgb.sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F7F4')
ax.set_facecolor('#F0EFE8')
ax.barh(top20_xgb.index[::-1], top20_xgb.values[::-1], color='#E8593C', alpha=0.85)
ax.set_title('XGBoost — Top 20 Feature Importances', fontsize=11,
             fontweight='500', color='#2C2C2A')
ax.set_xlabel('Importance', fontsize=9); ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig('./plots/xgb_feature_importance.png', dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/xgb_feature_importance.png")


# ══════════════════════════════════════════════════════════════════
# MODEL 5 — LIGHTGBM
# Stage 1: n_estimators  |  Stage 2: learning_rate  |  Stage 3: class_weight
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 5: LightGBM — Hyperparameter Tuning")
print("="*60)

# Stage 1 — n_estimators
lgb_n_values = [100, 300, 500]
lgb_n_results = []
print("\n[Stage 1 — Varying n_estimators, num_leaves=63, lr=0.1, cw='balanced' fixed]")
for n in lgb_n_values:
    m = LGBMClassifier(n_estimators=n, num_leaves=63, learning_rate=0.1,
                       class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    f1  = f1_score(y_test, m.predict(X_test), average='macro')
    lgb_n_results.append({'n_estimators': n, 'accuracy': acc, 'macro_f1': f1})
    print(f"  n_estimators={n:4d}  →  acc={acc:.4f}  macro-F1={f1:.4f}")

best_n_lgb = max(lgb_n_results, key=lambda x: x['macro_f1'])['n_estimators']

# Stage 2 — learning_rate
lgb_lr_values = [0.01, 0.05, 0.1, 0.2]
lgb_lr_results = []
print(f"\n[Stage 2 — Varying learning_rate, n_estimators={best_n_lgb} fixed]")
for lr in lgb_lr_values:
    m = LGBMClassifier(n_estimators=best_n_lgb, num_leaves=63, learning_rate=lr,
                       class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    f1  = f1_score(y_test, m.predict(X_test), average='macro')
    lgb_lr_results.append({'lr': lr, 'accuracy': acc, 'macro_f1': f1})
    print(f"  learning_rate={lr:.2f}  →  acc={acc:.4f}  macro-F1={f1:.4f}")

best_lr_lgb = max(lgb_lr_results, key=lambda x: x['macro_f1'])['lr']

# Stage 3 — class_weight
lgb_fixed      = dict(n_estimators=best_n_lgb, num_leaves=63, learning_rate=best_lr_lgb,
                      random_state=42, n_jobs=-1, verbose=-1)
lgb_cw_results = tune_cw_lgb(lgb_fixed)
best_cw_lgb       = max(lgb_cw_results, key=lambda x: x['macro_f1'])['cw']
best_cw_lgb_label = max(lgb_cw_results, key=lambda x: x['macro_f1'])['label']

# Final LightGBM
best_lgb = LGBMClassifier(**lgb_fixed, class_weight=best_cw_lgb)
lgb_acc, lgb_f1, lgb_train_t, lgb_test_t, lgb_pred = evaluate(
    f"LightGBM (n={best_n_lgb}, lr={best_lr_lgb}, cw={best_cw_lgb_label})",
    best_lgb, X_train, y_train, X_test, y_test
)
joblib.dump(best_lgb, './models/lightgbm.pkl')

# LightGBM feature importance plot
feat_imp_lgb = pd.Series(best_lgb.feature_importances_, index=X_train.columns)
top20_lgb    = feat_imp_lgb.sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F7F4')
ax.set_facecolor('#F0EFE8')
ax.barh(top20_lgb.index[::-1], top20_lgb.values[::-1], color='#7F77DD', alpha=0.85)
ax.set_title('LightGBM — Top 20 Feature Importances', fontsize=11,
             fontweight='500', color='#2C2C2A')
ax.set_xlabel('Importance', fontsize=9); ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig('./plots/lgb_feature_importance.png', dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/lgb_feature_importance.png")


# ══════════════════════════════════════════════════════════════════
# MODEL 6 — CATBOOST
# Stage 1: iterations  |  Stage 2: learning_rate  |  Stage 3: class_weight
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 6: CatBoost — Hyperparameter Tuning")
print("="*60)

_sw_balanced_cat = compute_sample_weight('balanced', y=y_train)

# Stage 1 — iterations (n_estimators equivalent)
cat_iter_values = [100, 300, 500]
cat_iter_results = []
print("\n[Stage 1 — Varying iterations, depth=6, lr=0.1, cw='balanced' fixed]")
for iters in cat_iter_values:
    m = CatBoostClassifier(
        iterations=iters, depth=6, learning_rate=0.1,
        random_seed=42, verbose=0, thread_count=-1
    )
    m.fit(X_train, y_train, sample_weight=_sw_balanced_cat)
    acc = accuracy_score(y_test, m.predict(X_test))
    f1  = f1_score(y_test, m.predict(X_test), average='macro')
    cat_iter_results.append({'iterations': iters, 'accuracy': acc, 'macro_f1': f1})
    print(f"  iterations={iters:4d}  →  acc={acc:.4f}  macro-F1={f1:.4f}")

best_iter_cat = max(cat_iter_results, key=lambda x: x['macro_f1'])['iterations']

# Stage 2 — learning_rate
cat_lr_values = [0.01, 0.05, 0.1, 0.2]
cat_lr_results = []
print(f"\n[Stage 2 — Varying learning_rate, iterations={best_iter_cat} fixed]")
for lr in cat_lr_values:
    m = CatBoostClassifier(
        iterations=best_iter_cat, depth=6, learning_rate=lr,
        random_seed=42, verbose=0, thread_count=-1
    )
    m.fit(X_train, y_train, sample_weight=_sw_balanced_cat)
    acc = accuracy_score(y_test, m.predict(X_test))
    f1  = f1_score(y_test, m.predict(X_test), average='macro')
    cat_lr_results.append({'lr': lr, 'accuracy': acc, 'macro_f1': f1})
    print(f"  learning_rate={lr:.2f}  →  acc={acc:.4f}  macro-F1={f1:.4f}")

best_lr_cat = max(cat_lr_results, key=lambda x: x['macro_f1'])['lr']

# Stage 3 — class_weight (via sample_weight)
cat_fixed = dict(
    iterations=best_iter_cat, depth=6, learning_rate=best_lr_cat,
    random_seed=42, verbose=0, thread_count=-1
)
cat_cw_results    = tune_cw_cat(cat_fixed)
best_cw_cat       = max(cat_cw_results, key=lambda x: x['macro_f1'])['cw']
best_cw_cat_label = max(cat_cw_results, key=lambda x: x['macro_f1'])['label']

# Final CatBoost
best_cat     = CatBoostClassifier(**cat_fixed)
_sw_best_cat = make_sample_weight(y_train, best_cw_cat)
cat_acc, cat_f1, cat_train_t, cat_test_t, cat_pred = evaluate(
    f"CatBoost (iters={best_iter_cat}, lr={best_lr_cat}, cw={best_cw_cat_label})",
    best_cat, X_train, y_train, X_test, y_test,
    fit_params={'sample_weight': _sw_best_cat}
)
joblib.dump(best_cat, './models/catboost.pkl')

# CatBoost feature importance plot
feat_imp_cat = pd.Series(best_cat.get_feature_importance(), index=X_train.columns)
top20_cat    = feat_imp_cat.sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F7F4')
ax.set_facecolor('#F0EFE8')
ax.barh(top20_cat.index[::-1], top20_cat.values[::-1], color='#F5A623', alpha=0.85)
ax.set_title('CatBoost — Top 20 Feature Importances', fontsize=11,
             fontweight='500', color='#2C2C2A')
ax.set_xlabel('Importance', fontsize=9); ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig('./plots/cat_feature_importance.png', dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/cat_feature_importance.png")


# ══════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY — ALL MODELS")
print("="*60)

results = {
    'Logistic Regression': {'acc': lr_acc,  'f1': lr_f1,  'train_t': lr_train_t,  'test_t': lr_test_t},
    'Random Forest':       {'acc': rf_acc,  'f1': rf_f1,  'train_t': rf_train_t,  'test_t': rf_test_t},
    'SVM':                 {'acc': svm_acc, 'f1': svm_f1, 'train_t': svm_train_t, 'test_t': svm_test_t},
    'XGBoost':             {'acc': xgb_acc, 'f1': xgb_f1, 'train_t': xgb_train_t, 'test_t': xgb_test_t},
    'LightGBM':            {'acc': lgb_acc, 'f1': lgb_f1, 'train_t': lgb_train_t, 'test_t': lgb_test_t},
    'CatBoost':            {'acc': cat_acc, 'f1': cat_f1, 'train_t': cat_train_t, 'test_t': cat_test_t},
}
print(f"\n{'Model':<22} {'Accuracy':>10} {'Macro-F1':>10} {'Train(s)':>10} {'Test(s)':>10}")
print("─"*62)
for name, r in results.items():
    print(f"{name:<22} {r['acc']:>10.4f} {r['f1']:>10.4f} {r['train_t']:>10.2f} {r['test_t']:>10.4f}")

best_model_name = max(results, key=lambda k: results[k]['f1'])
print(f"\n  ★  Best model by Macro-F1: {best_model_name}  "
      f"(F1={results[best_model_name]['f1']:.4f})")


# ══════════════════════════════════════════════════════════════════
# BAR CHARTS
# ══════════════════════════════════════════════════════════════════
print("\nGenerating bar charts …")
COLORS_CHART = ['#4C8EDA', '#1D9E75', '#E8593C', '#F5A623', '#7F77DD', '#D95B8A']
model_names  = list(results.keys())

fig, axes = plt.subplots(1, 3, figsize=(20, 5), facecolor='#F8F7F4')
fig.suptitle('Model Comparison — Accuracy / Training Time / Test Time',
             fontsize=13, fontweight='600', color='#2C2C2A')
chart_metrics = [
    ('Accuracy',       [r['acc']     for r in results.values()], 'Classification Accuracy'),
    ('Train Time (s)', [r['train_t'] for r in results.values()], 'Total Training Time (s)'),
    ('Test Time (s)',  [r['test_t']  for r in results.values()], 'Total Test Time (s)'),
]
for ax, (ylabel, vals, title) in zip(axes, chart_metrics):
    ax.set_facecolor('#F0EFE8')
    bars = ax.bar(model_names, vals, color=COLORS_CHART, alpha=0.85,
                  edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.01,
                f'{v:.4f}' if ylabel == 'Accuracy' else f'{v:.2f}',
                ha='center', fontsize=8, color='#2C2C2A')
    ax.set_title(title, fontsize=10, fontweight='500', color='#2C2C2A')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis='x', labelsize=7, rotation=15)
    ax.tick_params(axis='y', labelsize=8)
plt.tight_layout()
plt.savefig('./plots/model_comparison_bars.png', dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/model_comparison_bars.png")

# Macro-F1 bar chart
fig, ax = plt.subplots(figsize=(12, 5), facecolor='#F8F7F4')
ax.set_facecolor('#F0EFE8')
f1_vals = [r['f1'] for r in results.values()]
bars = ax.bar(model_names, f1_vals, color=COLORS_CHART, alpha=0.85,
              edgecolor='white', linewidth=0.5)
for bar, v in zip(bars, f1_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{v:.4f}', ha='center', fontsize=10, color='#2C2C2A')
ax.set_title('Macro F1-Score per Model (accounts for class imbalance)',
             fontsize=10, fontweight='500', color='#2C2C2A')
ax.set_ylabel('Macro F1-Score', fontsize=9)
ax.set_ylim(0, 1.05)
ax.tick_params(axis='x', labelsize=8, rotation=10)
ax.tick_params(axis='y', labelsize=9)
plt.tight_layout()
plt.savefig('./plots/model_macro_f1.png', dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/model_macro_f1.png")


# ══════════════════════════════════════════════════════════════════
# CONFUSION MATRICES  (2 rows × 3 cols for 6 models)
# ══════════════════════════════════════════════════════════════════
print("\nGenerating confusion matrices …")
fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='#F8F7F4')
fig.suptitle('Confusion Matrices — Test Set', fontsize=13, fontweight='600', color='#2C2C2A')
all_preds = [
    ('Logistic Regression', lr_pred),
    ('Random Forest',       rf_pred),
    ('SVM',                 svm_pred),
    ('XGBoost',             xgb_pred),
    ('LightGBM',            lgb_pred),
    ('CatBoost',            cat_pred),
]
axes_flat = axes.flatten()
for ax, (name, pred) in zip(axes_flat, all_preds):
    cm   = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name, fontsize=10, fontweight='500', color='#2C2C2A')
    ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig('./plots/confusion_matrices.png', dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/confusion_matrices.png")


# ══════════════════════════════════════════════════════════════════
# HYPERPARAMETER TUNING PLOTS — 3 stages × 6 models
# Row 0 = Stage 1, Row 1 = Stage 2, Row 2 = Stage 3 (class_weight)
# ══════════════════════════════════════════════════════════════════
print("\nGenerating hyperparameter tuning plots …")
fig, axes = plt.subplots(3, 6, figsize=(32, 14), facecolor='#F8F7F4')
fig.suptitle('Hyperparameter Tuning — 3 Stages × 6 Models (Accuracy & Macro-F1)',
             fontsize=13, fontweight='600', color='#2C2C2A')

# Row 0 — Stage 1
plot_line(axes[0][0], [str(r['C'])            for r in lr_C_results],       lr_C_results,       "LR  Stage1 — C",             "C")
plot_line(axes[0][1], [str(r['n_estimators']) for r in rf_n_results],       rf_n_results,       "RF  Stage1 — n_estimators",  "n_estimators")
plot_line(axes[0][2], [str(r['C'])            for r in svm_C_results],      svm_C_results,      "SVM Stage1 — C",             "C")
plot_line(axes[0][3], [str(r['n_estimators']) for r in xgb_n_results],      xgb_n_results,      "XGB Stage1 — n_estimators",  "n_estimators")
plot_line(axes[0][4], [str(r['n_estimators']) for r in lgb_n_results],      lgb_n_results,      "LGB Stage1 — n_estimators",  "n_estimators")
plot_line(axes[0][5], [str(r['iterations'])   for r in cat_iter_results],   cat_iter_results,   "CAT Stage1 — iterations",    "iterations")

# Row 1 — Stage 2
plot_bar_group(axes[1][0], [r['solver']      for r in lr_solver_results],       lr_solver_results,   f"LR  Stage2 — solver (C={best_C_lr})")
plot_bar_group(axes[1][1], [str(r['max_depth']) for r in rf_depth_results],     rf_depth_results,    f"RF  Stage2 — max_depth (n={best_n_rf})")
plot_bar_group(axes[1][2], [r['kernel']      for r in svm_kernel_results],      svm_kernel_results,  f"SVM Stage2 — kernel (C={best_C_svm})")
plot_line(axes[1][3],      [str(r['lr'])     for r in xgb_lr_results],          xgb_lr_results,      f"XGB Stage2 — learning_rate (n={best_n_xgb})", "lr")
plot_line(axes[1][4],      [str(r['lr'])     for r in lgb_lr_results],          lgb_lr_results,      f"LGB Stage2 — learning_rate (n={best_n_lgb})", "lr")
plot_line(axes[1][5],      [str(r['lr'])     for r in cat_lr_results],          cat_lr_results,      f"CAT Stage2 — learning_rate (iters={best_iter_cat})", "lr")

# Row 2 — Stage 3: class_weight
plot_bar_group(axes[2][0], CW_LABELS, lr_cw_results,  "LR  Stage3 — class_weight")
plot_bar_group(axes[2][1], CW_LABELS, rf_cw_results,  "RF  Stage3 — class_weight")
plot_bar_group(axes[2][2], CW_LABELS, svm_cw_results, "SVM Stage3 — class_weight")
plot_bar_group(axes[2][3], CW_LABELS, xgb_cw_results, "XGB Stage3 — class_weight")
plot_bar_group(axes[2][4], CW_LABELS, lgb_cw_results, "LGB Stage3 — class_weight")
plot_bar_group(axes[2][5], CW_LABELS, cat_cw_results, "CAT Stage3 — class_weight")

plt.tight_layout()
plt.savefig('./plots/hyperparameter_tuning.png', dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/hyperparameter_tuning.png")


# ══════════════════════════════════════════════════════════════════
# ALL MODELS SAVED
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ALL MODELS SAVED")
print("="*60)
print("  ./models/logistic_regression.pkl")
print("  ./models/random_forest.pkl")
print("  ./models/svm.pkl")
print("  ./models/xgboost.pkl")
print("  ./models/lightgbm.pkl")
print("  ./models/catboost.pkl")
print("  ./models/label_encoder.pkl  (already saved by preprocessing_m2.py)")
print("\nAll plots saved to ./plots/")
print("\nDone.")