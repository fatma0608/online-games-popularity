import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    BaggingRegressor,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform

os.makedirs('./models', exist_ok=True)
os.makedirs('./plots',  exist_ok=True)

TARGET = 'RecommendationCount'

COLORS = {
    'dt'  : '#8D6E63',
    'rf'  : '#1D9E75',
    'bag' : '#4C8EDA',
}


# ══════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA AND ALIGN SPLITS
# ══════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1: Loading Data and Aligning Splits")
print("=" * 65)

train_num     = pd.read_csv('./data/processed/train.csv')
test_num      = pd.read_csv('./data/processed/test.csv')
nlp_train_raw = pd.read_csv('./data/processed/nlp_features_train.csv')
nlp_test_raw  = pd.read_csv('./data/processed/nlp_features_test.csv')

train_num_sorted = train_num.reset_index(drop=True)
nlp_train_sorted = nlp_train_raw.reset_index(drop=True)
test_num_sorted  = test_num.reset_index(drop=True)
nlp_test_sorted  = nlp_test_raw.reset_index(drop=True)

if len(train_num_sorted) != len(nlp_train_sorted):
    raise ValueError(f"Train row mismatch: num={len(train_num_sorted)}, nlp={len(nlp_train_sorted)}")
if len(test_num_sorted) != len(nlp_test_sorted):
    raise ValueError(f"Test row mismatch: num={len(test_num_sorted)}, nlp={len(nlp_test_sorted)}")

print(f"  [OK] Train rows : {len(train_num_sorted)}")
print(f"  [OK] Test  rows : {len(test_num_sorted)}")

def merge_features(num_df, nlp_df):
    merged = pd.concat([num_df.reset_index(drop=True),
                        nlp_df.reset_index(drop=True)], axis=1)
    return merged.loc[:, ~merged.columns.duplicated()]

train_merged = merge_features(train_num_sorted, nlp_train_sorted)
test_merged  = merge_features(test_num_sorted,  nlp_test_sorted)

print(f"  Merged train : {train_merged.shape}")
print(f"  Merged test  : {test_merged.shape}")

os.makedirs('./data/combined', exist_ok=True)
train_merged.to_csv('./data/combined/train_combined.csv', index=False)
test_merged.to_csv('./data/combined/test_combined.csv',   index=False)


# ══════════════════════════════════════════════════════════════════
# STEP 2 — SEPARATE FEATURES AND TARGET
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2: Separating Features and Target  (raw counts)")
print("=" * 65)

X_train = train_merged.drop(columns=[TARGET])
X_test  = test_merged.drop(columns=[TARGET])
y_train = train_merged[TARGET].astype(float)
y_test  = test_merged[TARGET].astype(float)

print(f"  X_train : {X_train.shape}  |  X_test : {X_test.shape}")
print(f"  Target  — mean={y_train.mean():.1f}  median={y_train.median():.0f}  "
      f"max={y_train.max():.0f}  skew={y_train.skew():.2f}")
print(f"  Zeros   — {(y_train==0).sum()} / {len(y_train)} ({(y_train==0).mean()*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE SELECTION  (fit on train only)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3: Feature Selection (fit on train only)")
print("=" * 65)

initial_features = X_train.shape[1]

# 3a — Dominant-value filter
DOMINANT_THRESHOLD = 0.95
dominant_drop = [col for col in X_train.columns
                 if X_train[col].value_counts(normalize=True, dropna=False).max() >= DOMINANT_THRESHOLD]
X_train.drop(columns=dominant_drop, inplace=True)
X_test.drop(columns=dominant_drop,  inplace=True)
print(f"\n  [3a] Dominant-value filter → dropped {len(dominant_drop):3d} | remaining: {X_train.shape[1]}")

# 3b — VarianceThreshold
VAR_THRESHOLD = 0.001
before_var  = X_train.columns.tolist()
var_sel     = VarianceThreshold(threshold=VAR_THRESHOLD)
X_train_arr = var_sel.fit_transform(X_train)
X_test_arr  = var_sel.transform(X_test)
kept_var    = [c for c, k in zip(before_var, var_sel.get_support()) if k]
X_train     = pd.DataFrame(X_train_arr, columns=kept_var)
X_test      = pd.DataFrame(X_test_arr,  columns=kept_var)
print(f"  [3b] VarianceThreshold={VAR_THRESHOLD} → dropped {len(before_var)-len(kept_var):3d} | remaining: {X_train.shape[1]}")

# 3e — RF importance (fixed threshold)
rf_for_sel = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_for_sel.fit(X_train, y_train)
importances   = rf_for_sel.feature_importances_
IMP_THRESHOLD = 0.001
imp_mask      = importances >= IMP_THRESHOLD
selected_cols = X_train.columns[imp_mask].tolist()
dropped_imp   = X_train.columns[~imp_mask].tolist()
X_train = X_train[selected_cols]
X_test  = X_test[selected_cols]

imp_df = (pd.DataFrame({'feature': selected_cols, 'importance': importances[imp_mask]})
            .sort_values('importance', ascending=False))
print(f"\n  [3e] RF importance >= {IMP_THRESHOLD} → dropped {len(dropped_imp):3d} | remaining: {X_train.shape[1]}")
print("\n  Top 20 features by importance:")
print(imp_df.head(20).to_string(index=False))
print(f"\n  Initial: {initial_features}  Final: {X_train.shape[1]}")


# ══════════════════════════════════════════════════════════════════
# STEP 4 — SAVE SELECTED FEATURES
# ══════════════════════════════════════════════════════════════════
os.makedirs('./data/selected', exist_ok=True)
train_save = X_train.copy(); train_save[TARGET] = y_train.values
test_save  = X_test.copy();  test_save[TARGET]  = y_test.values
train_save.to_csv('./data/selected/train_selected.csv', index=False)
test_save.to_csv('./data/selected/test_selected.csv',   index=False)
print(f"\n  Saved → ./data/selected/  (train {train_save.shape}, test {test_save.shape})")


# ══════════════════════════════════════════════════════════════════
# EVALUATION HELPER
# ══════════════════════════════════════════════════════════════════
def evaluate(name, pred_train, pred_test, y_tr, y_te):
    def _m(yt, yp):
        return dict(
            rmse = np.sqrt(mean_squared_error(yt, yp)),
            mae  = mean_absolute_error(yt, yp),
            r2   = r2_score(yt, yp),
        )
    tr = _m(y_tr, pred_train)
    te = _m(y_te, pred_test)
    gap = tr['r2'] - te['r2']

    print(f"\n  [{name}]")
    print(f"    {'Metric':<10} {'Train':>12} {'Test':>12}")
    print(f"    {'-'*36}")
    for k in ['rmse', 'mae', 'r2']:
        print(f"    {k:<10} {tr[k]:>12.4f} {te[k]:>12.4f}")
    flag = "  ← watch (overfit)" if gap > 0.10 else "  ✓ OK"
    print(f"    {'R² gap':<10} {gap:>12.4f}{flag}")

    return dict(name=name, pred_test=pred_test, pred_train=pred_train,
                **{f'test_{k}': v for k, v in te.items()},
                **{f'train_{k}': v for k, v in tr.items()},
                gap=gap)


# ══════════════════════════════════════════════════════════════════
# MODEL 1 — Decision Tree
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("MODEL 1 — Decision Tree  |  RandomizedSearchCV 5-fold")
print("=" * 65)

dt_params = {
    'max_depth':          [3, 4, 5, 6, 8, 10, 15, None],
    'min_samples_split':  randint(2, 40),
    'min_samples_leaf':   randint(1, 30),
    'max_features':       ['sqrt', 'log2', 0.5, 0.7, None],
    'criterion':          ['squared_error', 'friedman_mse', 'absolute_error'],
}
dt_cv = RandomizedSearchCV(
    DecisionTreeRegressor(random_state=42), dt_params,
    n_iter=40, cv=5, scoring='r2', n_jobs=-1, random_state=42, verbose=0,
)
dt_cv.fit(X_train, y_train)
best_dt = dt_cv.best_estimator_
print(f"  Best params : {dt_cv.best_params_}")
print(f"  Best CV R²  : {dt_cv.best_score_:.4f}")

dt_res = evaluate("DecisionTree (Tuned)",
                  best_dt.predict(X_train),
                  best_dt.predict(X_test),
                  y_train, y_test)
joblib.dump(best_dt, './models/decision_tree_tuned.pkl')
print("  Saved → ./models/decision_tree_tuned.pkl")


# ══════════════════════════════════════════════════════════════════
# MODEL 2 — Random Forest
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("MODEL 2 — Random Forest  |  RandomizedSearchCV 5-fold")
print("=" * 65)

rf_params = {
    'n_estimators':      randint(100, 600),
    'max_depth':         [None, 5, 10, 15, 20, 30],
    'min_samples_leaf':  randint(1, 30),
    'min_samples_split': randint(2, 20),
    'max_features':      ['sqrt', 'log2', 0.3, 0.5, 0.7],
}
rf_cv = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1), rf_params,
    n_iter=40, cv=5, scoring='r2', n_jobs=-1, random_state=42, verbose=0,
)
rf_cv.fit(X_train, y_train)
best_rf = rf_cv.best_estimator_
print(f"  Best params : {rf_cv.best_params_}")
print(f"  Best CV R²  : {rf_cv.best_score_:.4f}")

rf_res = evaluate("RandomForest (Tuned)",
                  best_rf.predict(X_train),
                  best_rf.predict(X_test),
                  y_train, y_test)
joblib.dump(best_rf, './models/random_forest_tuned.pkl')
print("  Saved → ./models/random_forest_tuned.pkl")


# ══════════════════════════════════════════════════════════════════
# MODEL 3 — Bagging Regressor
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("MODEL 3 — Bagging Regressor  |  RandomizedSearchCV 5-fold")
print("=" * 65)

bag_params = {
    'n_estimators':       randint(50, 300),
    'max_samples':        uniform(0.5, 0.5),
    'max_features':       uniform(0.5, 0.5),
    'bootstrap':          [True, False],
    'bootstrap_features': [True, False],
}
bag_cv = RandomizedSearchCV(
    BaggingRegressor(
        estimator=DecisionTreeRegressor(random_state=42),
        random_state=42, n_jobs=-1
    ),
    bag_params,
    n_iter=30, cv=5, scoring='r2', n_jobs=-1, random_state=42, verbose=0,
)
bag_cv.fit(X_train, y_train)
best_bag = bag_cv.best_estimator_
print(f"  Best params : {bag_cv.best_params_}")
print(f"  Best CV R²  : {bag_cv.best_score_:.4f}")

bag_res = evaluate("Bagging (Tuned)",
                   best_bag.predict(X_train),
                   best_bag.predict(X_test),
                   y_train, y_test)
joblib.dump(best_bag, './models/bagging_tuned.pkl')
print("  Saved → ./models/bagging_tuned.pkl")


# ══════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════
all_results = [dt_res, rf_res, bag_res]

print("\n" + "=" * 95)
print("FINAL MODEL COMPARISON — TEST SET  (raw RecommendationCount)")
print("=" * 95)
hdr = (f"{'Model':<28} {'RMSE':>12} {'MAE':>12} "
       f"{'R²(test)':>10} {'R²(train)':>10} {'Gap':>8}")
print(hdr)
print("-" * 95)
for m in all_results:
    flag = " !" if m['gap'] > 0.10 else "  "
    print(f"{m['name']:<28} {m['test_rmse']:>12.2f} {m['test_mae']:>12.2f} "
          f"{m['test_r2']:>10.4f} {m['train_r2']:>10.4f} {m['gap']:>7.4f}{flag}")
print("=" * 95)
best = max(all_results, key=lambda x: x['test_r2'])
print(f"\n  Best model : {best['name']}  (test R²={best['test_r2']:.4f})")


# ══════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════
model_names  = [r['name'] for r in all_results]
model_preds  = [r['pred_test'] for r in all_results]

_color_map = {
    'DecisionTree': COLORS['dt'],
    'RandomForest': COLORS['rf'],
    'Bagging':      COLORS['bag'],
}
model_colors = [_color_map.get(n.split()[0], '#888780') for n in model_names]

n_models = len(all_results)
ncols = 3
nrows = (n_models + ncols - 1) // ncols


# Plot 1 — Actual vs Predicted
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(6 * ncols, 5 * nrows),
                         facecolor='#F8F7F4')
axes = axes.flatten()
fig.suptitle("Actual vs Predicted — Test Set (raw RecommendationCount)",
             fontsize=13, fontweight='600', color='#2C2C2A')

for ax, pred, name, color in zip(axes, model_preds, model_names, model_colors):
    ax.set_facecolor('#F0EFE8')
    ax.scatter(y_test, pred, alpha=0.25, s=8, color=color, edgecolors='none')
    lo = min(y_test.min(), pred.min())
    hi = max(y_test.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], color='#2C2C2A', linewidth=1.2,
            linestyle='--', label='Perfect fit')
    ax.set_title(f'{name}\nR²={r2_score(y_test, pred):.4f}',
                 fontsize=9, fontweight='500', color='#2C2C2A')
    ax.set_xlabel('Actual', fontsize=8)
    ax.set_ylabel('Predicted', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

for ax in axes[n_models:]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig('./plots/model_actual_vs_predicted.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("\nSaved: ./plots/model_actual_vs_predicted.png")


# Plot 2 — Residuals vs Predicted
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(6 * ncols, 5 * nrows),
                         facecolor='#F8F7F4')
axes = axes.flatten()
fig.suptitle("Residuals vs Predicted — Test Set",
             fontsize=13, fontweight='600', color='#2C2C2A')

for ax, pred, name, color in zip(axes, model_preds, model_names, model_colors):
    residuals = np.array(y_test) - pred
    ax.set_facecolor('#F0EFE8')
    ax.scatter(pred, residuals, alpha=0.25, s=8, color=color, edgecolors='none')
    ax.axhline(0, color='#2C2C2A', linewidth=1.2, linestyle='--')
    ax.axhline( np.std(residuals), color='#888780', linewidth=0.8, linestyle=':', label='+1 std')
    ax.axhline(-np.std(residuals), color='#888780', linewidth=0.8, linestyle=':', label='-1 std')
    ax.set_title(f'{name}\nstd={np.std(residuals):.2f}',
                 fontsize=9, fontweight='500', color='#2C2C2A')
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('Residual', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

for ax in axes[n_models:]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig('./plots/model_residuals_vs_predicted.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/model_residuals_vs_predicted.png")


# Plot 3 — Residual Distributions
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(6 * ncols, 4 * nrows),
                         facecolor='#F8F7F4')
axes = axes.flatten()
fig.suptitle("Residual Distribution — Test Set",
             fontsize=13, fontweight='600', color='#2C2C2A')

for ax, pred, name, color in zip(axes, model_preds, model_names, model_colors):
    residuals = np.array(y_test) - pred
    ax.set_facecolor('#F0EFE8')
    ax.hist(residuals, bins=60, color=color, alpha=0.85, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='#2C2C2A', linewidth=1.2, linestyle='--', label='Zero')
    ax.axvline(residuals.mean(), color='#E8593C', linewidth=1.0,
               linestyle='-', label=f'Mean={residuals.mean():.1f}')
    ax.set_title(name, fontsize=9, fontweight='500', color='#2C2C2A')
    ax.set_xlabel('Residual', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

for ax in axes[n_models:]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig('./plots/model_residual_distribution.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/model_residual_distribution.png")


# Plot 4 — Model comparison bar chart
metrics_to_plot = ['test_r2', 'test_rmse', 'test_mae']
metric_labels   = ['R² (higher = better)',
                   'RMSE (lower = better)',
                   'MAE  (lower = better)']
short_names = [r['name'].split()[0] for r in all_results]

fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor='#F8F7F4')
fig.suptitle("Model Comparison — Test Set Metrics (raw counts)",
             fontsize=13, fontweight='600', color='#2C2C2A')

for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
    ax.set_facecolor('#F0EFE8')
    vals = [r[metric] for r in all_results]
    bars = ax.bar(short_names, vals, color=model_colors, alpha=0.85,
                  edgecolor='white', linewidth=0.4, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(abs(v) for v in vals) * 0.01,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=8, color='#2C2C2A')
    ax.set_title(label, fontsize=9, fontweight='500', color='#2C2C2A')
    ax.tick_params(labelsize=8, axis='x')
    ax.tick_params(labelsize=8, axis='y')
    ax.set_ylim(0, max(abs(v) for v in vals) * 1.22)

plt.tight_layout()
plt.savefig('./plots/model_comparison_metrics.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/model_comparison_metrics.png")


# Plot 5 — Feature importance (RF only — Bagging has no direct importances)
fig, ax = plt.subplots(figsize=(10, 8), facecolor='#F8F7F4')
ax.set_facecolor('#F0EFE8')
imp   = pd.Series(best_rf.feature_importances_, index=X_train.columns)
top25 = imp.nlargest(25)
bcolors = [COLORS['rf'] if 'lsa' not in f else '#7F77DD' for f in top25.index]
ax.barh(range(25), top25.values[::-1], color=bcolors[::-1], alpha=0.85, height=0.75)
ax.set_yticks(range(25))
ax.set_yticklabels(top25.index[::-1], fontsize=8)
ax.set_title('Random Forest — Top 25 Feature Importances\n(purple = LSA/NLP features)',
             fontsize=10, fontweight='500', color='#2C2C2A')
ax.set_xlabel('Importance', fontsize=9)
ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig('./plots/model_feature_importance.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/model_feature_importance.png")


# Plot 6 — Train vs Test R² overfitting check
fig, ax = plt.subplots(figsize=(10, 5), facecolor='#F8F7F4')
ax.set_facecolor('#F0EFE8')
x = np.arange(n_models)
w = 0.35
train_r2s = [r['train_r2'] for r in all_results]
test_r2s  = [r['test_r2']  for r in all_results]

bars1 = ax.bar(x - w/2, train_r2s, w, label='Train R²',
               color='#4C8EDA', alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + w/2, test_r2s,  w, label='Test R²',
               color='#E8593C', alpha=0.85, edgecolor='white')

for bar, val in zip(list(bars1) + list(bars2), train_r2s + test_r2s):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom',
            fontsize=8, color='#2C2C2A')

ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=10)
ax.set_ylabel('R²', fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_title('Train vs Test R² — Overfitting Check',
             fontsize=10, fontweight='500', color='#2C2C2A')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('./plots/model_overfitting_check.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/model_overfitting_check.png")


# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("OUTPUTS SUMMARY")
print("=" * 65)
print("Models saved:")
for m in ["decision_tree_tuned.pkl", "random_forest_tuned.pkl", "bagging_tuned.pkl"]:
    path = f'./models/{m}'
    if os.path.exists(path):
        print(f"  {path}")

print("\nPlots saved:")
for p in [
    "model_actual_vs_predicted.png",
    "model_residuals_vs_predicted.png",
    "model_residual_distribution.png",
    "model_comparison_metrics.png",
    "model_feature_importance.png",
    "model_overfitting_check.png",
]:
    print(f"  ./plots/{p}")

print("\nNOTE: All metrics are on raw RecommendationCount (no log transform).")
print("NOTE: Models — DecisionTree, RandomForest, Bagging")