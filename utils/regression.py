import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


# STEP 1 — LOAD AND MERGE NUMERICAL + NLP FEATURES

print("=" * 60)
print("STEP 1: Loading and Merging Numerical + NLP Features")
print("=" * 60)

TARGET = 'target_log'

train_num = pd.read_csv('./data/processed/train.csv')
test_num  = pd.read_csv('./data/processed/test.csv')

train_nlp = pd.read_csv('./data/processed/nlp_features_train.csv')
test_nlp  = pd.read_csv('./data/processed/nlp_features_test.csv')

def merge_split(num_df, nlp_df):
    num_df = num_df.reset_index(drop=True)
    nlp_df = nlp_df.reset_index(drop=True)
    assert len(num_df) == len(nlp_df), \
        f"Row count mismatch: num={len(num_df)}, nlp={len(nlp_df)}"
    merged = pd.concat([num_df, nlp_df], axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged

train_merged = merge_split(train_num, train_nlp)
test_merged  = merge_split(test_num,  test_nlp)

print(f"  Merged train : {train_merged.shape}")
print(f"  Merged test  : {test_merged.shape}")

import os
os.makedirs('./data/combined', exist_ok=True)
train_merged.to_csv('./data/combined/train_combined.csv', index=False)
test_merged.to_csv('./data/combined/test_combined.csv',   index=False)
print("\n  Saved combined files to ./data/combined/")

# STEP 2 — SEPARATE FEATURES AND TARGET
print("\n" + "=" * 60)
print("STEP 2: Separating Features and Target")
print("=" * 60)

X_train = train_merged.drop(columns=[TARGET])
y_train = train_merged[TARGET]

X_test  = test_merged.drop(columns=[TARGET])
y_test  = test_merged[TARGET]

print(f"  X_train : {X_train.shape}  |  y_train : {y_train.shape}")
print(f"  X_test  : {X_test.shape}   |  y_test  : {y_test.shape}")

# STEP 3 — FEATURE SELECTION (fit on train only)
print("\n" + "=" * 60)
print("STEP 3: Feature Selection (fit on train only)")
print("=" * 60)

initial_features = X_train.shape[1]

# ── 3a: Dominant-value filter 
dominant_threshold = 0.95
dominant_drop = []
for col in X_train.columns:
    dominant_ratio = X_train[col].value_counts(normalize=True, dropna=False).max()
    if dominant_ratio >= dominant_threshold:
        dominant_drop.append(col)

X_train.drop(columns=dominant_drop, inplace=True)
X_test.drop(columns=dominant_drop,  inplace=True)
print(f"\n  [3a] Dominant-value filter  → dropped {len(dominant_drop):3d} cols | "
      f"remaining: {X_train.shape[1]}")
if dominant_drop:
    print(f"       Dropped: {dominant_drop}")

# ── 3b: VarianceThreshold 
VAR_THRESHOLD = 0.01
feature_names_before_var = X_train.columns.tolist()

var_selector = VarianceThreshold(threshold=VAR_THRESHOLD)
X_train_arr  = var_selector.fit_transform(X_train)
X_test_arr   = var_selector.transform(X_test)

selected_mask = var_selector.get_support()
selected_cols = [c for c, keep in zip(feature_names_before_var, selected_mask) if keep]
var_dropped   = [c for c, keep in zip(feature_names_before_var, selected_mask) if not keep]

X_train = pd.DataFrame(X_train_arr, columns=selected_cols)
X_test  = pd.DataFrame(X_test_arr,  columns=selected_cols)

print(f"\n  [3b] VarianceThreshold={VAR_THRESHOLD} → dropped {len(var_dropped):3d} cols | "
      f"remaining: {X_train.shape[1]}")
if var_dropped:
    print(f"       Dropped: {var_dropped[:10]}{'...' if len(var_dropped) > 10 else ''}")

# ── 3c: High inter-feature correlation 
CORR_THRESHOLD = 0.90

corr_matrix = X_train.corr().abs()
upper_tri   = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

target_corr = X_train.corrwith(y_train).abs()
corr_drop   = []
for col in upper_tri.columns:
    partners = upper_tri.index[upper_tri[col] > CORR_THRESHOLD].tolist()
    for partner in partners:
        if col not in corr_drop and partner not in corr_drop:
            if target_corr.get(col, 0) < target_corr.get(partner, 0):
                corr_drop.append(col)
            else:
                corr_drop.append(partner)

X_train.drop(columns=corr_drop, inplace=True, errors='ignore')
X_test.drop(columns=corr_drop,  inplace=True, errors='ignore')

print(f"\n  [3c] High-corr filter (|r|>{CORR_THRESHOLD}) → dropped {len(corr_drop):3d} cols | "
      f"remaining: {X_train.shape[1]}")
if corr_drop:
    print(f"       Dropped: {corr_drop}")

# ── 3d: Low target-correlation filter 
LOW_CORR_THRESHOLD = 0.01

target_corr_final = X_train.corrwith(y_train).abs()
low_corr_drop     = target_corr_final[target_corr_final < LOW_CORR_THRESHOLD].index.tolist()

X_train.drop(columns=low_corr_drop, inplace=True)
X_test.drop(columns=low_corr_drop,  inplace=True)

print(f"\n  [3d] Low target-corr (<{LOW_CORR_THRESHOLD}) → dropped {len(low_corr_drop):3d} cols | "
      f"remaining: {X_train.shape[1]}")
if low_corr_drop:
    print(f"       Dropped: {low_corr_drop[:10]}{'...' if len(low_corr_drop) > 10 else ''}")

# ── 3e: RandomForest-based selection 
from sklearn.feature_selection import SelectFromModel

rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)
selector    = SelectFromModel(rf_selector, threshold="median")
selector.fit(X_train, y_train)

mask          = selector.get_support()
selected_cols = X_train.columns[mask]

X_train_sel = selector.transform(X_train)
X_test_sel  = selector.transform(X_test)

# Fit again to get importances for display
rf_selector.fit(X_train, y_train)
importances   = rf_selector.feature_importances_
importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
importance_df = importance_df[importance_df['feature'].isin(selected_cols)]
importance_df = importance_df.sort_values('importance', ascending=False)

print("\nTop 20 features by RandomForest importance:")
print(importance_df.head(20).to_string(index=False))

X_train = pd.DataFrame(X_train_sel, columns=selected_cols)
X_test  = pd.DataFrame(X_test_sel,  columns=selected_cols)

# STEP 4 — FEATURE SELECTION REPORT
print("\n" + "=" * 60)
print("STEP 4: Feature Selection Report")
print("=" * 60)
print(f"\nInitial number of features : {initial_features}")
print(f"Final number of features   : {X_train.shape[1]}")
print(f"Total features removed     : {initial_features - X_train.shape[1]}")

# STEP 5 — SAVE SELECTED FEATURES

print("\n" + "=" * 60)
print("STEP 5: Saving Selected Feature Sets")
print("=" * 60)

os.makedirs('./data/selected', exist_ok=True)

X_train_save = X_train.copy(); X_train_save[TARGET] = y_train.values
X_test_save  = X_test.copy();  X_test_save[TARGET]  = y_test.values

X_train_save.to_csv('./data/selected/train_selected.csv', index=False)
X_test_save.to_csv('./data/selected/test_selected.csv',   index=False)

print(f"  Saved: ./data/selected/train_selected.csv  → {X_train_save.shape}")
print(f"  Saved: ./data/selected/test_selected.csv   → {X_test_save.shape}")
print(f"\n  Total features kept: {X_train.shape[1]} (from {initial_features} initial)")
print(f"  Features dropped  : {initial_features - X_train.shape[1]}")


# MODELS
import joblib

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import loguniform, randint, uniform

os.makedirs('./models', exist_ok=True)
os.makedirs('./plots',  exist_ok=True)

COLORS = {
    'ridge': '#4C8EDA',
    'rf':    '#1D9E75',
    'gb':    '#E8593C',
    'lgbm':  '#F39C12',
    'dt':    '#8D6E63',
}

#  Load selected data 
print("\n" + "=" * 65)
print("LOADING SELECTED DATA")
print("=" * 65)

train_df = pd.read_csv('./data/selected/train_selected.csv')
test_df  = pd.read_csv('./data/selected/test_selected.csv')

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET]

X_trainval = X_train.reset_index(drop=True)
y_trainval = y_train.reset_index(drop=True)

print(f"  Train     : {X_train.shape}")
print(f"  Test      : {X_test.shape}")
print(f"  TrainVal  : {X_trainval.shape}  (same as train — CV splits internally)")

# Evaluation helper 
def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    pred_train = model.predict(X_tr)
    pred_test  = model.predict(X_te)

    def _metrics(y_true, y_pred):
        return dict(
            rmse      = np.sqrt(mean_squared_error(y_true, y_pred)),
            mae       = mean_absolute_error(y_true, y_pred),
            r2        = r2_score(y_true, y_pred),
            rmse_orig = np.sqrt(mean_squared_error(np.expm1(y_true), np.expm1(y_pred))),
            mae_orig  = mean_absolute_error(np.expm1(y_true), np.expm1(y_pred)),
        )

    train_m = _metrics(y_tr,  pred_train)
    test_m  = _metrics(y_te,  pred_test)
    gap     = train_m['r2'] - test_m['r2']

    print(f"\n  [{name}]")
    print(f"    {'Metric':<16} {'Train':>10} {'Test':>10}")
    print(f"    {'-'*38}")
    for k in ['rmse', 'mae', 'r2']:
        print(f"    {k:<16} {train_m[k]:>10.4f} {test_m[k]:>10.4f}")
    print(f"    {'R² gap (overfit)':16} {gap:>10.4f}",
          "  ← watch if > 0.10" if gap > 0.10 else "  ✓ OK")
    print(f"    RMSE orig-space  train={train_m['rmse_orig']:>10,.1f}  "
          f"test={test_m['rmse_orig']:>10,.1f}")

    return dict(name=name, **test_m,
                train_r2=train_m['r2'], gap=gap,
                pred_test=pred_test, pred_train=pred_train)


# MODEL 1 — Ridge

print("\n" + "=" * 65)
print("MODEL 1 — Ridge Regression  |  RandomizedSearchCV (5-fold)")
print("=" * 65)
ridge_param_dist = {
    'alpha'         : loguniform(1e-3, 1e4),
    'fit_intercept' : [True, False],
    'solver'        : ['auto', 'svd', 'cholesky', 'lsqr', 'saga'],
}
ridge_search = RandomizedSearchCV(
    Ridge(), ridge_param_dist,
    n_iter=40, cv=5, scoring='r2', n_jobs=-1, random_state=42, verbose=1,
)
ridge_search.fit(X_trainval, y_trainval)
print(f"\n  Best params : {ridge_search.best_params_}")
print(f"  Best CV R²  : {ridge_search.best_score_:.4f}  "
      f"(std={ridge_search.cv_results_['std_test_score'][ridge_search.best_index_]:.4f})")
best_ridge = ridge_search.best_estimator_
ridge_res  = evaluate("Ridge (Tuned)", best_ridge, X_trainval, y_trainval, X_test, y_test)
joblib.dump(best_ridge, './models/ridge_tuned.pkl')


# MODEL 2 — Decision Tree

print("\n" + "=" * 65)
print("MODEL 2 — Decision Tree  |  RandomizedSearchCV (5-fold)")
print("=" * 65)
dt_param_dist = {
    'max_depth'         : [3, 4, 5, 6, 8, 10, 15, None],
    'min_samples_split' : randint(2, 40),
    'min_samples_leaf'  : randint(1, 30),
    'max_features'      : ['sqrt', 'log2', 0.5, 0.7, None],
    'criterion'         : ['squared_error', 'friedman_mse', 'absolute_error'],
}
dt_search = RandomizedSearchCV(
    DecisionTreeRegressor(random_state=42), dt_param_dist,
    n_iter=40, cv=5, scoring='r2', n_jobs=-1, random_state=42, verbose=1,
)
dt_search.fit(X_trainval, y_trainval)
print(f"\n  Best params : {dt_search.best_params_}")
print(f"  Best CV R²  : {dt_search.best_score_:.4f}  "
      f"(std={dt_search.cv_results_['std_test_score'][dt_search.best_index_]:.4f})")
best_dt = dt_search.best_estimator_
dt_res  = evaluate("DecisionTree (Tuned)", best_dt, X_trainval, y_trainval, X_test, y_test)
joblib.dump(best_dt, './models/decision_tree_tuned.pkl')


# MODEL 3 — Random Forest

print("\n" + "=" * 65)
print("MODEL 3 — Random Forest  |  RandomizedSearchCV (5-fold)")
print("=" * 65)
rf_param_dist = {
    'n_estimators'     : randint(100, 600),
    'max_depth'        : [None, 5, 10, 15, 20, 30],
    'min_samples_leaf' : randint(1, 30),
    'min_samples_split': randint(2, 20),
    'max_features'     : ['sqrt', 'log2', 0.3, 0.5, 0.7],
}
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1), rf_param_dist,
    n_iter=40, cv=5, scoring='r2', n_jobs=-1, random_state=42, verbose=1,
)
rf_search.fit(X_trainval, y_trainval)
print(f"\n  Best params : {rf_search.best_params_}")
print(f"  Best CV R²  : {rf_search.best_score_:.4f}  "
      f"(std={rf_search.cv_results_['std_test_score'][rf_search.best_index_]:.4f})")
best_rf = rf_search.best_estimator_
rf_res  = evaluate("RandomForest (Tuned)", best_rf, X_trainval, y_trainval, X_test, y_test)
joblib.dump(best_rf, './models/random_forest_tuned.pkl')


# MODEL 4 — Gradient Boosting

print("\n" + "=" * 65)
print("MODEL 4 — Gradient Boosting  |  RandomizedSearchCV (5-fold)")
print("=" * 65)
gb_param_dist = {
    'n_estimators'    : randint(100, 600),
    'learning_rate'   : loguniform(0.01, 0.3),
    'max_depth'       : randint(2, 8),
    'subsample'       : uniform(0.6, 0.4),
    'min_samples_leaf': randint(5, 40),
    'max_features'    : ['sqrt', 'log2', 0.5, 0.7, None],
}
gb_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42), gb_param_dist,
    n_iter=40, cv=5, scoring='r2', n_jobs=-1, random_state=42, verbose=1,
)
gb_search.fit(X_trainval, y_trainval)
print(f"\n  Best params : {gb_search.best_params_}")
print(f"  Best CV R²  : {gb_search.best_score_:.4f}  "
      f"(std={gb_search.cv_results_['std_test_score'][gb_search.best_index_]:.4f})")
best_gb = gb_search.best_estimator_
gb_res  = evaluate("GradBoost (Tuned)", best_gb, X_trainval, y_trainval, X_test, y_test)
joblib.dump(best_gb, './models/gradient_boosting_tuned.pkl')



# COMPARISON TABLE
all_results = [ridge_res, dt_res, rf_res, gb_res, lgbm_res]

print("\n" + "=" * 90)
print("FINAL MODEL COMPARISON — TEST SET")
print("=" * 90)
hdr = (f"{'Model':<28} {'RMSE(log)':>10} {'MAE(log)':>10} "
       f"{'R²(test)':>10} {'R²(train)':>10} {'Gap':>8}")
print(hdr)
print("-" * 90)
for m in all_results:
    flag = " !" if m['gap'] > 0.10 else "  "
    print(f"{m['name']:<28} {m['rmse']:>10.4f} {m['mae']:>10.4f} "
          f"{m['r2']:>10.4f} {m['train_r2']:>10.4f} {m['gap']:>7.4f}{flag}")
print("=" * 90)

best_model_info = max(all_results, key=lambda x: x['r2'])
print(f"\n  Best model : {best_model_info['name']}  "
      f"(test R²={best_model_info['r2']:.4f})")


# VISUALIZATIONS
model_names  = ['Ridge', 'DecTree', 'RandForest', 'GradBoost']
model_colors = [
    COLORS['ridge'], COLORS['dt'], COLORS['rf'], COLORS['gb'], COLORS['lgbm'],
]
model_preds = [r['pred_test'] for r in all_results]

#  Plot 1: Actual vs Predicted 
fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor='#F8F7F4')
axes = axes.flatten()
fig.suptitle("Actual vs Predicted — Test Set (log-space)",
             fontsize=13, fontweight='600', color='#2C2C2A', y=1.01)

for ax, pred, name, color in zip(axes, model_preds, model_names, model_colors):
    ax.set_facecolor('#F0EFE8')
    ax.scatter(y_test, pred, alpha=0.25, s=8, color=color, edgecolors='none')
    lo = min(y_test.min(), pred.min())
    hi = max(y_test.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], color='#2C2C2A', linewidth=1.2,
            linestyle='--', label='Perfect fit')
    r2 = r2_score(y_test, pred)
    ax.set_title(f'{name}\nR² = {r2:.4f}', fontsize=9, fontweight='500', color='#2C2C2A')
    ax.set_xlabel('Actual (log)', fontsize=8)
    ax.set_ylabel('Predicted (log)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

# Hide the unused 6th subplot
axes[5].set_visible(False)

plt.tight_layout()
plt.savefig('./plots/model_actual_vs_predicted.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("\nSaved: ./plots/model_actual_vs_predicted.png")

#  Plot 2: Residuals vs Predicted 
fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor='#F8F7F4')
axes = axes.flatten()
fig.suptitle("Residuals vs Predicted — Test Set",
             fontsize=13, fontweight='600', color='#2C2C2A', y=1.01)

for ax, pred, name, color in zip(axes, model_preds, model_names, model_colors):
    residuals = np.array(y_test) - pred
    ax.set_facecolor('#F0EFE8')
    ax.scatter(pred, residuals, alpha=0.25, s=8, color=color, edgecolors='none')
    ax.axhline(0, color='#2C2C2A', linewidth=1.2, linestyle='--')
    ax.axhline( np.std(residuals), color='#888780', linewidth=0.8, linestyle=':', label='+1 std')
    ax.axhline(-np.std(residuals), color='#888780', linewidth=0.8, linestyle=':', label='-1 std')
    ax.set_title(f'{name}\nstd={np.std(residuals):.3f}', fontsize=9, fontweight='500', color='#2C2C2A')
    ax.set_xlabel('Predicted (log)', fontsize=8)
    ax.set_ylabel('Residual', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

axes[5].set_visible(False)

plt.tight_layout()
plt.savefig('./plots/model_residuals_vs_predicted.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/model_residuals_vs_predicted.png")

#  Plot 3: Residual Distribution 
fig, axes = plt.subplots(2, 3, figsize=(18, 9), facecolor='#F8F7F4')
axes = axes.flatten()
fig.suptitle("Residual Distribution — Test Set",
             fontsize=13, fontweight='600', color='#2C2C2A', y=1.01)

for ax, pred, name, color in zip(axes, model_preds, model_names, model_colors):
    residuals = np.array(y_test) - pred
    ax.set_facecolor('#F0EFE8')
    ax.hist(residuals, bins=60, color=color, alpha=0.85, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='#2C2C2A', linewidth=1.2, linestyle='--', label='Zero')
    ax.axvline(residuals.mean(), color='#E8593C', linewidth=1.0,
               linestyle='-', label=f'Mean={residuals.mean():.3f}')
    ax.set_title(name, fontsize=9, fontweight='500', color='#2C2C2A')
    ax.set_xlabel('Residual', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

axes[5].set_visible(False)

plt.tight_layout()
plt.savefig('./plots/model_residual_distribution.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/model_residual_distribution.png")

# Plot 4: Model comparison bar chart 
metrics_to_plot = ['r2', 'rmse', 'mae']
metric_labels   = ['R² (higher = better)', 'RMSE log (lower = better)', 'MAE log (lower = better)']
names_short     = ['Ridge', 'DTree', 'RF', 'GB', 'LGBM']

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#F8F7F4')
fig.suptitle("Model Comparison — Test Set Metrics",
             fontsize=13, fontweight='600', color='#2C2C2A', y=1.01)

for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
    ax.set_facecolor('#F0EFE8')
    vals = [r[metric] for r in all_results]
    bars = ax.bar(names_short, vals, color=model_colors, alpha=0.85,
                  edgecolor='white', linewidth=0.4, width=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=7, color='#2C2C2A', rotation=45)
    ax.set_title(label, fontsize=9, fontweight='500', color='#2C2C2A')
    ax.tick_params(labelsize=7, axis='x', rotation=30)
    ax.tick_params(labelsize=7, axis='y')
    ax.set_ylim(0, max(vals) * 1.20)

plt.tight_layout()
plt.savefig('./plots/model_comparison_metrics.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/model_comparison_metrics.png")

#  Plot 5: Feature importance (tree-based) 
tree_models = [best_rf, best_gb]
tree_names  = ['Random Forest', 'Gradient Boosting']
tree_colors = [COLORS['rf'], COLORS['gb'], COLORS['lgbm']]

fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor='#F8F7F4')
fig.suptitle("Feature Importance — Top 25 Features",
             fontsize=13, fontweight='600', color='#2C2C2A', y=1.01)

for ax, model, name, color in zip(axes, tree_models, tree_names, tree_colors):
    ax.set_facecolor('#F0EFE8')
    imp    = pd.Series(model.feature_importances_, index=X_train.columns)
    top25  = imp.nlargest(25)
    bcolors = [color if 'lsa' not in f else '#7F77DD' for f in top25.index]
    ax.barh(range(25), top25.values[::-1], color=bcolors[::-1], alpha=0.85, height=0.75)
    ax.set_yticks(range(25))
    ax.set_yticklabels(top25.index[::-1], fontsize=7.5)
    ax.set_title(f'{name}\n(purple = LSA/NLP features)',
                 fontsize=9, fontweight='500', color='#2C2C2A')
    ax.set_xlabel('Importance', fontsize=8)
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig('./plots/model_feature_importance.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/model_feature_importance.png")

#  Plot 6: Train vs Test R² overfitting check 
fig, ax = plt.subplots(figsize=(14, 5), facecolor='#F8F7F4')
ax.set_facecolor('#F0EFE8')
x = np.arange(len(all_results))
w = 0.35
train_r2s = [r['train_r2'] for r in all_results]
test_r2s  = [r['r2']       for r in all_results]

bars1 = ax.bar(x - w/2, train_r2s, w, label='Train R²', color='#4C8EDA', alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + w/2, test_r2s,  w, label='Test R²',  color='#E8593C', alpha=0.85, edgecolor='white')

for bar, val in zip(list(bars1) + list(bars2), train_r2s + test_r2s):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom',
            fontsize=7, color='#2C2C2A', rotation=45)

ax.set_xticks(x)
ax.set_xticklabels(names_short, fontsize=9)
ax.set_ylabel('R²', fontsize=9)
ax.set_ylim(0, 1.10)
ax.set_title('Train vs Test R² — Overfitting Check\n(large gap = overfitting)',
             fontsize=10, fontweight='500', color='#2C2C2A')
ax.legend(fontsize=9)
ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig('./plots/model_overfitting_check.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/model_overfitting_check.png")

#  Plot 7: GB learning curve 
print("\nGenerating GB staged prediction curve …")
staged_test  = list(best_gb.staged_predict(X_test))
staged_train = list(best_gb.staged_predict(X_trainval))
rounds = range(1, len(staged_test) + 1)

rmse_test_gb  = [np.sqrt(mean_squared_error(y_test,     p)) for p in staged_test]
rmse_train_gb = [np.sqrt(mean_squared_error(y_trainval, p)) for p in staged_train]
best_round_gb = int(np.argmin(rmse_test_gb)) + 1

fig, ax = plt.subplots(figsize=(10, 4), facecolor='#F8F7F4')
ax.set_facecolor('#F0EFE8')
ax.plot(rounds, rmse_train_gb, color='#4C8EDA', linewidth=1.5, label='Train RMSE', alpha=0.9)
ax.plot(rounds, rmse_test_gb,  color='#E8593C', linewidth=1.5, label='Test RMSE',  alpha=0.9)
ax.axvline(best_round_gb, color='#2C2C2A', linewidth=1, linestyle='--',
           label=f'Best round = {best_round_gb}')
ax.set_xlabel('Boosting round', fontsize=9)
ax.set_ylabel('RMSE (log-space)', fontsize=9)
ax.set_title('Gradient Boosting — Train vs Test RMSE per Round',
             fontsize=10, fontweight='500', color='#2C2C2A')
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig('./plots/model_gb_learning_curve.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/model_gb_learning_curve.png")

#  Plot 9: Linear model coefficients (Ridge only) 
print("\nGenerating Ridge coefficient plot …")
fig, ax = plt.subplots(figsize=(8, 6), facecolor='#F8F7F4')
ax.set_facecolor('#F0EFE8')
coef       = pd.Series(best_ridge.coef_, index=X_train.columns)
top20      = coef.abs().nlargest(20)
top20_vals = coef[top20.index]
bcolors    = [COLORS['ridge'] if v > 0 else '#888780' for v in top20_vals.values[::-1]]
ax.barh(range(20), top20_vals.values[::-1], color=bcolors, alpha=0.85, height=0.75)
ax.set_yticks(range(20))
ax.set_yticklabels(top20.index[::-1], fontsize=7.5)
ax.axvline(0, color='#2C2C2A', linewidth=0.8, linestyle='--')
ax.set_title('Ridge — Top 20 Coefficients by |coef|\n(grey = negative)',
             fontsize=9, fontweight='500', color='#2C2C2A')
ax.set_xlabel('Coefficient value', fontsize=8)
ax.tick_params(labelsize=7)
plt.tight_layout()
plt.savefig('./plots/model_linear_coefficients.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: ./plots/model_linear_coefficients.png")


# FINAL SUMMARY
print("\n" + "=" * 65)
print("OUTPUTS SUMMARY")
print("=" * 65)
print("Models saved:")
for m in ["ridge_tuned.pkl", "decision_tree_tuned.pkl", "random_forest_tuned.pkl",
          "gradient_boosting_tuned.pkl"]:
    print(f"  ./models/{m}")

print("\nPlots saved:")
for p in [
    "model_actual_vs_predicted.png      — scatter: actual vs predicted (2×3 grid)",
    "model_residuals_vs_predicted.png   — residual pattern check (2×3 grid)",
    "model_residual_distribution.png    — residual histogram (2×3 grid)",
    "model_comparison_metrics.png       — R², RMSE, MAE bar chart (5 models)",
    "model_feature_importance.png       — top-25 features (RF, GB, LGBM)",
    "model_overfitting_check.png        — train vs test R² per model",
    "model_gb_learning_curve.png        — GB RMSE over boosting rounds",
    "model_linear_coefficients.png      — Ridge top coefficients",
]:
    print(f"  ./plots/{p}")
