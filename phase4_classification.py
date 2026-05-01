import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import joblib
import os
import warnings
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore')
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             f1_score)
 
os.makedirs('./models', exist_ok=True)
os.makedirs('./plots', exist_ok=True)
 
# ============================================================
# LOAD DATA
# ============================================================
train_df = pd.read_csv('./data/processed/train_with_nlp.csv')
test_df  = pd.read_csv('./data/processed/test_with_nlp.csv')
 
train_df = train_df.fillna(0)
test_df  = test_df.fillna(0)
 
TARGET = 'GamePopularity_encoded'
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET]
 
le = joblib.load('./models/label_encoder.pkl')
class_names = le.classes_
 
print("=" * 60)
print("PHASE 4: CLASSIFICATION")
print("=" * 60)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Classes: {class_names}")
 
# ── NOTE: NO SMOTE here — already applied in phase2 after scaling ─
# Applying SMOTEENN again on top of SMOTE-balanced data would double-resample
# and corrupt the distribution. The data loaded here is already balanced.
print("\nClass distribution (already balanced from phase2 SMOTE):")
print(pd.Series(y_train).value_counts().sort_index().to_dict())
 
# ============================================================
# MODELS (IMPROVED)
# ============================================================
models = {

    'Decision Tree (Improved)': DecisionTreeClassifier(
        max_depth=20,
        min_samples_split=3,
        class_weight='balanced',
        random_state=42
    ),

    'Random Forest (Improved)': RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=3,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    ),

    'SVM (Improved)': SVC(
        kernel='rbf',
        C=3.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    ),

    'XGBoost': XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    ),

    'LightGBM ': LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=50,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
,
    'KNN': KNeighborsClassifier(
        n_neighbors=7,
        weights='distance'
    ),
 
    'Decision Tree': DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    ),
 
    'Random Forest': RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    ),
 
    'SVM': SVC(
        kernel='rbf',
        C=2.0,
        gamma='scale',
        class_weight='balanced',
        random_state=42
    ),
}
 
# ============================================================
# TRAIN & EVALUATE
# ============================================================
results = {}
 
print("\n" + "="*60)
print("Training Models...")
print("="*60)
 
for name, model in models.items():
    print(f"\n── {name} ──")
 
    # Train
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
 
    # Predict
    t0 = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - t0
 
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro')
 
    results[name] = {
        'accuracy': acc,
        'f1_macro': f1,
        'train_time': train_time,
        'test_time': test_time,
        'y_pred': y_pred,
    }
 
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-macro: {f1:.4f}")
    print(f"  Train time: {train_time:.2f}s")
    print(f"  Test time : {test_time:.2f}s")
 
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
 
    # Save model
    safe_name = name.lower().replace(' ', '_')
    joblib.dump(model, f'./models/model_{safe_name}.pkl')
 
# ============================================================
# SUMMARY (IMPORTANT: USE F1, NOT ACCURACY)
# ============================================================
print("\n" + "="*60)
print("RESULTS SUMMARY (USE F1!)")
print("="*60)
 
summary = pd.DataFrame({
    name: {
        'Accuracy (%)': f"{v['accuracy']*100:.2f}",
        'F1 Macro': f"{v['f1_macro']:.4f}",
        'Train Time (s)': f"{v['train_time']:.2f}",
        'Test Time (s)': f"{v['test_time']:.2f}",
    }
    for name, v in results.items()
}).T
 
print(summary)
 
# ============================================================
# BAR CHART (F1 INCLUDED)
# ============================================================
COLORS = ['#4C8EDA', '#E8593C', '#1D9E75', '#7F77DD']

# ============================================================
# BAR CHARTS — Accuracy, Macro-F1, Train Time, Test Time
# ============================================================
model_names  = list(results.keys())
accuracies   = [results[n]['accuracy']   for n in model_names]
f1_scores    = [results[n]['f1_macro']   for n in model_names]
train_times  = [results[n]['train_time'] for n in model_names]
test_times   = [results[n]['test_time']  for n in model_names]
 
fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='#F8F7F4')
fig.suptitle('Phase 4 — Model Comparison', fontsize=13, fontweight='700', color='#2C2C2A')
 
def styled_bar(ax, names, values, title, ylabel, fmt='{:.3f}'):
    ax.set_facecolor('#F0EFE8')
    bars = ax.bar(names, values, color=COLORS, edgecolor='#2C2C2A', linewidth=0.6, width=0.55)
    ax.set_title(title, fontsize=10, fontweight='600', color='#2C2C2A', pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis='x', labelrotation=15, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines[['top','right']].set_visible(False)
    mx = max(values) if values else 1
    for bar, v in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+mx*0.01,
                fmt.format(v), ha='center', va='bottom', fontsize=8, fontweight='500')
 
styled_bar(axes[0], model_names, accuracies,  'Classification Accuracy', 'Accuracy')
axes[0].set_ylim(0, 1.12)
styled_bar(axes[1], model_names, f1_scores,   'Macro-F1 Score',          'Macro-F1')
axes[1].set_ylim(0, 1.12)
styled_bar(axes[2], model_names, train_times, 'Training Time',            'Seconds', '{:.2f}s')
styled_bar(axes[3], model_names, test_times,  'Test Time',                'Seconds', '{:.4f}s')
 
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig('./plots/phase4_model_comparison.png', dpi=150,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: plots/phase4_model_comparison.png")
 
# ============================================================
# CONFUSION MATRICES
# ============================================================
fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4),
                         facecolor='#F8F7F4')
fig.suptitle('Confusion Matrices', fontsize=12, fontweight='700', color='#2C2C2A')
 
for ax, (name, v) in zip(axes, results.items()):
    ax.set_facecolor('#F0EFE8')
    cm = confusion_matrix(y_test, v['y_pred'])
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=9); ax.set_ylabel('Actual', fontsize=9)
    ax.set_title(f'{name}\nacc={v["accuracy"]:.3f}  F1={v["f1_macro"]:.3f}',
                 fontsize=9, fontweight='600', color='#2C2C2A')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=11, fontweight='600',
                    color='white' if cm[i,j] > cm.max()*0.5 else '#2C2C2A')
 
plt.tight_layout(rect=[0,0,1,0.93])
plt.savefig('./plots/phase4_confusion_matrices.png', dpi=150,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("Saved: plots/phase4_confusion_matrices.png")
 
# ============================================================
# BEST MODEL (BY F1, NOT ACCURACY)
# ============================================================
best_model = max(results, key=lambda x: results[x]['f1_macro'])
 
print("\n" + "="*60)
print(f" BEST MODEL (by F1): {best_model}")
print("="*60)