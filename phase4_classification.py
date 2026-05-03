import pandas as pd
import numpy as np
import time
import os

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

os.makedirs('./plots', exist_ok=True)

print("=" * 60)
print("PHASE 4: CLASSIFICATION (FINAL VERSION)")
print("=" * 60)

# =========================================================
# LOAD DATA
# =========================================================
train = pd.read_csv('./data/processed/train_processed.csv')
test  = pd.read_csv('./data/processed/test_processed.csv')

X_train = train.drop(columns=['target'])
y_train = train['target']

X_test = test.drop(columns=['target'])
y_test = test['target']

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print("Balanced:", np.bincount(y_train))

# =========================================================
# MODELS
# =========================================================
models = {

    "Random Forest": Pipeline([
        ('model', RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample"
        ))
    ]),

    "Decision Tree": Pipeline([
        ('model', DecisionTreeClassifier(
            max_depth=10,
            class_weight={0:1, 1:1, 2:8},
            random_state=42
        ))
    ]),

    "SVM": Pipeline([
        ('model', SVC(
            kernel='rbf',
            C=2.0,
            class_weight={0:1, 1:1, 2:8}
        ))
    ]),

    "KNN": Pipeline([
        ('model', KNeighborsClassifier(n_neighbors=5))
    ]),

    "XGBoost": Pipeline([
        ('model', XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42
        ))
    ]),

    "LightGBM": Pipeline([
        ('model', LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            num_leaves=63,
            min_child_samples=20,
            random_state=42
        ))
    ]),

    "CatBoost": Pipeline([
        ('model', CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            loss_function='MultiClass',
            class_weights=[1, 1, 8],
            verbose=0,
            random_state=42
        ))
    ])
}
def hierarchical_predict(X_train, y_train, X_test, y_test):

    # ── Stage 1: Low vs Rest ──
    y_train_s1 = (y_train != 0).astype(int)
    y_test_s1  = (y_test  != 0).astype(int)

    stage1 = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=42
    )
    stage1.fit(X_train, y_train_s1)

    # ── Stage 2: Medium vs High ──
    mask_train = y_train != 0
    mask_test  = y_test  != 0

    X_train_s2 = X_train[mask_train]
    y_train_s2 = y_train[mask_train]

    X_test_s2  = X_test[mask_test]
    y_test_s2  = y_test[mask_test]

    y_train_s2 = (y_train_s2 == 2).astype(int)
    y_test_s2  = (y_test_s2  == 2).astype(int)

    stage2 = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=42
    )
    stage2.fit(X_train_s2, y_train_s2)

    # ── Prediction ──
    final_preds = []

    for i in range(len(X_test)):
        pred_s1 = stage1.predict(X_test.iloc[[i]])[0]

        if pred_s1 == 0:
            final_preds.append(0)  # Low
        else:
            pred_s2 = stage2.predict(X_test.iloc[[i]])[0]
            final_preds.append(1 if pred_s2 == 0 else 2)

    return np.array(final_preds), stage1, stage2

# =========================================================
# TRAIN MODELS
# =========================================================
results = []

print("\nTraining Models...\n")


print("\n── Hierarchical (2-Stage) ──")

t0 = time.time()
y_pred_h, stage1_model, stage2_model = hierarchical_predict(
    X_train, y_train, X_test, y_test
)
train_time = time.time() - t0

t0 = time.time()
_ = y_pred_h  # already predicted
test_time = time.time() - t0

acc = accuracy_score(y_test, y_pred_h)
f1  = f1_score(y_test, y_pred_h, average='macro')

print(f"  Accuracy : {acc:.4f}")
print(f"  F1-macro: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_h))

results.append([
    "Hierarchical (2-Stage)",
    acc,
    f1,
    train_time,
    test_time
])


for name, model in models.items():

    print(f"── {name} ──")

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - start

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro')

    print(f"Accuracy : {acc:.4f}")
    print(f"F1-macro: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    results.append([name, acc, f1, train_time, test_time])

# =========================================================
# ENSEMBLE (VOTING)
# =========================================================
print("\n── Ensemble Model ──")

ensemble_model = VotingClassifier(
    estimators=[
        ('lgbm', LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            num_leaves=63,
            random_state=42
        )),

        ('xgb', XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42
        )),

        ('rf', RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        ))
    ],
    voting='soft'
)

start = time.time()
ensemble_model.fit(X_train, y_train)
train_time = time.time() - start

start = time.time()
y_pred = ensemble_model.predict(X_test)
test_time = time.time() - start

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average='macro')

print("Accuracy:", acc)
print("F1-macro:", f1)
print(classification_report(y_test, y_pred))

# =========================================================
# SUMMARY
# =========================================================
results.append(["Ensemble", acc, f1, train_time, test_time])

results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "F1-macro", "Train Time", "Test Time"
])

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(results_df.sort_values(by="F1-macro", ascending=False))

best = results_df.sort_values(by="F1-macro", ascending=False).iloc[0]
print("\n BEST MODEL:", best["Model"])

print("\nPred distribution:")
print(np.bincount(y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))