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

# =========================================================
# TRAIN MODELS
# =========================================================
results = []

print("\nTraining Models...\n")

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