import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import joblib
import os

os.makedirs('./data/processed', exist_ok=True)

print("=" * 60)
print("PHASE 2: PREPROCESSING (MEMORY SAFE)")
print("=" * 60)

# 1 Load data
df = pd.read_csv('./data/raw/train_data.csv')

# 2 Target
y = df['GamePopularity']
X = df.drop(columns=['GamePopularity'])

# 3 Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# =========================================================
# 🔥 FIX 1: HANDLE HIGH CARDINALITY (VERY IMPORTANT)
# =========================================================
cat_cols = X.select_dtypes(include='object').columns

for col in cat_cols:
    if X[col].nunique() > 20:
        print(f"Dropping high-cardinality column: {col}")
        X = X.drop(columns=[col])

# =========================================================
# 🔥 FIX 2: SAFE ONE-HOT ENCODING
# =========================================================
X = pd.get_dummies(X, drop_first=True)

# Reduce memory usage
X = X.astype(np.float32)

print("Shape after encoding:", X.shape)

# =========================================================
# 4 Train/Test split (stratified)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    stratify=y,
    random_state=42
)

# =========================================================
# 5 Scaling
# =========================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# رجعهم DataFrame
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test  = pd.DataFrame(X_test, columns=X.columns)

# =========================================================
# 🔥 FIX 3: Feature Selection on REAL DATA
# =========================================================
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

mi_df = pd.DataFrame({
    'feature': X_train.columns,
    'MI': mi_scores
}).sort_values('MI', ascending=False)

MI_THRESHOLD = 0.005
selected_features = mi_df[mi_df['MI'] > MI_THRESHOLD]['feature'].tolist()

print(f"Selected features: {len(selected_features)}")

X_train = X_train[selected_features]
X_test  = X_test[selected_features]

# =========================================================
# 6 Save
# =========================================================
train_processed = pd.concat([X_train, pd.Series(y_train, name='target')], axis=1)
test_processed  = pd.concat([X_test, pd.Series(y_test, name='target')], axis=1)

train_processed.to_csv('./data/processed/train_processed.csv', index=False)
test_processed.to_csv('./data/processed/test_processed.csv', index=False)

joblib.dump(scaler, './data/processed/scaler.pkl')
joblib.dump(le, './data/processed/label_encoder.pkl')

print("Saved processed data successfully.")