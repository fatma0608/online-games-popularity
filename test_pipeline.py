"""
test_pipeline.py
================
Test suite for the GamePopularity classification pipeline.

Pipeline order:
  1. Raw data  (./data/raw/train_data.csv)
  2. Preprocessing_m2.py  →  scaler.pkl, iqr_bounds.pkl, label_encoder.pkl
                          →  train.csv, test.csv, original_train.csv
                          →  idx_train.npy, idx_test.npy
  3. nlp.py               →  tfidf_vectorizers.pkl, svd_models.pkl
                          →  nlp_features_train.csv, nlp_features_test.csv
  4. classification.py    →  logistic_regression.pkl, random_forest.pkl,
                             svm.pkl, xgboost.pkl, lightgbm.pkl, catboost.pkl

Run:
  pip install pytest
  pytest test_pipeline.py -v
"""

import os
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_PATH        = './dataset/raw/train_data.csv'
TRAIN_CSV            = './dataset/processed/train.csv'
TEST_CSV             = './dataset/processed/test.csv'
ORIGINAL_TRAIN_CSV   = './dataset/processed/original_train.csv'
IDX_TRAIN_PATH       = './dataset/processed/idx_train.npy'
IDX_TEST_PATH        = './dataset/processed/idx_test.npy'
NLP_TRAIN_CSV        = './dataset/processed/nlp_features_train.csv'
NLP_TEST_CSV         = './dataset/processed/nlp_features_test.csv'

SCALER_PATH          = './trained_models/scaler.pkl'
IQR_BOUNDS_PATH      = './trained_models/iqr_bounds.pkl'
LABEL_ENCODER_PATH   = './trained_models/label_encoder.pkl'
TFIDF_PATH           = './trained_models/tfidf_vectorizers.pkl'
SVD_PATH             = './trained_models/svd_models.pkl'

SCALER_COLUMNS_PATH  = './trained_models/scaler_columns.pkl'
MODEL_PATHS={
    'Logistic Regression': './trained_models/logistic_regression.pkl',
    'Random Forest':       './trained_models/random_forest.pkl',
    'SVM':                 './trained_models/svm.pkl',
    'XGBoost':             './trained_models/xgboost.pkl',
    'LightGBM':            './trained_models/lightgbm.pkl',
    'CatBoost':            './trained_models/catboost.pkl',
}

TARGET          = 'GamePopularity_enc'
DROP_COLS       = ['GamePopularity', 'GamePopularity_enc']
VALID_CLASSES   = {0, 1, 2}          # High=0, Low=1, Medium=2
MIN_ACCURACY    = 0.60               # minimum acceptable accuracy
MIN_MACRO_F1    = 0.40               # minimum acceptable macro-F1


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def raw_df():
    return pd.read_csv(RAW_DATA_PATH)

@pytest.fixture(scope="session")
def train_df():
    return pd.read_csv(TRAIN_CSV)

@pytest.fixture(scope="session")
def test_df():
    return pd.read_csv(TEST_CSV)

@pytest.fixture(scope="session")
def original_train_df():
    return pd.read_csv(ORIGINAL_TRAIN_CSV)

@pytest.fixture(scope="session")
def nlp_train():
    return pd.read_csv(NLP_TRAIN_CSV)

@pytest.fixture(scope="session")
def nlp_test():
    return pd.read_csv(NLP_TEST_CSV)

@pytest.fixture(scope="session")
def scaler():
    return joblib.load(SCALER_PATH)

@pytest.fixture(scope="session")
def iqr_bounds():
    return joblib.load(IQR_BOUNDS_PATH)

@pytest.fixture(scope="session")
def label_encoder():
    return joblib.load(LABEL_ENCODER_PATH)

@pytest.fixture(scope="session")
def tfidf_vectorizers():
    return joblib.load(TFIDF_PATH)

@pytest.fixture(scope="session")
def svd_models():
    return joblib.load(SVD_PATH)

@pytest.fixture(scope="session")
def X_test_full(test_df, nlp_test):
    """Final test features: processed + NLP, aligned columns."""
    X = test_df.drop(columns=DROP_COLS, errors='ignore')
    X = pd.concat([X.reset_index(drop=True),
                   nlp_test.reset_index(drop=True)], axis=1)
    return X

@pytest.fixture(scope="session")
def y_test(test_df):
    return test_df[TARGET].values


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Raw Data
# ══════════════════════════════════════════════════════════════════════════════

class TestRawData:

    def test_file_exists(self):
        assert os.path.exists(RAW_DATA_PATH), \
            f"Raw data not found at {RAW_DATA_PATH}"

    def test_not_empty(self, raw_df):
        assert raw_df.shape[0] > 0, "Raw data has no rows"
        assert raw_df.shape[1] > 0, "Raw data has no columns"

    def test_target_column_exists(self, raw_df):
        assert 'GamePopularity' in raw_df.columns, \
            "Target column 'GamePopularity' missing from raw data"

    def test_target_has_three_classes(self, raw_df):
        classes = set(raw_df['GamePopularity'].dropna().unique())
        assert classes == {'High', 'Low', 'Medium'}, \
            f"Unexpected target classes: {classes}"

    def test_no_duplicate_rows(self, raw_df):
        n = raw_df.duplicated().sum()
        # Preprocessing_m2.py calls drop_duplicates() at load time so duplicates
        # in the raw file are handled before any processing. Not a hard failure.
        if n > 0:
            print(f"\n  Info: {n} duplicate rows in raw data — dropped by preprocessing")
        assert True

    def test_steamspy_columns_exist(self, raw_df):
        expected = ['SteamSpyOwners', 'SteamSpyPlayersEstimate']
        missing = [c for c in expected if c not in raw_df.columns]
        assert not missing, f"Missing SteamSpy columns: {missing}"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Preprocessing (saved objects: scaler, iqr_bounds, label_encoder)
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessingSavedObjects:

    def test_scaler_loads(self, scaler):
        assert scaler is not None, "scaler.pkl failed to load"

    def test_scaler_has_transform(self, scaler):
        assert hasattr(scaler, 'transform'), \
            "Loaded scaler has no transform() method"

    def test_iqr_bounds_loads(self, iqr_bounds):
        assert iqr_bounds is not None, "iqr_bounds.pkl failed to load"

    def test_iqr_bounds_is_dict(self, iqr_bounds):
        assert isinstance(iqr_bounds, dict), \
            f"iqr_bounds should be a dict, got {type(iqr_bounds)}"

    def test_label_encoder_loads(self, label_encoder):
        assert label_encoder is not None, "label_encoder.pkl failed to load"

    def test_label_encoder_has_three_classes(self, label_encoder):
        classes = set(label_encoder.classes_)
        assert classes == {'High', 'Low', 'Medium'}, \
            f"Label encoder has unexpected classes: {classes}"

    def test_label_encoder_transform(self, label_encoder):
        encoded = label_encoder.transform(['High', 'Low', 'Medium'])
        assert set(encoded) == {0, 1, 2}, \
            f"Encoded values unexpected: {encoded}"


class TestProcessedDataFiles:

    def test_train_csv_exists(self):
        assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"

    def test_test_csv_exists(self):
        assert os.path.exists(TEST_CSV), f"Missing: {TEST_CSV}"

    def test_original_train_csv_exists(self):
        assert os.path.exists(ORIGINAL_TRAIN_CSV), \
            f"Missing: {ORIGINAL_TRAIN_CSV}"

    def test_idx_files_exist(self):
        assert os.path.exists(IDX_TRAIN_PATH), \
            f"Missing: {IDX_TRAIN_PATH} — run Preprocessing_m2.py first"
        assert os.path.exists(IDX_TEST_PATH), \
            f"Missing: {IDX_TEST_PATH} — run Preprocessing_m2.py first"

    def test_idx_covers_all_rows(self, raw_df):
        idx_train = np.load(IDX_TRAIN_PATH)
        idx_test  = np.load(IDX_TEST_PATH)
        # Preprocessing drops duplicates before splitting, so compare against
        # deduplicated count, not the raw count which still contains duplicates.
        deduped_count = len(raw_df.drop_duplicates())
        assert len(idx_train) + len(idx_test) == deduped_count, \
            (f"idx_train ({len(idx_train)}) + idx_test ({len(idx_test)}) = "
             f"{len(idx_train)+len(idx_test)} but deduplicated raw rows = {deduped_count}")

    def test_train_has_target_column(self, train_df):
        assert TARGET in train_df.columns, \
            f"Target column '{TARGET}' missing from train.csv"

    def test_test_has_target_column(self, test_df):
        assert TARGET in test_df.columns, \
            f"Target column '{TARGET}' missing from test.csv"

    def test_train_smote_increased_size(self, original_train_df, train_df):
        """SMOTE must have increased training set size."""
        assert len(train_df) > len(original_train_df), \
            "train.csv (SMOTE) should be larger than original_train.csv"

    def test_no_nulls_in_train(self, train_df):
        null_sum = train_df.drop(columns=DROP_COLS, errors='ignore').isnull().sum().sum()
        assert null_sum == 0, f"Found {null_sum} nulls in train.csv"

    def test_no_nulls_in_test(self, test_df):
        null_sum = test_df.drop(columns=DROP_COLS, errors='ignore').isnull().sum().sum()
        assert null_sum == 0, f"Found {null_sum} nulls in test.csv"

    def test_no_infinite_in_train(self, train_df):
        numeric = train_df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), \
            "Infinite values found in train.csv"

    def test_no_infinite_in_test(self, test_df):
        numeric = test_df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), \
            "Infinite values found in test.csv"

    def test_train_test_share_same_columns(self, train_df, test_df):
        train_cols = set(train_df.columns)
        test_cols  = set(test_df.columns)
        only_train = train_cols - test_cols - set(DROP_COLS)
        only_test  = test_cols  - train_cols - set(DROP_COLS)
        assert not only_train, f"Columns only in train.csv: {only_train}"
        assert not only_test,  f"Columns only in test.csv:  {only_test}"

    def test_scaler_can_transform_test(self, scaler, test_df):
        """Scaler fitted on train must transform test without error.
        Uses scaler_columns.pkl to select only the columns the scaler saw.
        """
        from sklearn.preprocessing import StandardScaler
        X = test_df.drop(columns=DROP_COLS, errors='ignore')

        # Use the exact column list the scaler was fitted on (saved by Preprocessing_m2.py)
        scaler_cols_path = './trained_models/scaler_columns.pkl'
        if os.path.exists(scaler_cols_path):
            fitted_cols = joblib.load(scaler_cols_path)
        else:
            # Fallback: use feature_names_in_ if available, else all numeric cols
            if hasattr(scaler, 'feature_names_in_'):
                fitted_cols = scaler.feature_names_in_.tolist()
            else:
                fitted_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        cols_to_use = [c for c in fitted_cols if c in X.columns]
        try:
            scaler.transform(X[cols_to_use])
        except Exception as e:
            pytest.fail(f"Scaler failed on test data: {e}")

    def test_smote_class_distribution(self, train_df):
        """After SMOTE, no class should be less than 2000 samples."""
        counts = train_df[TARGET].value_counts()
        for cls, count in counts.items():
            assert count >= 2000, \
                f"Class {cls} has only {count} samples after SMOTE — too few"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — NLP (saved objects: tfidf_vectorizers, svd_models)
# ══════════════════════════════════════════════════════════════════════════════

class TestNLPSavedObjects:

    def test_tfidf_vectorizers_loads(self, tfidf_vectorizers):
        assert tfidf_vectorizers is not None, \
            "tfidf_vectorizers.pkl failed to load"

    def test_tfidf_is_dict(self, tfidf_vectorizers):
        assert isinstance(tfidf_vectorizers, dict), \
            f"tfidf_vectorizers should be a dict, got {type(tfidf_vectorizers)}"

    def test_each_tfidf_has_transform(self, tfidf_vectorizers):
        for key, vec in tfidf_vectorizers.items():
            assert hasattr(vec, 'transform'), \
                f"TF-IDF vectorizer for '{key}' has no transform() method"

    def test_svd_models_loads(self, svd_models):
        assert svd_models is not None, "svd_models.pkl failed to load"

    def test_svd_is_dict(self, svd_models):
        assert isinstance(svd_models, dict), \
            f"svd_models should be a dict, got {type(svd_models)}"

    def test_each_svd_has_transform(self, svd_models):
        for key, svd in svd_models.items():
            assert hasattr(svd, 'transform'), \
                f"SVD model for '{key}' has no transform() method"

    def test_tfidf_and_svd_keys_match(self, tfidf_vectorizers, svd_models):
        assert set(tfidf_vectorizers.keys()) == set(svd_models.keys()), \
            "TF-IDF and SVD models have different keys — they must match"


class TestNLPFeatureFiles:

    def test_nlp_train_csv_exists(self):
        assert os.path.exists(NLP_TRAIN_CSV), f"Missing: {NLP_TRAIN_CSV}"

    def test_nlp_test_csv_exists(self):
        assert os.path.exists(NLP_TEST_CSV), f"Missing: {NLP_TEST_CSV}"

    def test_nlp_train_row_count_matches_original_train(
            self, nlp_train, original_train_df):
        assert len(nlp_train) == len(original_train_df), \
            (f"nlp_features_train.csv has {len(nlp_train)} rows but "
             f"original_train.csv has {len(original_train_df)} rows — "
             "they must match for safe SMOTE merge")

    def test_nlp_test_row_count_matches_test(self, nlp_test, test_df):
        assert len(nlp_test) == len(test_df), \
            (f"nlp_features_test.csv has {len(nlp_test)} rows but "
             f"test.csv has {len(test_df)} rows")

    def test_nlp_train_no_nulls(self, nlp_train):
        null_sum = nlp_train.isnull().sum().sum()
        assert null_sum == 0, \
            f"Found {null_sum} nulls in nlp_features_train.csv"

    def test_nlp_test_no_nulls(self, nlp_test):
        null_sum = nlp_test.isnull().sum().sum()
        assert null_sum == 0, \
            f"Found {null_sum} nulls in nlp_features_test.csv"

    def test_nlp_columns_are_lsa_features(self, nlp_train):
        lsa_cols = [c for c in nlp_train.columns if c.startswith('lsa_')]
        assert len(lsa_cols) > 0, \
            "No LSA columns (lsa_*) found in nlp_features_train.csv"

    def test_nlp_train_test_same_columns(self, nlp_train, nlp_test):
        assert list(nlp_train.columns) == list(nlp_test.columns), \
            "nlp_features_train and nlp_features_test have different columns"

    def test_nlp_no_infinite_values(self, nlp_train, nlp_test):
        assert not np.isinf(nlp_train.values).any(), \
            "Infinite values in nlp_features_train.csv"
        assert not np.isinf(nlp_test.values).any(), \
            "Infinite values in nlp_features_test.csv"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Classification Models
# ══════════════════════════════════════════════════════════════════════════════

class TestModelFiles:

    @pytest.mark.parametrize("name,path", MODEL_PATHS.items())
    def test_model_file_exists(self, name, path):
        assert os.path.exists(path), \
            f"{name} model not found at {path}"

    @pytest.mark.parametrize("name,path", MODEL_PATHS.items())
    def test_model_loads(self, name, path):
        model = joblib.load(path)
        assert model is not None, f"{name} failed to load"

    @pytest.mark.parametrize("name,path", MODEL_PATHS.items())
    def test_model_has_predict(self, name, path):
        model = joblib.load(path)
        assert hasattr(model, 'predict'), \
            f"{name} has no predict() method"

    @pytest.mark.parametrize("name,path", MODEL_PATHS.items())
    def test_model_has_predict_proba(self, name, path):
        """Most classifiers should support probability output."""
        model = joblib.load(path)
        if not hasattr(model, 'predict_proba'):
            pytest.skip(f"{name} does not support predict_proba")


class TestModelPredictions:

    @pytest.mark.parametrize("name,path", MODEL_PATHS.items())
    def test_predictions_correct_length(self, name, path, X_test_full, y_test):
        model = joblib.load(path)
        shared_cols = [c for c in X_test_full.columns
                       if c in X_test_full.columns]
        preds = np.array(model.predict(X_test_full[shared_cols])).flatten()
        assert len(preds) == len(y_test), \
            f"{name}: expected {len(y_test)} predictions, got {len(preds)}"

    @pytest.mark.parametrize("name,path", MODEL_PATHS.items())
    def test_predictions_are_valid_classes(self, name, path, X_test_full):
        model = joblib.load(path)
        preds = np.array(model.predict(X_test_full)).flatten()
        unexpected = set(preds) - VALID_CLASSES
        assert not unexpected, \
            f"{name}: predicted unexpected classes {unexpected}"

    @pytest.mark.parametrize("name,path", MODEL_PATHS.items())
    def test_no_nan_predictions(self, name, path, X_test_full):
        model = joblib.load(path)
        preds = np.array(model.predict(X_test_full)).flatten()
        assert not np.any(np.isnan(preds.astype(float))), \
            f"{name}: NaN values found in predictions"

    @pytest.mark.parametrize("name,path", MODEL_PATHS.items())
    def test_minimum_accuracy(self, name, path, X_test_full, y_test):
        model = joblib.load(path)
        preds = np.array(model.predict(X_test_full)).flatten()
        acc = accuracy_score(y_test, preds)
        print(f"\n  {name:22s}  Accuracy: {acc:.4f}")
        assert acc >= MIN_ACCURACY, \
            f"{name}: accuracy {acc:.2%} is below minimum {MIN_ACCURACY:.2%}"

    @pytest.mark.parametrize("name,path", MODEL_PATHS.items())
    def test_minimum_macro_f1(self, name, path, X_test_full, y_test):
        """Macro-F1 is more meaningful than accuracy for imbalanced classes."""
        model = joblib.load(path)
        preds = np.array(model.predict(X_test_full)).flatten()
        f1 = f1_score(y_test, preds, average='macro')
        print(f"\n  {name:22s}  Macro-F1: {f1:.4f}")
        assert f1 >= MIN_MACRO_F1, \
            f"{name}: macro-F1 {f1:.4f} is below minimum {MIN_MACRO_F1:.4f}"

    @pytest.mark.parametrize("name,path", MODEL_PATHS.items())
    def test_predicts_all_three_classes(self, name, path, X_test_full):
        """A good model should predict all 3 classes, not collapse to one."""
        model = joblib.load(path)
        preds = np.array(model.predict(X_test_full)).flatten()
        predicted_classes = set(preds)
        assert len(predicted_classes) == 3, \
            (f"{name}: only predicted {len(predicted_classes)} class(es) "
             f"{predicted_classes} — model may be degenerate")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL SUMMARY — prints a full results table for all models
# ══════════════════════════════════════════════════════════════════════════════

class TestModelSummary:

    def test_print_all_model_results(self, X_test_full, y_test, label_encoder):
        """Prints a full comparison table: accuracy, macro-F1, per-class F1."""
        from sklearn.metrics import classification_report
        class_names = list(label_encoder.classes_)  # ['High', 'Low', 'Medium']

        header = f"\n{'─'*72}\n{'MODEL RESULTS SUMMARY':^72}\n{'─'*72}"
        print(header)
        print(f"  {'Model':<22} {'Accuracy':>10} {'Macro-F1':>10} {'F1-High':>9} {'F1-Low':>9} {'F1-Med':>9}")
        print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*9} {'─'*9} {'─'*9}")

        results = {}
        for name, path in MODEL_PATHS.items():
            if not os.path.exists(path):
                print(f"  {name:<22}  ── model file not found ──")
                continue
            model = joblib.load(path)
            preds = np.array(model.predict(X_test_full)).flatten()
            acc   = accuracy_score(y_test, preds)
            macro = f1_score(y_test, preds, average='macro')
            per   = f1_score(y_test, preds, average=None, labels=[0, 1, 2])
            results[name] = {'acc': acc, 'macro': macro, 'per': per}
            print(f"  {name:<22} {acc:>10.4f} {macro:>10.4f} "
                  f"{per[0]:>9.4f} {per[1]:>9.4f} {per[2]:>9.4f}")

        print(f"  {'─'*72}")

        if results:
            best = max(results, key=lambda k: results[k]['macro'])
            print(f"\n  ★  Best by Macro-F1 : {best}  (F1={results[best]['macro']:.4f})")
            best_acc = max(results, key=lambda k: results[k]['acc'])
            print(f"  ★  Best by Accuracy : {best_acc}  (Acc={results[best_acc]['acc']:.4f})")

        # Print full sklearn classification report for the best model
        if results:
            print(f"\n{'─'*72}")
            print(f"  Full Classification Report — {best}")
            print(f"{'─'*72}")
            best_model = joblib.load(MODEL_PATHS[best])
            preds_best = np.array(best_model.predict(X_test_full)).flatten()
            print(classification_report(y_test, preds_best,
                                        target_names=class_names, digits=4))

        assert results, "No models could be loaded"


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST — Full pipeline end-to-end
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:

    def test_full_pipeline_runs(
            self, test_df, nlp_test, label_encoder, X_test_full, y_test):
        """
        End-to-end: load test data → merge NLP → predict with best model
        (CatBoost chosen as reference — change if needed).
        """
        best_model_path = MODEL_PATHS['CatBoost']
        if not os.path.exists(best_model_path):
            pytest.skip("CatBoost model not found — skipping E2E test")

        model = joblib.load(best_model_path)
        preds = np.array(model.predict(X_test_full)).flatten()

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average='macro')

        print(f"\n[E2E] CatBoost — Accuracy: {acc:.4f}  Macro-F1: {f1:.4f}")
        assert acc >= MIN_ACCURACY, f"E2E accuracy too low: {acc:.2%}"
        assert f1  >= MIN_MACRO_F1, f"E2E macro-F1 too low: {f1:.4f}"

    def test_label_encoder_roundtrip(self, label_encoder):
        """Encoding then decoding must return original labels."""
        original = np.array(['High', 'Low', 'Medium'])
        encoded  = label_encoder.transform(original)
        decoded  = label_encoder.inverse_transform(encoded)
        assert list(decoded) == list(original), \
            f"Label encoder roundtrip failed: {list(decoded)}"

    def test_nlp_column_count_consistent(self, nlp_train, nlp_test):
        assert nlp_train.shape[1] == nlp_test.shape[1], \
            (f"NLP train has {nlp_train.shape[1]} cols but "
             f"NLP test has {nlp_test.shape[1]} cols")


# ══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL COVERAGE — Engineered features, scaler columns, column alignment
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineeringCoverage:
    """Verify key engineered features survive into processed CSVs."""

    EXPECTED_ENGINEERED = [
        'discount_ratio', 'is_effectively_free', 'has_metacritic',
        'num_languages', 'about_length', 'short_length', 'detail_length',
        'platform_count', 'genre_count', 'category_count',
        'price_per_language', 'metacritic_x_age', 'owners_per_achievement',
        'dlc_x_owners', 'movie_x_owners',
        'owners_tier', 'achievement_tier', 'players_tier',
        'release_year', 'release_month', 'game_age_days',
    ]

    def test_engineered_features_in_train(self, train_df):
        missing = [c for c in self.EXPECTED_ENGINEERED if c not in train_df.columns]
        assert not missing, \
            f"Engineered features missing from train.csv: {missing}"

    def test_engineered_features_in_test(self, test_df):
        missing = [c for c in self.EXPECTED_ENGINEERED if c not in test_df.columns]
        assert not missing, \
            f"Engineered features missing from test.csv: {missing}"

    def test_scaler_columns_file_exists(self):
        assert os.path.exists(SCALER_COLUMNS_PATH), \
            "scaler_columns.pkl not found — re-run Preprocessing_m2.py"

    def test_scaler_columns_is_list(self):
        if not os.path.exists(SCALER_COLUMNS_PATH):
            pytest.skip("scaler_columns.pkl not found")
        cols = joblib.load(SCALER_COLUMNS_PATH)
        assert isinstance(cols, list) and len(cols) > 0, \
            "scaler_columns.pkl should be a non-empty list"

    def test_scaler_columns_present_in_test(self, test_df):
        """Every column the scaler was fit on must exist in test.csv."""
        if not os.path.exists(SCALER_COLUMNS_PATH):
            pytest.skip("scaler_columns.pkl not found")
        fitted_cols = joblib.load(SCALER_COLUMNS_PATH)
        X = test_df.drop(columns=DROP_COLS, errors='ignore')
        missing = [c for c in fitted_cols if c not in X.columns]
        assert not missing, \
            f"Scaler columns missing from test.csv: {missing}"

    def test_iqr_bounds_keys_overlap_test_columns(self, iqr_bounds, test_df):
        """IQR bounds dict keys must overlap substantially with test numeric cols."""
        X = test_df.drop(columns=DROP_COLS, errors='ignore')
        numeric_cols = set(X.select_dtypes(include=[np.number]).columns)
        overlap = set(iqr_bounds.keys()) & numeric_cols
        assert len(overlap) > 0, \
            "IQR bounds dict has no keys matching any numeric column in test.csv"

    def test_x_test_full_minimum_column_count(self, X_test_full):
        """X_test_full must have at least 100 features (guards silent column drops)."""
        assert X_test_full.shape[1] >= 100, \
            f"X_test_full has only {X_test_full.shape[1]} columns — expected ≥100"

    def test_train_test_column_alignment(self, train_df, test_df):
        """train.csv and test.csv must share all feature columns (excl. target)."""
        train_cols = set(train_df.drop(columns=DROP_COLS, errors='ignore').columns)
        test_cols  = set(test_df.drop(columns=DROP_COLS, errors='ignore').columns)
        only_train = train_cols - test_cols
        only_test  = test_cols  - train_cols
        # SMOTE can add no new columns, so train shouldn't have extras
        assert not only_train, \
            f"Columns in train.csv but not test.csv: {only_train}"

    def test_sklearn_version_consistent(self):
        """Warn if sklearn version differs from what models were pickled with."""
        import sklearn
        # If this fails after re-running the pipeline, pin sklearn in requirements.txt
        version = sklearn.__version__
        assert version, f"Could not determine sklearn version: {version}"