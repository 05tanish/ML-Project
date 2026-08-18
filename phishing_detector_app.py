import io
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Phishing Website Detection",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
.main-title { text-align:center; padding:.35rem 0 0; }
.subtitle { text-align:center; color:#6b7280; font-size:1.02rem; margin-bottom:1.2rem; }
.hero-card { padding:1.6rem; border-radius:16px; border:1px solid rgba(128,128,128,.18); background:linear-gradient(135deg,#111827,#1f2937); color:white; }
.hero-card h1 { margin:0; }
.step-card { padding:1rem; border-radius:12px; border:1px solid rgba(128,128,128,.18); background:rgba(128,128,128,.04); min-height:115px; }
.best-model-card { background:linear-gradient(135deg,#111827,#1e3a5f); color:white; padding:1.7rem; border-radius:14px; text-align:center; margin:1rem 0; }
.best-model-card h2 { color:#ffd54f; margin:0; }
.best-model-card h3 { color:#e5e7eb; margin:.3rem 0; }
.metric-row { display:flex; justify-content:space-around; flex-wrap:wrap; margin-top:1rem; }
.metric-item { text-align:center; padding:.5rem 1rem; }
.metric-item .value { font-size:1.45rem; font-weight:700; color:#67e8f9; }
.metric-item .label { font-size:.82rem; color:#cbd5e1; }
.prediction-result { padding:1.5rem; border-radius:12px; text-align:center; font-size:1.35rem; font-weight:700; margin:1rem 0; }
.phishing { background:#fff1f2; color:#b91c1c; border:2px solid #ef4444; }
.legitimate { background:#f0fdf4; color:#15803d; border:2px solid #22c55e; }
.risk-high { background:#fff1f2; border-left:5px solid #ef4444; padding:1rem; border-radius:8px; }
.risk-medium { background:#fffbeb; border-left:5px solid #f59e0b; padding:1rem; border-radius:8px; }
.risk-low { background:#f0fdf4; border-left:5px solid #22c55e; padding:1rem; border-radius:8px; }
.small-note { color:#6b7280; font-size:.88rem; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("The uploaded CSV file is empty.")
            return None
        return df
    except Exception as exc:
        st.error(f"Error reading CSV file: {exc}")
        return None


def get_dataset_info(df, target_col):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numerical_cols": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_cols": list(df.select_dtypes(exclude=[np.number]).columns),
        "duplicates": int(df.duplicated().sum()),
        "missing_total": int(df.isnull().sum().sum()),
        "size_bytes": int(df.memory_usage(deep=True).sum()),
        "target_col": target_col,
        "feature_count": max(df.shape[1] - 1, 0),
    }


def clean_data(df):
    before = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    return df_clean, before - len(df_clean)


def normalize_binary_target(y):
    """Return a binary 0/1 target when possible, plus a mapping description."""
    unique = list(pd.Series(y).dropna().unique())
    if len(unique) != 2:
        raise ValueError("Target column must contain exactly two classes for binary phishing detection.")

    if set(unique) == {0, 1}:
        return pd.Series(y).astype(int).values, {0: 0, 1: 1}

    # Common string labels.
    lowered = {str(v).strip().lower(): v for v in unique}
    phishing_keys = [k for k in lowered if "phish" in k or "malicious" in k or "bad" in k or k in {"1", "true", "yes"}]
    legit_keys = [k for k in lowered if "legit" in k or "safe" in k or "benign" in k or k in {"0", "false", "no"}]
    if phishing_keys and legit_keys:
        phishing_original = lowered[phishing_keys[0]]
        legit_original = lowered[legit_keys[0]]
        mapping = {legit_original: 0, phishing_original: 1}
        return pd.Series(y).map(mapping).astype(int).values, mapping

    # Stable fallback: first sorted class -> 0, second -> 1.
    sorted_unique = sorted(unique, key=lambda x: str(x))
    mapping = {sorted_unique[0]: 0, sorted_unique[1]: 1}
    return pd.Series(y).map(mapping).astype(int).values, mapping


def preprocess_data(df, target_col, test_size, random_state):
    X = df.drop(columns=[target_col]).copy()
    y_raw = df[target_col].copy()

    if y_raw.isnull().any():
        raise ValueError("Target column contains missing values. Remove or fill target values before training.")

    y, target_mapping = normalize_binary_target(y_raw)
    feature_columns = list(X.columns)
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    num_imputer = None
    cat_imputer = None
    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_train.loc[:, num_cols] = num_imputer.fit_transform(X_train[num_cols])
        X_test.loc[:, num_cols] = num_imputer.transform(X_test[num_cols])
    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train.loc[:, cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test.loc[:, cat_cols] = cat_imputer.transform(X_test[cat_cols])

    ohe = None
    ohe_feature_names = []
    if cat_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        encoded_train = ohe.fit_transform(X_train[cat_cols])
        encoded_test = ohe.transform(X_test[cat_cols])
        ohe_feature_names = list(ohe.get_feature_names_out(cat_cols))
        encoded_train_df = pd.DataFrame(encoded_train, columns=ohe_feature_names, index=X_train.index)
        encoded_test_df = pd.DataFrame(encoded_test, columns=ohe_feature_names, index=X_test.index)
        X_train = X_train.drop(columns=cat_cols).join(encoded_train_df).reset_index(drop=True)
        X_test = X_test.drop(columns=cat_cols).join(encoded_test_df).reset_index(drop=True)
    else:
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

    scaler = StandardScaler()
    all_final_features = list(X_train.columns)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_unscaled = X_train.values
    X_test_unscaled = X_test.values

    preprocessor = {
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "ohe": ohe,
        "scaler": scaler,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_columns": feature_columns,
        "all_final_features": all_final_features,
        "ohe_feature_names": ohe_feature_names,
        "target_mapping": target_mapping,
    }

    return (
        X_train_scaled,
        X_test_scaled,
        X_train_unscaled,
        X_test_unscaled,
        y_train,
        y_test,
        preprocessor,
    )


def get_models(random_state, n_estimators=200):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True, random_state=random_state),
    }


def needs_scaling(model_name):
    return model_name in {"Logistic Regression", "K-Nearest Neighbors", "SVM"}


def train_models(models, X_train_scaled, X_test_scaled, X_train_unscaled, X_test_unscaled, y_train, y_test, cv_folds=5):
    results, trained_models, model_predictions, errors, cv_scores = [], {}, {}, {}, {}
    progress = st.progress(0)
    status = st.empty()

    for i, (name, model) in enumerate(models.items(), start=1):
        status.text(f"Training {name}... ({i}/{len(models)})")
        progress.progress(i / len(models))
        try:
            X_tr, X_te = (X_train_scaled, X_test_scaled) if needs_scaling(name) else (X_train_unscaled, X_test_unscaled)
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc = np.nan
            if y_prob is not None:
                try:
                    roc = roc_auc_score(y_test, y_prob)
                except Exception:
                    pass

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            # CV on training data only.
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_data = X_train_scaled if needs_scaling(name) else X_train_unscaled
            cv_model = get_models(42)[name]
            scores = cross_val_score(cv_model, cv_data, y_train, cv=cv, scoring="f1", n_jobs=None)
            cv_scores[name] = {"mean": float(scores.mean()), "std": float(scores.std())}

            results.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1 Score": round(f1, 4),
                "ROC-AUC": round(roc, 4) if not np.isnan(roc) else "N/A",
                "CV F1 Mean": round(float(scores.mean()), 4),
                "CV F1 Std": round(float(scores.std()), 4),
                "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            })
            trained_models[name] = model
            model_predictions[name] = {"y_pred": y_pred, "y_prob": y_prob}
        except Exception as exc:
            errors[name] = str(exc)
            st.warning(f"⚠️ {name} could not be trained: {exc}")

    progress.empty()
    status.empty()
    return pd.DataFrame(results), trained_models, model_predictions, errors, cv_scores


def compare_models(results_df):
    df = results_df.copy()
    df["ROC-AUC_num"] = pd.to_numeric(df["ROC-AUC"], errors="coerce")
    df = df.sort_values(["F1 Score", "CV F1 Mean", "ROC-AUC_num"], ascending=[False, False, False]).reset_index(drop=True)
    return df.drop(columns=["ROC-AUC_num"])


def predict_single_sample(input_data, preprocessor, model, model_name):
    input_df = pd.DataFrame([input_data])
    num_cols = preprocessor["num_cols"]
    cat_cols = preprocessor["cat_cols"]

    if num_cols and preprocessor["num_imputer"] is not None:
        input_df[num_cols] = preprocessor["num_imputer"].transform(input_df[num_cols])
    if cat_cols and preprocessor["cat_imputer"] is not None:
        input_df[cat_cols] = preprocessor["cat_imputer"].transform(input_df[cat_cols])

    ohe = preprocessor["ohe"]
    if cat_cols and ohe is not None:
        encoded = ohe.transform(input_df[cat_cols])
        names = list(ohe.get_feature_names_out(cat_cols))
        encoded_df = pd.DataFrame(encoded, columns=names, index=input_df.index)
        input_df = input_df.drop(columns=cat_cols)
        input_df = pd.concat([input_df, encoded_df], axis=1)

    all_final = preprocessor["all_final_features"]
    for col in all_final:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[all_final]

    input_arr = preprocessor["scaler"].transform(input_df) if needs_scaling(model_name) else input_df.values
    prediction = int(model.predict(input_arr)[0])
    probability = model.predict_proba(input_arr)[0] if hasattr(model, "predict_proba") else None
    return prediction, probability


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = LinearSegmentedColormap.from_list("custom", ["#e3f2fd", "#1565c0"])
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    labels = ["Legitimate (0)", "Phishing (1)"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("Actual Label", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    thresh = cm.max() / 2.0 if cm.size else 0
    names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{names[i][j]}\n{cm[i, j]:,}", ha="center", va="center", fontsize=14, fontweight="bold", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    return fig


def plot_roc_curve(y_test, y_prob, model_name, auc_val):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2.5, label=f"{model_name} (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], lw=1.5, linestyle="--", label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title("ROC Curve", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_precision_recall_curve(y_test, y_prob, model_name):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, lw=2.5, label=model_name)
    ax.set_xlabel("Recall", fontweight="bold")
    ax.set_ylabel("Precision", fontweight="bold")
    ax.set_title("Precision-Recall Curve", fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, model_name, top_n=15):
    importances = None
    label = "Importance"
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        label = "Feature Importance"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
        label = "Absolute Coefficient"
    if importances is None:
        return None
    if len(importances) != len(feature_names):
        n = min(len(importances), len(feature_names))
        importances, feature_names = importances[:n], feature_names[:n]
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(9, max(5, len(indices) * 0.38)))
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel(label, fontweight="bold")
    ax.set_title(f"Top {len(indices)} Feature Importances — {model_name}", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_metric_bar(results_df, metric):
    fig, ax = plt.subplots(figsize=(9, 5))
    values = pd.to_numeric(results_df[metric], errors="coerce")
    bars = ax.bar(results_df["Model"], values)
    for bar, value in zip(bars, values):
        if pd.notna(value):
            ax.text(bar.get_x() + bar.get_width()/2, value + .005, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(metric, fontweight="bold")
    ax.set_title(f"Model Comparison — {metric}", fontweight="bold")
    ax.grid(True, axis="y", alpha=.3)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    return fig


def risk_level(phishing_prob):
    if phishing_prob >= 70:
        return "HIGH", "risk-high"
    if phishing_prob >= 30:
        return "MEDIUM", "risk-medium"
    return "LOW", "risk-low"


def feature_input_form(df, feature_cols, key_prefix="pred"):
    input_data = {}
    binary_features = set()
    categorical_features = set()

    for feat in feature_cols:
        if feat not in df.columns:
            continue
        series = df[feat].dropna()
        unique = set(series.unique())
        if unique and unique.issubset({0, 1, 0.0, 1.0}):
            binary_features.add(feat)
        elif not pd.api.types.is_numeric_dtype(df[feat]):
            categorical_features.add(feat)

    cols = st.columns(3)
    for i, feat in enumerate(feature_cols):
        with cols[i % 3]:
            if feat in binary_features:
                input_data[feat] = st.selectbox(feat, [0, 1], index=0, key=f"{key_prefix}_{feat}")
            elif feat in categorical_features:
                options = df[feat].dropna().astype(str).unique().tolist()
                if not options:
                    options = [""]
                input_data[feat] = st.selectbox(feat, options, key=f"{key_prefix}_{feat}")
            else:
                series = pd.to_numeric(df[feat], errors="coerce").dropna()
                default = float(series.median()) if len(series) else 0.0
                min_val = float(series.min()) if len(series) else 0.0
                max_val = float(series.max()) if len(series) else 1000.0
                if min_val == max_val:
                    min_val -= 1.0; max_val += 1.0
                input_data[feat] = st.number_input(feat, min_value=min_val, max_value=max_val, value=min(max(default, min_val), max_val), key=f"{key_prefix}_{feat}")
    return input_data


# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------
defaults = {
    "df": None,
    "df_clean": None,
    "eda_done": False,
    "models_trained": False,
    "results_df": None,
    "trained_models": None,
    "model_predictions": None,
    "model_errors": None,
    "best_model_name": None,
    "preprocessor": None,
    "X_train_scaled": None,
    "X_test_scaled": None,
    "X_train_unscaled": None,
    "X_test_unscaled": None,
    "y_train": None,
    "y_test": None,
    "cv_scores": None,
    "prediction_history": [],
    "dataset_signature": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔐 Phishing Detector")
    st.caption("Machine-learning security dashboard")
    st.markdown("---")

    uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])
    if uploaded_file is not None:
        signature = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.dataset_signature != signature:
            loaded = load_data(uploaded_file)
            if loaded is not None:
                st.session_state.df = loaded
                st.session_state.dataset_signature = signature
                for key in ["df_clean", "results_df", "trained_models", "model_predictions", "model_errors", "best_model_name", "preprocessor", "cv_scores"]:
                    st.session_state[key] = defaults[key]
                st.session_state.models_trained = False
                st.session_state.eda_done = False
                st.session_state.prediction_history = []
                st.rerun()

    if st.session_state.df is not None:
        df_sidebar = st.session_state.df
        columns = list(df_sidebar.columns)
        default_idx = columns.index("Is_Phishing") if "Is_Phishing" in columns else 0
        target_col = st.selectbox("🎯 Target Column", columns, index=default_idx)
    else:
        target_col = "Is_Phishing"
        st.selectbox("🎯 Target Column", ["Upload dataset first"], disabled=True)

    st.markdown("---")
    st.markdown("### ⚙️ Training Settings")
    test_size = st.slider("Test Size", 0.10, 0.40, 0.20, 0.05)
    random_state = st.number_input("Random State", 0, 1000, 42, step=1)
    cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
    n_estimators = st.slider("Random Forest Trees", 50, 500, 200, 50)

    st.markdown("---")
    st.markdown("### 🚀 Actions")
    btn_eda = st.button("📊 Refresh EDA", use_container_width=True, disabled=st.session_state.df is None)
    btn_train = st.button("🤖 Train & Compare Models", use_container_width=True, type="primary", disabled=st.session_state.df is None)
    btn_reset = st.button("🔄 Reset Application", use_container_width=True)

    if btn_reset:
        for key, value in defaults.items():
            st.session_state[key] = value.copy() if isinstance(value, list) else value
        st.rerun()

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown('<h1 class="main-title">🔐 Phishing Website Detection Using Machine Learning</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">End-to-End Classification • Model Comparison • Explainability • Prediction</p>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HOME / EMPTY STATE
# -----------------------------------------------------------------------------
if st.session_state.df is None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>AI-Powered Phishing Detection</h1>
            <p>Upload a labelled CSV dataset to explore the data, train multiple classifiers, compare performance, inspect the best model, and make predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 🚀 Workflow")
    a, b, c, d = st.columns(4)
    cards = [
        ("1️⃣ Upload", "Upload your CSV dataset and select the target column."),
        ("2️⃣ Analyze", "Inspect quality, class balance, distributions and correlations."),
        ("3️⃣ Train", "Train six models and compare test and cross-validation F1."),
        ("4️⃣ Detect", "Use the best model for single or batch prediction."),
    ]
    for col, (title, desc) in zip([a, b, c, d], cards):
        with col:
            st.markdown(f'<div class="step-card"><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
    st.info("👈 Start by uploading a CSV dataset from the sidebar.")
    st.stop()

# -----------------------------------------------------------------------------
# TRAINING / EDA ACTIONS
# -----------------------------------------------------------------------------
df = st.session_state.df
if btn_eda:
    st.session_state.eda_done = True

if btn_train:
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in the dataset.")
    else:
        try:
            if df[target_col].nunique(dropna=True) != 2:
                raise ValueError("Target column must contain exactly two classes for binary phishing detection.")
            df_work, duplicates_removed = clean_data(df.copy())
            st.session_state.df_clean = df_work
            with st.spinner("Preprocessing data and training models..."):
                processed = preprocess_data(df_work, target_col, test_size, random_state)
                X_train_s, X_test_s, X_train_u, X_test_u, y_train, y_test, pp = processed
                models = get_models(random_state, n_estimators)
                results_df, trained_models, predictions, errors, cv_scores = train_models(
                    models, X_train_s, X_test_s, X_train_u, X_test_u, y_train, y_test, cv_folds
                )

            if results_df.empty:
                st.error("No models could be trained successfully.")
            else:
                results_df = compare_models(results_df)
                best_name = results_df.iloc[0]["Model"]
                st.session_state.X_train_scaled = X_train_s
                st.session_state.X_test_scaled = X_test_s
                st.session_state.X_train_unscaled = X_train_u
                st.session_state.X_test_unscaled = X_test_u
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.preprocessor = pp
                st.session_state.results_df = results_df
                st.session_state.trained_models = trained_models
                st.session_state.model_predictions = predictions
                st.session_state.model_errors = errors
                st.session_state.best_model_name = best_name
                st.session_state.cv_scores = cv_scores
                st.session_state.models_trained = True
                if duplicates_removed:
                    st.info(f"Removed {duplicates_removed:,} duplicate rows before training.")
                st.success(f"✅ Training completed. Best model: **{best_name}**")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📋 Overview", "📊 EDA", "⚙️ Preprocessing", "🤖 Training", "📈 Comparison", "🏆 Best Model", "🔮 Detect", "📜 History"
])

# TAB 1 -----------------------------------------------------------------------
with tab1:
    st.header("📋 Dataset Overview")
    info = get_dataset_info(df, target_col)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{info['rows']:,}")
    c2.metric("Columns", info["columns"])
    c3.metric("Features", info["feature_count"])
    c4.metric("Missing Values", f"{info['missing_total']:,}")
    c5.metric("Duplicates", f"{info['duplicates']:,}")

    a, b = st.columns(2)
    a.metric("Dataset Size", f"{info['size_bytes'] / 1024:.1f} KB")
    b.metric("Target", target_col)

    st.markdown("---")
    st.subheader("Dataset Preview")
    n_rows = st.selectbox("Rows to display", [5, 10, 20, 50], index=0)
    st.dataframe(df.head(n_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    a, b = st.columns(2)
    with a:
        st.subheader("Column Data Types")
        dtype_df = pd.DataFrame({"Column": df.columns, "Data Type": [str(dt) for dt in df.dtypes]})
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
    with b:
        st.subheader("Numerical vs Categorical")
        st.write(f"**Numerical ({len(info['numerical_cols'])}):** {', '.join(info['numerical_cols']) or 'None'}")
        st.write(f"**Categorical ({len(info['categorical_cols'])}):** {', '.join(info['categorical_cols']) or 'None'}")

    st.markdown("---")
    st.subheader("Target Distribution")
    if target_col in df.columns:
        target_counts = df[target_col].value_counts(dropna=False)
        st.dataframe(pd.DataFrame({"Class": target_counts.index.astype(str), "Count": target_counts.values, "Percentage": (target_counts.values / len(df) * 100).round(2)}), use_container_width=True, hide_index=True)
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

# TAB 2 -----------------------------------------------------------------------
with tab2:
    st.header("📊 Exploratory Data Analysis")
    if not st.session_state.eda_done:
        st.info("EDA is ready. Click **Refresh EDA** in the sidebar to mark the analysis as refreshed.")
    tc = df[target_col].value_counts().sort_index() if target_col in df.columns else pd.Series(dtype=int)
    if not tc.empty:
        st.subheader("Target Distribution")
        labels = ["Legitimate (0)" if v == 0 else "Phishing (1)" if v == 1 else str(v) for v in tc.index]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, tc.values)
        ax.set_ylabel("Count")
        ax.set_title("Target Distribution")
        ax.grid(True, axis="y", alpha=.3)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("---")
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    missing_df = pd.DataFrame({"Column": missing.index, "Missing Count": missing.values, "Missing %": (missing.values / len(df) * 100).round(2)})
    missing_df = missing_df[missing_df["Missing Count"] > 0]
    if missing_df.empty:
        st.success("✅ No missing values detected.")
    else:
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Feature Distribution")
    numeric_features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    if numeric_features:
        selected = st.selectbox("Select feature", numeric_features, key="eda_hist")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[selected].dropna(), bins=40)
        ax.set_xlabel(selected); ax.set_ylabel("Frequency"); ax.set_title(f"Distribution of {selected}")
        ax.grid(True, axis="y", alpha=.3)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    else:
        st.info("No numerical features available.")

    st.markdown("---")
    st.subheader("Box Plot / Outliers")
    if numeric_features:
        selected_box = st.selectbox("Select feature", numeric_features, key="eda_box")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.boxplot(df[selected_box].dropna(), vert=False)
        ax.set_xlabel(selected_box); ax.set_title(f"Box Plot — {selected_box}")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        q1, q3 = df[selected_box].quantile(.25), df[selected_box].quantile(.75)
        iqr = q3 - q1
        outliers = df[(df[selected_box] < q1 - 1.5 * iqr) | (df[selected_box] > q3 + 1.5 * iqr)]
        st.info(f"Potential outliers by IQR rule: **{len(outliers):,}**")

    st.markdown("---")
    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    if not corr.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        cmap = LinearSegmentedColormap.from_list("corr", ["#c62828", "#ffffff", "#1565c0"])
        im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=.046, pad=.04)
        ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(corr.columns, fontsize=8)
        ax.set_title("Feature Correlation Heatmap")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        with st.expander("🔍 Strong Correlations (|r| > 0.7)"):
            strong = []
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    value = corr.iloc[i, j]
                    if abs(value) > .7:
                        strong.append((corr.columns[i], corr.columns[j], round(float(value), 4)))
            if strong:
                for f1, f2, r in sorted(strong, key=lambda x: abs(x[2]), reverse=True):
                    st.write(f"• **{f1}** ↔ **{f2}**: r = {r}")
            else:
                st.write("No strongly correlated pairs found.")

# TAB 3 -----------------------------------------------------------------------
with tab3:
    st.header("⚙️ Preprocessing Pipeline")
    if not st.session_state.models_trained:
        st.info("Preprocessing is performed automatically when models are trained.")
        st.markdown("""
1. **Separate features and target**
2. **Remove duplicate rows**
3. **Stratified train/test split**
4. **Median imputation for numerical features**
5. **Most-frequent imputation for categorical features**
6. **One-hot encoding for categorical features**
7. **Standard scaling for Logistic Regression, KNN and SVM**

**Leakage prevention:** preprocessing statistics are fitted using training data only.
""")
    else:
        pp = st.session_state.preprocessor
        c1, c2, c3 = st.columns(3)
        c1.metric("Training Samples", f"{len(st.session_state.y_train):,}")
        c2.metric("Test Samples", f"{len(st.session_state.y_test):,}")
        c3.metric("Final Features", len(pp["all_final_features"]))
        steps = [
            ("Features / Target", f"{len(pp['feature_columns'])} original features; target = {target_col}"),
            ("Train/Test Split", f"{1-test_size:.0%} train / {test_size:.0%} test, stratified"),
            ("Missing Values", f"Numerical: median ({len(pp['num_cols'])} columns); categorical: most frequent ({len(pp['cat_cols'])} columns)"),
            ("Encoding", f"One-hot encoded {len(pp['cat_cols'])} categorical columns → {len(pp['ohe_feature_names'])} encoded features" if pp["cat_cols"] else "No categorical columns"),
            ("Scaling", "StandardScaler for Logistic Regression, KNN and SVM; tree-based models use unscaled features"),
        ]
        for title, desc in steps:
            with st.expander(title, expanded=False):
                st.write(desc)
        st.subheader("Final Feature List")
        st.dataframe(pd.DataFrame({"#": range(1, len(pp["all_final_features"]) + 1), "Feature": pp["all_final_features"]}), use_container_width=True, hide_index=True)

# TAB 4 -----------------------------------------------------------------------
with tab4:
    st.header("🤖 Model Training")
    if not st.session_state.models_trained:
        st.info("Use **Train & Compare Models** in the sidebar.")
        model_info = {
            "Logistic Regression": "Linear baseline; scaled features.",
            "Decision Tree": "Tree-based classifier; unscaled features.",
            "Random Forest": f"Ensemble classifier with {n_estimators} trees; unscaled features.",
            "K-Nearest Neighbors": "Distance-based classifier; scaled features.",
            "Naive Bayes": "Gaussian probabilistic classifier.",
            "SVM": "Support Vector Classifier with probability estimates; scaled features.",
        }
        for name, desc in model_info.items():
            with st.expander(name): st.write(desc)
    else:
        results = st.session_state.results_df
        st.success(f"✅ {len(st.session_state.trained_models)} models trained successfully.")
        for _, row in results.iterrows():
            with st.expander(f"📌 {row['Model']}"):
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Accuracy", f"{row['Accuracy']:.4f}")
                c2.metric("Precision", f"{row['Precision']:.4f}")
                c3.metric("Recall", f"{row['Recall']:.4f}")
                c4.metric("F1", f"{row['F1 Score']:.4f}")
                c5.metric("CV F1", f"{row['CV F1 Mean']:.4f} ± {row['CV F1 Std']:.4f}")
                st.write(f"TP: {row['TP']:,} | TN: {row['TN']:,} | FP: {row['FP']:,} | FN: {row['FN']:,}")
        if st.session_state.model_errors:
            st.subheader("Training Errors")
            for name, err in st.session_state.model_errors.items():
                st.error(f"**{name}:** {err}")

# TAB 5 -----------------------------------------------------------------------
with tab5:
    st.header("📈 Model Comparison")
    if not st.session_state.models_trained:
        st.info("Train models first to see the comparison.")
    else:
        results = st.session_state.results_df
        display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "CV F1 Mean", "CV F1 Std"]
        st.dataframe(results[display_cols], use_container_width=True, hide_index=True)
        st.caption("Models are ranked primarily by held-out test F1 Score, then cross-validation F1 and ROC-AUC.")

        metric = st.selectbox("Metric to visualize", ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "CV F1 Mean"])
        if metric == "ROC-AUC":
            valid = results.copy(); valid["ROC-AUC"] = pd.to_numeric(valid["ROC-AUC"], errors="coerce"); valid = valid.dropna(subset=["ROC-AUC"])
        else:
            valid = results
        fig = plot_metric_bar(valid, metric)
        st.pyplot(fig); plt.close(fig)

# TAB 6 -----------------------------------------------------------------------
with tab6:
    st.header("🏆 Best Model")
    if not st.session_state.models_trained:
        st.info("Train models first to see the best model.")
    else:
        best_name = st.session_state.best_model_name
        results = st.session_state.results_df
        best_row = results[results["Model"] == best_name].iloc[0]
        best_model = st.session_state.trained_models[best_name]
        best_preds = st.session_state.model_predictions[best_name]
        y_test = st.session_state.y_test

        st.markdown(f"""
        <div class="best-model-card">
            <h2>🏆 Best Model</h2>
            <h3>{best_name}</h3>
            <p>Selected using highest held-out F1 Score, with CV F1 and ROC-AUC as secondary ranking signals.</p>
            <div class="metric-row">
                <div class="metric-item"><div class="value">{best_row['F1 Score']:.4f}</div><div class="label">F1</div></div>
                <div class="metric-item"><div class="value">{best_row['Accuracy']:.4f}</div><div class="label">Accuracy</div></div>
                <div class="metric-item"><div class="value">{best_row['Precision']:.4f}</div><div class="label">Precision</div></div>
                <div class="metric-item"><div class="value">{best_row['Recall']:.4f}</div><div class="label">Recall</div></div>
                <div class="metric-item"><div class="value">{best_row['CV F1 Mean']:.4f}</div><div class="label">CV F1</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        cm = confusion_matrix(y_test, best_preds["y_pred"], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        st.subheader("Confusion Matrix")
        a, b = st.columns([2, 1])
        with a:
            fig = plot_confusion_matrix(cm, f"Confusion Matrix — {best_name}")
            st.pyplot(fig); plt.close(fig)
        with b:
            st.write(f"**TN:** {tn:,} — legitimate correctly detected")
            st.write(f"**FP:** {fp:,} — legitimate flagged as phishing")
            st.write(f"**FN:** {fn:,} — phishing missed")
            st.write(f"**TP:** {tp:,} — phishing correctly detected")

        if best_preds["y_prob"] is not None:
            st.subheader("ROC Curve")
            auc_value = float(best_row["ROC-AUC"]) if best_row["ROC-AUC"] != "N/A" else 0
            fig = plot_roc_curve(y_test, best_preds["y_prob"], best_name, auc_value)
            st.pyplot(fig); plt.close(fig)

            st.subheader("Precision-Recall Curve")
            fig = plot_precision_recall_curve(y_test, best_preds["y_prob"], best_name)
            st.pyplot(fig); plt.close(fig)

        st.subheader("Feature Importance / Coefficients")
        fig = plot_feature_importance(best_model, st.session_state.preprocessor["all_final_features"], best_name)
        if fig is not None:
            st.pyplot(fig); plt.close(fig)
        else:
            st.info("Direct feature importance is not available for this model.")

        st.subheader("Classification Report")
        st.code(classification_report(y_test, best_preds["y_pred"], target_names=["Legitimate (0)", "Phishing (1)"], zero_division=0), language="text")

        st.subheader("💾 Download Best Model")
        bundle = {
            "model": best_model,
            "preprocessor": st.session_state.preprocessor,
            "target_column": target_col,
            "feature_columns": st.session_state.preprocessor["feature_columns"],
            "best_model_name": best_name,
            "metrics": best_row.to_dict(),
        }
        buffer = io.BytesIO(); joblib.dump(bundle, buffer); buffer.seek(0)
        st.download_button("⬇️ Download Best Model (.joblib)", buffer, file_name=f"best_model_{best_name.lower().replace(' ', '_')}.joblib", mime="application/octet-stream", use_container_width=True)

# TAB 7 -----------------------------------------------------------------------
with tab7:
    st.header("🔮 Detect Website")
    if not st.session_state.models_trained:
        st.info("Train models first to enable detection.")
    else:
        best_name = st.session_state.best_model_name
        best_model = st.session_state.trained_models[best_name]
        pp = st.session_state.preprocessor
        st.write(f"Using **{best_name}** as the selected model.")

        mode = st.radio("Detection Mode", ["Single Prediction", "Batch CSV Prediction"], horizontal=True)

        if mode == "Single Prediction":
            st.markdown("### Enter Website Features")
            st.caption("The current model expects the feature columns from the uploaded training dataset. The URL itself is not automatically converted into features unless your dataset contains URL-derived features.")
            with st.form("prediction_form"):
                input_data = feature_input_form(df, pp["feature_columns"], key_prefix="single")
                submitted = st.form_submit_button("🔍 Analyze Website", type="primary", use_container_width=True)

            if submitted:
                try:
                    prediction, probability = predict_single_sample(input_data, pp, best_model, best_name)
                    phishing_prob = float(probability[1] * 100) if probability is not None else None
                    legitimate_prob = float(probability[0] * 100) if probability is not None else None
                    result_label = "PHISHING" if prediction == 1 else "LEGITIMATE"

                    if prediction == 1:
                        st.markdown('<div class="prediction-result phishing">🚨 Prediction: PHISHING</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="prediction-result legitimate">✅ Prediction: LEGITIMATE</div>', unsafe_allow_html=True)

                    if phishing_prob is not None:
                        risk, css = risk_level(phishing_prob)
                        st.markdown(f'<div class="{css}"><b>Model Risk Level: {risk}</b><br>Phishing probability: <b>{phishing_prob:.2f}%</b></div>', unsafe_allow_html=True)
                        a, b = st.columns(2)
                        a.metric("Phishing Probability", f"{phishing_prob:.2f}%")
                        b.metric("Legitimate Probability", f"{legitimate_prob:.2f}%")
                        st.progress(min(max(phishing_prob / 100, 0.0), 1.0), text=f"Phishing probability: {phishing_prob:.1f}%")

                    st.session_state.prediction_history.append({
                        "Result": result_label,
                        "Phishing Probability (%)": round(phishing_prob, 2) if phishing_prob is not None else None,
                        "Legitimate Probability (%)": round(legitimate_prob, 2) if legitimate_prob is not None else None,
                        "Model": best_name,
                    })
                    st.caption("This is a machine-learning prediction, not a guarantee that a website is safe or malicious.")
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        else:
            st.markdown("### Batch Prediction")
            batch_file = st.file_uploader("Upload CSV containing the same feature columns used for training", type=["csv"], key="batch_upload")
            if batch_file is not None:
                try:
                    batch_df = pd.read_csv(batch_file)
                    missing_cols = [c for c in pp["feature_columns"] if c not in batch_df.columns]
                    if missing_cols:
                        st.error(f"Missing required feature columns: {missing_cols}")
                    else:
                        rows = []
                        with st.spinner(f"Analyzing {len(batch_df):,} rows..."):
                            for idx, row in batch_df.iterrows():
                                prediction, probability = predict_single_sample(row[pp["feature_columns"]].to_dict(), pp, best_model, best_name)
                                rows.append({
                                    "Prediction": "PHISHING" if prediction == 1 else "LEGITIMATE",
                                    "Phishing Probability (%)": round(float(probability[1] * 100), 2) if probability is not None else None,
                                    "Legitimate Probability (%)": round(float(probability[0] * 100), 2) if probability is not None else None,
                                })
                        result_df = pd.concat([batch_df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
                        st.success(f"✅ Analyzed {len(result_df):,} rows.")
                        c1, c2 = st.columns(2)
                        c1.metric("Phishing", int((result_df["Prediction"] == "PHISHING").sum()))
                        c2.metric("Legitimate", int((result_df["Prediction"] == "LEGITIMATE").sum()))
                        st.dataframe(result_df, use_container_width=True, hide_index=True)
                        st.download_button("⬇️ Download Prediction Results", result_df.to_csv(index=False).encode("utf-8"), file_name="phishing_predictions.csv", mime="text/csv", use_container_width=True)
                except Exception as exc:
                    st.error(f"Batch prediction failed: {exc}")

# TAB 8 -----------------------------------------------------------------------
with tab8:
    st.header("📜 Prediction History")
    history = st.session_state.prediction_history
    if not history:
        st.info("No predictions have been made in this session yet.")
    else:
        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download History CSV", hist_df.to_csv(index=False).encode("utf-8"), file_name="prediction_history.csv", mime="text/csv", use_container_width=True)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#888;font-size:.85rem;">🔐 Phishing Website Detection Dashboard — Built with Streamlit & Scikit-learn</p>',
    unsafe_allow_html=True,
)
