# =========================
# IMPORTS
# =========================
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Phishing Website Detection",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# CUSTOM CSS
# =========================
st.markdown(
    """
<style>
    .main-title {
        text-align: center;
        padding: 0.5rem 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .best-model-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .best-model-card h2 {
        color: #ffd700;
        margin: 0;
    }
    .best-model-card h3 {
        color: #e0e0e0;
        margin: 0.3rem 0;
    }
    .metric-row {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .metric-item {
        text-align: center;
        padding: 0.5rem 1rem;
    }
    .metric-item .value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4fc3f7;
    }
    .metric-item .label {
        font-size: 0.85rem;
        color: #bbb;
    }
    .prediction-result {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .phishing {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #c62828;
    }
    .legitimate {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #2e7d32;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# HELPER FUNCTIONS
# =========================


def load_data(uploaded_file):
    """Load CSV data from uploaded file."""
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("The uploaded CSV file is empty.")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None


def get_dataset_info(df, target_col):
    """Return a dictionary of dataset information."""
    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numerical_cols": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_cols": list(df.select_dtypes(exclude=[np.number]).columns),
        "duplicates": int(df.duplicated().sum()),
        "missing_total": int(df.isnull().sum().sum()),
        "size_bytes": df.memory_usage(deep=True).sum(),
        "target_col": target_col,
        "feature_count": df.shape[1] - 1,
    }
    return info


def clean_data(df):
    """Remove duplicate rows from the dataset."""
    before = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    after = len(df_clean)
    return df_clean, before - after


def preprocess_data(df, target_col, test_size, random_state):
    """
    Full preprocessing: separate X/y, handle missing values,
    encode categoricals, split, scale. All fitted on training data only.
    Returns everything needed for evaluation and prediction.
    """
    # --- Step 1: Separate features and target ---
    X = df.drop(columns=[target_col])
    y = df[target_col]

    feature_columns = list(X.columns)

    # --- Step 2: Identify column types ---
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)

    # --- Step 3: Train/Test Split (before any fitting) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- Step 4: Handle missing values (fit on train only) ---
    num_imputer = None
    cat_imputer = None

    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
        X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    # --- Step 5: Encode categorical variables (fit on train only) ---
    ohe = None
    ohe_feature_names = []
    if cat_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded_train = ohe.fit_transform(X_train[cat_cols])
        encoded_test = ohe.transform(X_test[cat_cols])
        ohe_feature_names = list(ohe.get_feature_names_out(cat_cols))

        encoded_train_df = pd.DataFrame(
            encoded_train, columns=ohe_feature_names, index=X_train.index
        )
        encoded_test_df = pd.DataFrame(
            encoded_test, columns=ohe_feature_names, index=X_test.index
        )

        X_train = X_train.drop(columns=cat_cols).reset_index(drop=True)
        X_test = X_test.drop(columns=cat_cols).reset_index(drop=True)
        encoded_train_df = encoded_train_df.reset_index(drop=True)
        encoded_test_df = encoded_test_df.reset_index(drop=True)

        X_train = pd.concat([X_train, encoded_train_df], axis=1)
        X_test = pd.concat([X_test, encoded_test_df], axis=1)

    # --- Step 6: Feature scaling (fit on train only) ---
    scaler = StandardScaler()
    all_final_features = list(X_train.columns)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Also keep unscaled versions for tree-based models
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
    }

    return (
        X_train_scaled,
        X_test_scaled,
        X_train_unscaled,
        X_test_unscaled,
        y_train.values,
        y_test.values,
        preprocessor,
    )


def get_models(random_state):
    """Return a dictionary of model name -> model instance."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True, random_state=random_state),
    }
    return models


def needs_scaling(model_name):
    """Determine if a model benefits from feature scaling."""
    scaled_models = {"Logistic Regression", "K-Nearest Neighbors", "SVM"}
    return model_name in scaled_models


def train_models(
    models, X_train_scaled, X_test_scaled, X_train_unscaled, X_test_unscaled, y_train, y_test
):
    """
    Train all models, evaluate them, return results DataFrame,
    trained model dict, and per-model predictions.
    """
    results = []
    trained_models = {}
    model_predictions = {}
    errors = {}

    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(models)

    for i, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}... ({i + 1}/{total})")
        progress_bar.progress((i + 1) / total)

        try:
            # Select scaled or unscaled data
            if needs_scaling(name):
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train_unscaled, X_test_unscaled

            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)

            # Probability predictions
            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_te)[:, 1]

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            roc = np.nan
            if y_prob is not None:
                try:
                    roc = roc_auc_score(y_test, y_prob)
                except Exception:
                    roc = np.nan

            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            results.append(
                {
                    "Model": name,
                    "Accuracy": round(acc, 4),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "F1 Score": round(f1, 4),
                    "ROC-AUC": round(roc, 4) if not np.isnan(roc) else "N/A",
                    "TP": tp,
                    "TN": tn,
                    "FP": fp,
                    "FN": fn,
                }
            )
            trained_models[name] = model
            model_predictions[name] = {
                "y_pred": y_pred,
                "y_prob": y_prob,
            }

        except Exception as e:
            errors[name] = str(e)
            st.warning(f"⚠️ {name} could not be trained. Reason: {e}")

    progress_bar.empty()
    status_text.empty()

    results_df = pd.DataFrame(results)
    return results_df, trained_models, model_predictions, errors


def evaluate_model(y_test, y_pred, y_prob):
    """Return a dictionary of evaluation metrics for a single model."""
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = np.nan
    if y_prob is not None:
        try:
            roc = roc_auc_score(y_test, y_prob)
        except Exception:
            roc = np.nan
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def compare_models(results_df):
    """Sort results by F1 Score (primary) and ROC-AUC (secondary)."""
    df = results_df.copy()
    # Convert ROC-AUC to numeric for sorting; "N/A" becomes NaN
    df["ROC-AUC_num"] = pd.to_numeric(df["ROC-AUC"], errors="coerce")
    df = df.sort_values(
        by=["F1 Score", "ROC-AUC_num"], ascending=[False, False]
    ).reset_index(drop=True)
    df = df.drop(columns=["ROC-AUC_num"])
    return df


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Plot a styled confusion matrix using Matplotlib."""
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = LinearSegmentedColormap.from_list("custom", ["#e3f2fd", "#1565c0"])
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = ["Legitimate (0)", "Phishing (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual Label", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Annotate cells
    thresh = cm.max() / 2.0
    cell_labels = [
        ["TN", "FP"],
        ["FN", "TP"],
    ]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(
                j, i,
                f"{cell_labels[i][j]}\n{cm[i, j]:,}",
                ha="center", va="center",
                fontsize=14, fontweight="bold", color=color,
            )

    plt.tight_layout()
    return fig


def plot_roc_curve(y_test, y_prob, model_name, auc_val):
    """Plot ROC curve for a model."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1565c0", lw=2.5, label=f"{model_name} (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#bdbdbd", lw=1.5, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#1565c0")
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_precision_recall_curve(y_test, y_prob, model_name):
    """Plot Precision-Recall curve."""
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec_vals, prec_vals, color="#c62828", lw=2.5, label=model_name)
    ax.fill_between(rec_vals, prec_vals, alpha=0.1, color="#c62828")
    ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, model_name, top_n=15):
    """Plot feature importance for tree-based or linear models."""
    importances = None
    label = "Importance"

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        label = "Feature Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        label = "Absolute Coefficient"

    if importances is None:
        return None

    # Match lengths (in case encoding changed count)
    n = min(len(importances), len(feature_names))
    importances = importances[:n]
    feature_names = feature_names[:n]

    indices = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.4)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    ax.barh(range(len(top_features)), top_importances, color=colors, edgecolor="none")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=10)
    ax.set_xlabel(label, fontsize=12, fontweight="bold")
    ax.set_title(f"Top {len(top_features)} Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_model_comparison_bar(results_df, metric, color):
    """Bar chart comparing all models on a single metric."""
    fig, ax = plt.subplots(figsize=(9, 5))
    models = results_df["Model"]
    values = pd.to_numeric(results_df[metric], errors="coerce")

    bars = ax.bar(models, values, color=color, edgecolor="white", width=0.55)

    # Annotate bars
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    ax.set_ylabel(metric, fontsize=12, fontweight="bold")
    ax.set_title(f"Model Comparison — {metric}", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.tight_layout()
    return fig


def plot_combined_comparison(results_df):
    """Grouped bar chart comparing all metrics for all models."""
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    models = results_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.18
    colors = ["#1565c0", "#2e7d32", "#e65100", "#6a1b9a"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        vals = pd.to_numeric(results_df[metric], errors="coerce").values
        ax.bar(x + i * width, vals, width, label=metric, color=colors[i], edgecolor="white")

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Combined Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def predict_single_sample(input_data, preprocessor, model, model_name):
    """
    Preprocess a single input row and return prediction + probability.
    input_data: dict of feature_name -> value
    """
    input_df = pd.DataFrame([input_data])

    num_cols = preprocessor["num_cols"]
    cat_cols = preprocessor["cat_cols"]
    num_imputer = preprocessor["num_imputer"]
    cat_imputer = preprocessor["cat_imputer"]
    ohe = preprocessor["ohe"]
    scaler = preprocessor["scaler"]

    # Impute
    if num_cols and num_imputer is not None:
        cols_present = [c for c in num_cols if c in input_df.columns]
        if cols_present:
            input_df[cols_present] = num_imputer.transform(input_df[cols_present])

    if cat_cols and cat_imputer is not None:
        cols_present = [c for c in cat_cols if c in input_df.columns]
        if cols_present:
            input_df[cols_present] = cat_imputer.transform(input_df[cols_present])

    # Encode
    if cat_cols and ohe is not None:
        cols_present = [c for c in cat_cols if c in input_df.columns]
        if cols_present:
            encoded = ohe.transform(input_df[cols_present])
            ohe_names = list(ohe.get_feature_names_out(cols_present))
            encoded_df = pd.DataFrame(encoded, columns=ohe_names, index=input_df.index)
            input_df = input_df.drop(columns=cols_present)
            input_df = pd.concat([input_df, encoded_df], axis=1)

    # Ensure column order matches training
    all_final = preprocessor["all_final_features"]
    for col in all_final:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[all_final]

    # Scale or not depending on model
    if needs_scaling(model_name):
        input_arr = scaler.transform(input_df)
    else:
        input_arr = input_df.values

    prediction = model.predict(input_arr)[0]
    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_arr)[0]

    return prediction, probability


# =========================
# SESSION STATE DEFAULTS
# =========================
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
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/security-checked.png",
        width=64,
    )
    st.markdown("## 🔐 Phishing Detector")
    st.markdown("---")

    # Dataset upload
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV file", type=["csv"], label_visibility="collapsed"
    )

    if uploaded_file is not None and st.session_state.df is None:
        df_loaded = load_data(uploaded_file)
        if df_loaded is not None:
            st.session_state.df = df_loaded

    st.markdown("---")

    # Target column
    target_col = "Is_Phishing"
    if st.session_state.df is not None:
        cols = list(st.session_state.df.columns)
        default_idx = cols.index("Is_Phishing") if "Is_Phishing" in cols else 0
        target_col = st.selectbox("🎯 Select Target Column", cols, index=default_idx)
    else:
        st.selectbox("🎯 Select Target Column", ["Upload dataset first"], disabled=True)

    st.markdown("---")

    # Hyperparameters
    st.markdown("### ⚙️ Settings")
    test_size = st.slider("Test Size", 0.10, 0.40, 0.20, 0.05)
    random_state = st.number_input("Random State", 0, 1000, 42, step=1)

    st.markdown("---")

    # Action buttons
    st.markdown("### 🚀 Actions")
    btn_eda = st.button("📊 Run EDA", use_container_width=True)
    btn_train = st.button("🤖 Train Models", use_container_width=True)

    st.markdown("---")
    btn_reset = st.button("🔄 Reset Application", use_container_width=True)

    if btn_reset:
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.rerun()


# =========================
# MAIN CONTENT
# =========================
st.markdown('<h1 class="main-title">🔐 Phishing Website Detection Using Machine Learning</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">End-to-End Classification, Model Comparison &amp; Explainability Dashboard<br>'
    "<em>Dataset → EDA → Preprocessing → Model Training → Model Comparison → Best Model → Evaluation → Prediction</em></p>",
    unsafe_allow_html=True,
)

# Handle EDA and Train button presses
if btn_eda and st.session_state.df is not None:
    st.session_state.eda_done = True

if btn_train and st.session_state.df is not None:
    # Validate target
    if target_col not in st.session_state.df.columns:
        st.error(f"Target column '{target_col}' not found in the dataset.")
    elif st.session_state.df[target_col].nunique() < 2:
        st.error("Target column must contain at least two classes.")
    else:
        df_work = st.session_state.df.copy()
        df_work, dups_removed = clean_data(df_work)
        st.session_state.df_clean = df_work
        if dups_removed > 0:
            st.info(f"Removed {dups_removed:,} duplicate rows before training.")

        with st.spinner("Preprocessing data..."):
            (
                X_train_s, X_test_s,
                X_train_u, X_test_u,
                y_train, y_test,
                preprocessor,
            ) = preprocess_data(df_work, target_col, test_size, random_state)

        st.session_state.X_train_scaled = X_train_s
        st.session_state.X_test_scaled = X_test_s
        st.session_state.X_train_unscaled = X_train_u
        st.session_state.X_test_unscaled = X_test_u
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.preprocessor = preprocessor

        models = get_models(random_state)
        results_df, trained_models, model_predictions, errors = train_models(
            models,
            X_train_s, X_test_s,
            X_train_u, X_test_u,
            y_train, y_test,
        )

        if not results_df.empty:
            results_df = compare_models(results_df)
            best_name = results_df.iloc[0]["Model"]
            st.session_state.results_df = results_df
            st.session_state.trained_models = trained_models
            st.session_state.model_predictions = model_predictions
            st.session_state.model_errors = errors
            st.session_state.best_model_name = best_name
            st.session_state.models_trained = True
            st.success(f"✅ All models trained! Best model: **{best_name}**")
        else:
            st.error("No models could be trained successfully.")

# Guard: dataset required
if st.session_state.df is None:
    st.info("👈 Upload a CSV dataset using the sidebar to begin.")
    st.stop()

df = st.session_state.df

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "📋 Dataset Overview",
        "📊 EDA",
        "⚙️ Preprocessing",
        "🤖 Model Training",
        "📈 Model Comparison",
        "🏆 Best Model",
        "🔮 Prediction",
    ]
)


# =========================
# TAB 1 — DATASET OVERVIEW
# =========================
with tab1:
    st.header("Dataset Overview")

    info = get_dataset_info(df, target_col)

    # Metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{info['rows']:,}")
    c2.metric("Columns", f"{info['columns']}")
    c3.metric("Features", f"{info['feature_count']}")
    c4.metric("Missing Values", f"{info['missing_total']:,}")
    c5.metric("Duplicates", f"{info['duplicates']:,}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Dataset Size", f"{info['size_bytes'] / 1024:.1f} KB")
    with col_b:
        st.metric("Target Column", info["target_col"])

    st.markdown("---")

    # Preview
    st.subheader("Dataset Preview")
    n_rows = st.selectbox("Number of rows to display", [5, 10, 20], index=0)
    st.dataframe(df.head(n_rows), use_container_width=True)

    st.markdown("---")

    # Column info
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Column Data Types")
        dtype_df = pd.DataFrame(
            {"Column": df.columns, "Data Type": [str(dt) for dt in df.dtypes]}
        )
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Numerical vs Categorical")
        st.write(f"**Numerical columns ({len(info['numerical_cols'])}):** {', '.join(info['numerical_cols']) if info['numerical_cols'] else 'None'}")
        st.write(f"**Categorical columns ({len(info['categorical_cols'])}):** {', '.join(info['categorical_cols']) if info['categorical_cols'] else 'None'}")

    st.markdown("---")

    # Target distribution
    if target_col in df.columns:
        st.subheader("Target Distribution")
        tc = df[target_col].value_counts()
        c1, c2 = st.columns(2)
        for i, (val, cnt) in enumerate(tc.items()):
            label = "Legitimate" if val == 0 else "Phishing" if val == 1 else str(val)
            pct = cnt / len(df) * 100
            (c1 if i % 2 == 0 else c2).metric(label, f"{cnt:,} ({pct:.1f}%)")

    st.markdown("---")

    # Descriptive stats
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe().T, use_container_width=True)


# =========================
# TAB 2 — EDA
# =========================
with tab2:
    st.header("Exploratory Data Analysis")

    if not st.session_state.eda_done:
        st.info('Click **"📊 Run EDA"** in the sidebar to generate visualizations.')
    else:
        # 7.1 Target Distribution
        st.subheader("7.1 — Target Distribution")
        if target_col in df.columns:
            tc = df[target_col].value_counts().sort_index()
            labels = []
            for v in tc.index:
                if v == 0:
                    labels.append("Legitimate (0)")
                elif v == 1:
                    labels.append("Phishing (1)")
                else:
                    labels.append(str(v))

            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ["#2e7d32", "#c62828"] if len(tc) == 2 else plt.cm.Set2.colors[: len(tc)]
                bars = ax.bar(labels, tc.values, color=colors, edgecolor="white", width=0.5)
                for bar, val in zip(bars, tc.values):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                            f"{val:,}", ha="center", va="bottom", fontweight="bold")
                ax.set_ylabel("Count", fontweight="bold")
                ax.set_title("Target Distribution", fontweight="bold")
                ax.grid(True, axis="y", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            with col2:
                st.markdown("**Class Counts**")
                for lbl, val in zip(labels, tc.values):
                    pct = val / len(df) * 100
                    st.write(f"• **{lbl}:** {val:,} ({pct:.1f}%)")
        else:
            st.warning(f"Target column '{target_col}' not found.")

        st.markdown("---")

        # 7.2 Missing Values
        st.subheader("7.2 — Missing Values")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing Percentage (%)": missing_pct.values,
        })
        missing_df = missing_df[missing_df["Missing Count"] > 0].reset_index(drop=True)

        if missing_df.empty:
            st.success("✅ No missing values detected.")
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
            with col2:
                fig, ax = plt.subplots(figsize=(7, max(3, len(missing_df) * 0.4)))
                ax.barh(missing_df["Column"], missing_df["Missing Count"], color="#e65100")
                ax.set_xlabel("Missing Count", fontweight="bold")
                ax.set_title("Missing Values per Column", fontweight="bold")
                ax.grid(True, axis="x", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        st.markdown("---")

        # 7.3 Duplicate Rows
        st.subheader("7.3 — Duplicate Rows")
        dup_count = int(df.duplicated().sum())
        st.metric("Number of Duplicate Rows", f"{dup_count:,}")

        st.markdown("---")

        # 7.4 Feature Distributions
        st.subheader("7.4 — Feature Distributions")
        num_features = list(df.select_dtypes(include=[np.number]).columns)
        if target_col in num_features:
            num_features.remove(target_col)

        if num_features:
            selected_feat = st.selectbox("Select a feature for histogram", num_features, key="hist_feat")
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.hist(df[selected_feat].dropna(), bins=40, color="#1565c0", edgecolor="white", alpha=0.85)
                ax.set_xlabel(selected_feat, fontweight="bold")
                ax.set_ylabel("Frequency", fontweight="bold")
                ax.set_title(f"Distribution of {selected_feat}", fontweight="bold")
                ax.grid(True, axis="y", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            with col2:
                st.markdown("**Distribution Info**")
                desc = df[selected_feat].describe()
                for stat in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
                    st.write(f"• **{stat}:** {desc[stat]:.4f}")
        else:
            st.info("No numerical features available for histograms.")

        st.markdown("---")

        # 7.5 Box Plot
        st.subheader("7.5 — Box Plot")
        if num_features:
            selected_box = st.selectbox("Select a feature for box plot", num_features, key="box_feat")
            fig, ax = plt.subplots(figsize=(7, 4))
            bp = ax.boxplot(df[selected_box].dropna(), vert=False, patch_artist=True,
                            boxprops=dict(facecolor="#bbdefb", edgecolor="#1565c0"),
                            medianprops=dict(color="#c62828", linewidth=2),
                            whiskerprops=dict(color="#1565c0"),
                            capprops=dict(color="#1565c0"),
                            flierprops=dict(marker="o", markerfacecolor="#e65100", markersize=4, alpha=0.5))
            ax.set_xlabel(selected_box, fontweight="bold")
            ax.set_title(f"Box Plot — {selected_box}", fontweight="bold")
            ax.grid(True, axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Outlier info
            q1 = df[selected_box].quantile(0.25)
            q3 = df[selected_box].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = df[(df[selected_box] < lower) | (df[selected_box] > upper)]
            if len(outliers) > 0:
                st.warning(f"⚠️ **{len(outliers):,}** potential outliers detected (IQR method) in **{selected_box}**.")
            else:
                st.success(f"✅ No obvious outliers detected in **{selected_box}** (IQR method).")

        st.markdown("---")

        # 7.6 Correlation Heatmap
        st.subheader("7.6 — Correlation Heatmap")
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(12, 10))
        cmap = LinearSegmentedColormap.from_list("rg", ["#c62828", "#ffffff", "#1565c0"])
        im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)

        # Annotate with values
        for i in range(len(corr)):
            for j in range(len(corr)):
                val = corr.iloc[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Find strong correlations
        with st.expander("🔍 Strongly Correlated Feature Pairs (|r| > 0.7)"):
            strong = []
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if abs(val) > 0.7:
                        strong.append((corr.columns[i], corr.columns[j], round(val, 4)))
            if strong:
                for f1, f2, r in sorted(strong, key=lambda x: abs(x[2]), reverse=True):
                    st.write(f"• **{f1}** ↔ **{f2}**: r = {r}")
            else:
                st.write("No strongly correlated pairs found (|r| > 0.7).")

        st.markdown("---")

        # 7.7 Feature vs Target
        st.subheader("7.7 — Feature vs Target Analysis")
        if target_col in df.columns and num_features:
            feat_vs_target = st.selectbox("Select feature to compare against target", num_features, key="feat_target")
            classes = sorted(df[target_col].unique())
            class_data = [df[df[target_col] == c][feat_vs_target].dropna().values for c in classes]
            class_labels = []
            for c in classes:
                if c == 0:
                    class_labels.append("Legitimate (0)")
                elif c == 1:
                    class_labels.append("Phishing (1)")
                else:
                    class_labels.append(str(c))

            fig, ax = plt.subplots(figsize=(7, 5))
            bp = ax.boxplot(class_data, labels=class_labels, patch_artist=True,
                            medianprops=dict(color="#c62828", linewidth=2),
                            whiskerprops=dict(color="#555"),
                            capprops=dict(color="#555"),
                            flierprops=dict(marker="o", markersize=3, alpha=0.4))
            colors_box = ["#a5d6a7", "#ef9a9a"]
            for patch, color in zip(bp["boxes"], colors_box[: len(bp["boxes"])]):
                patch.set_facecolor(color)
            ax.set_ylabel(feat_vs_target, fontweight="bold")
            ax.set_title(f"{feat_vs_target} — by Target Class", fontweight="bold")
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Show means
            for lbl, data in zip(class_labels, class_data):
                if len(data) > 0:
                    st.write(f"• **{lbl}** — Mean: {data.mean():.4f}, Median: {np.median(data):.4f}")
        else:
            st.info("Target column or numerical features not available.")


# =========================
# TAB 3 — PREPROCESSING
# =========================
with tab3:
    st.header("Preprocessing Pipeline")

    if not st.session_state.models_trained:
        st.info('Click **"🤖 Train Models"** in the sidebar. Preprocessing runs automatically before training.')
        st.markdown("### Preprocessing Steps Applied")
        st.markdown("""
1. **Separate Features & Target** — Split into X (features) and y (target).
2. **Remove Duplicates** — Duplicate rows are dropped.
3. **Train/Test Split** — Stratified split to preserve class balance. **Split happens before fitting any preprocessor** to prevent data leakage.
4. **Handle Missing Values** — Median imputation for numerical columns, most-frequent for categorical columns. **Fitted on training data only.**
5. **Encode Categoricals** — OneHotEncoder with `handle_unknown='ignore'`. **Fitted on training data only.**
6. **Feature Scaling** — StandardScaler applied to models that need it (Logistic Regression, KNN, SVM). **Fitted on training data only.**
        """)
        st.warning("⚠️ **Data Leakage Prevention**: All preprocessing statistics (medians, encoders, scalers) are learned exclusively from the training set and then applied to the test set.")
    else:
        pp = st.session_state.preprocessor
        st.success("✅ Preprocessing completed successfully.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Training Samples", f"{len(st.session_state.y_train):,}")
        col2.metric("Test Samples", f"{len(st.session_state.y_test):,}")
        col3.metric("Final Features", f"{len(pp['all_final_features'])}")

        c1, c2 = st.columns(2)
        c1.metric("Test Size", f"{test_size:.0%}")
        c2.metric("Random State", f"{random_state}")

        st.markdown("---")
        st.subheader("Pipeline Summary")

        steps = [
            ("1. Separate Features & Target", f"X = {len(pp['feature_columns'])} features, y = `{target_col}`"),
            ("2. Remove Duplicates", "Duplicates removed before splitting."),
            ("3. Train/Test Split", f"Stratified split — {1 - test_size:.0%} train / {test_size:.0%} test."),
            ("4. Missing Value Imputation", f"Numerical ({len(pp['num_cols'])} cols): Median | Categorical ({len(pp['cat_cols'])} cols): Most Frequent"),
            ("5. Categorical Encoding", f"OneHotEncoder on {len(pp['cat_cols'])} categorical columns → {len(pp['ohe_feature_names'])} encoded features" if pp['cat_cols'] else "No categorical columns — encoding skipped."),
            ("6. Feature Scaling", "StandardScaler applied for Logistic Regression, KNN, SVM. Tree-based models use unscaled data."),
        ]
        for title, desc in steps:
            with st.expander(title):
                st.write(desc)

        st.markdown("---")
        st.subheader("Final Feature List")
        feat_df = pd.DataFrame({"#": range(1, len(pp['all_final_features']) + 1), "Feature": pp['all_final_features']})
        st.dataframe(feat_df, use_container_width=True, hide_index=True)


# =========================
# TAB 4 — MODEL TRAINING
# =========================
with tab4:
    st.header("Model Training")

    if not st.session_state.models_trained:
        st.info('Click **"🤖 Train Models"** in the sidebar to start training.')
        st.markdown("### Models to be Trained")
        model_info = {
            "Logistic Regression": "Linear model, max_iter=1000. Uses scaled data.",
            "Decision Tree": "Tree-based model, random_state=42. Uses unscaled data.",
            "Random Forest": "Ensemble of 200 trees, random_state=42. Uses unscaled data.",
            "K-Nearest Neighbors": "Instance-based learning, default k=5. Uses scaled data.",
            "Naive Bayes": "Gaussian Naive Bayes. Uses unscaled data.",
            "SVM": "Support Vector Classifier with probability=True, random_state=42. Uses scaled data.",
        }
        for name, desc in model_info.items():
            with st.expander(name):
                st.write(desc)
    else:
        st.success("✅ Models trained successfully!")
        results = st.session_state.results_df

        # Per-model results
        for _, row in results.iterrows():
            name = row["Model"]
            with st.expander(f"📌 {name}", expanded=False):
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Accuracy", f"{row['Accuracy']:.4f}")
                c2.metric("Precision", f"{row['Precision']:.4f}")
                c3.metric("Recall", f"{row['Recall']:.4f}")
                c4.metric("F1 Score", f"{row['F1 Score']:.4f}")
                c5.metric("ROC-AUC", f"{row['ROC-AUC']}")

                st.write(f"**TP:** {row['TP']:,} | **TN:** {row['TN']:,} | **FP:** {row['FP']:,} | **FN:** {row['FN']:,}")

        # Show errors
        if st.session_state.model_errors:
            st.markdown("---")
            st.subheader("⚠️ Training Errors")
            for name, err in st.session_state.model_errors.items():
                st.error(f"**{name}** could not be trained. Reason: {err}")


# =========================
# TAB 5 — MODEL COMPARISON
# =========================
with tab5:
    st.header("Model Comparison")

    if not st.session_state.models_trained:
        st.info("Train models first to see comparisons.")
    else:
        results = st.session_state.results_df

        st.subheader("Comparison Table (sorted by F1 Score)")
        display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        st.dataframe(
            results[display_cols].style.highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1 Score"],
                color="#c8e6c9",
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        # Individual metric bar charts
        st.subheader("Per-Metric Comparisons")
        metrics_colors = {
            "Accuracy": "#1565c0",
            "Precision": "#2e7d32",
            "Recall": "#e65100",
            "F1 Score": "#6a1b9a",
            "ROC-AUC": "#00838f",
        }

        chart_cols = st.columns(2)
        for i, (metric, color) in enumerate(metrics_colors.items()):
            with chart_cols[i % 2]:
                fig = plot_model_comparison_bar(results, metric, color)
                st.pyplot(fig)
                plt.close(fig)

        st.markdown("---")

        # Combined comparison
        st.subheader("Combined Comparison")
        fig = plot_combined_comparison(results)
        st.pyplot(fig)
        plt.close(fig)


# =========================
# TAB 6 — BEST MODEL
# =========================
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

        # Best model card
        st.markdown(
            f"""
        <div class="best-model-card">
            <h2>🏆 Best Model</h2>
            <h3>{best_name}</h3>
            <p style="color:#aaa;">Selected based on highest F1 Score</p>
            <div class="metric-row">
                <div class="metric-item">
                    <div class="value">{best_row['F1 Score']:.4f}</div>
                    <div class="label">F1 Score</div>
                </div>
                <div class="metric-item">
                    <div class="value">{best_row['Accuracy']:.4f}</div>
                    <div class="label">Accuracy</div>
                </div>
                <div class="metric-item">
                    <div class="value">{best_row['Precision']:.4f}</div>
                    <div class="label">Precision</div>
                </div>
                <div class="metric-item">
                    <div class="value">{best_row['Recall']:.4f}</div>
                    <div class="label">Recall</div>
                </div>
                <div class="metric-item">
                    <div class="value">{best_row['ROC-AUC']}</div>
                    <div class="label">ROC-AUC</div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("")

        # Confusion Matrix explanation
        cm = confusion_matrix(y_test, best_preds["y_pred"])
        tn, fp, fn, tp = cm.ravel()

        st.markdown("---")

        # 14.1 Confusion Matrix
        st.subheader("14.1 — Confusion Matrix")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = plot_confusion_matrix(cm, title=f"Confusion Matrix — {best_name}")
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            st.markdown("**Interpretation**")
            st.write(f"• **True Negatives (TN):** {tn:,} — Correctly predicted Legitimate")
            st.write(f"• **False Positives (FP):** {fp:,} — Legitimate misclassified as Phishing")
            st.write(f"• **False Negatives (FN):** {fn:,} — Phishing misclassified as Legitimate")
            st.write(f"• **True Positives (TP):** {tp:,} — Correctly predicted Phishing")

        st.markdown("---")

        # 14.2 ROC Curve
        st.subheader("14.2 — ROC Curve")
        if best_preds["y_prob"] is not None:
            roc_val = best_row["ROC-AUC"]
            roc_val_num = float(roc_val) if roc_val != "N/A" else 0
            fig = plot_roc_curve(y_test, best_preds["y_prob"], best_name, roc_val_num)
            st.pyplot(fig)
            plt.close(fig)
            st.caption("The ROC curve shows the trade-off between True Positive Rate and False Positive Rate. A curve closer to the top-left corner indicates better performance.")
        else:
            st.warning("ROC curve unavailable — model does not support probability predictions.")

        st.markdown("---")

        # 14.3 Precision-Recall Curve
        st.subheader("14.3 — Precision-Recall Curve")
        if best_preds["y_prob"] is not None:
            fig = plot_precision_recall_curve(y_test, best_preds["y_prob"], best_name)
            st.pyplot(fig)
            plt.close(fig)
            st.caption("For phishing detection, the Precision-Recall curve is especially important because it shows how well the model balances identifying phishing sites (recall) without too many false alarms (precision).")
        else:
            st.warning("Precision-Recall curve unavailable — model does not support probability predictions.")

        st.markdown("---")

        # 14.4 Feature Importance
        st.subheader("14.4 — Feature Importance")
        pp = st.session_state.preprocessor
        fig = plot_feature_importance(best_model, pp["all_final_features"], best_name)
        if fig is not None:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(f"Feature importance is not directly available for **{best_name}**.")

        st.markdown("---")

        # 14.5 Classification Report
        st.subheader("Classification Report")
        report_str = classification_report(
            y_test, best_preds["y_pred"],
            target_names=["Legitimate (0)", "Phishing (1)"],
            zero_division=0,
        )
        st.code(report_str, language="text")

        st.markdown("---")

        # Download best model
        st.subheader("💾 Download Best Model")
        model_bundle = {
            "model": best_model,
            "preprocessor": pp,
            "target_column": target_col,
            "feature_columns": pp["feature_columns"],
            "best_model_name": best_name,
            "metrics": {
                "accuracy": best_row["Accuracy"],
                "precision": best_row["Precision"],
                "recall": best_row["Recall"],
                "f1_score": best_row["F1 Score"],
                "roc_auc": best_row["ROC-AUC"],
            },
        }
        buffer = io.BytesIO()
        joblib.dump(model_bundle, buffer)
        buffer.seek(0)

        st.download_button(
            label="⬇️ Download Best Model (.joblib)",
            data=buffer,
            file_name=f"best_model_{best_name.lower().replace(' ', '_')}.joblib",
            mime="application/octet-stream",
            use_container_width=True,
        )
        st.caption("The download includes the trained model, preprocessor, feature names, and evaluation metrics.")


# =========================
# TAB 7 — PREDICTION
# =========================
with tab7:
    st.header("🔮 Predict Website")

    if not st.session_state.models_trained:
        st.info("Train models first to make predictions.")
    else:
        best_name = st.session_state.best_model_name
        best_model = st.session_state.trained_models[best_name]
        pp = st.session_state.preprocessor

        st.write(f"Using the best model: **{best_name}**")
        st.markdown("Enter feature values for the website you want to classify:")

        feature_cols = pp["feature_columns"]

        # Determine binary features (only 0 and 1 values)
        binary_features = set()
        if st.session_state.df is not None:
            for col in feature_cols:
                if col in st.session_state.df.columns:
                    unique_vals = st.session_state.df[col].dropna().unique()
                    if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                        binary_features.add(col)

        # Create input form
        input_data = {}
        n_cols = 3
        cols = st.columns(n_cols)

        for i, feat in enumerate(feature_cols):
            col_idx = i % n_cols
            with cols[col_idx]:
                if feat in binary_features:
                    val = st.selectbox(
                        feat,
                        options=[0, 1],
                        index=0,
                        key=f"pred_{feat}",
                    )
                    input_data[feat] = val
                else:
                    # Get reasonable defaults from the dataset
                    if st.session_state.df is not None and feat in st.session_state.df.columns:
                        col_data = st.session_state.df[feat].dropna()
                        default_val = float(col_data.median()) if len(col_data) > 0 else 0.0
                        min_val = float(col_data.min()) if len(col_data) > 0 else 0.0
                        max_val = float(col_data.max()) if len(col_data) > 0 else 1000.0
                    else:
                        default_val = 0.0
                        min_val = 0.0
                        max_val = 1000.0

                    val = st.number_input(
                        feat,
                        value=default_val,
                        key=f"pred_{feat}",
                    )
                    input_data[feat] = val

        st.markdown("---")

        if st.button("🔍 Predict", use_container_width=True, type="primary"):
            try:
                prediction, probability = predict_single_sample(
                    input_data, pp, best_model, best_name
                )

                st.markdown("### Prediction Result")

                if prediction == 1:
                    st.markdown(
                        '<div class="prediction-result phishing">🚨 Prediction: PHISHING</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="prediction-result legitimate">✅ Prediction: LEGITIMATE</div>',
                        unsafe_allow_html=True,
                    )

                if probability is not None:
                    col1, col2 = st.columns(2)
                    phishing_prob = probability[1] * 100
                    legit_prob = probability[0] * 100
                    col1.metric("Phishing Probability", f"{phishing_prob:.2f}%")
                    col2.metric("Legitimate Probability", f"{legit_prob:.2f}%")

                    # Visual bar
                    fig, ax = plt.subplots(figsize=(8, 1.5))
                    ax.barh([""], [legit_prob], color="#2e7d32", label=f"Legitimate ({legit_prob:.1f}%)")
                    ax.barh([""], [phishing_prob], left=[legit_prob], color="#c62828", label=f"Phishing ({phishing_prob:.1f}%)")
                    ax.set_xlim(0, 100)
                    ax.set_xlabel("Probability (%)", fontweight="bold")
                    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.85rem;'>"
    "🔐 Phishing Website Detection Dashboard — Built with Streamlit & Scikit-learn"
    "</p>",
    unsafe_allow_html=True,
)
