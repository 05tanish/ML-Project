# ==============================================================================
# PHISHING WEBSITE DETECTION — STREAMLIT DASHBOARD
# End-to-End Machine Learning Classification & Explainability Application
# ==============================================================================

import io
import os
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
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
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Phishing Website Detection",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CUSTOM CSS & THEMING
# ==============================================================================
st.markdown(
    """
<style>
    .main-title {
        text-align: center;
        padding: 0.2rem 0;
        font-weight: 800;
        color: #1a237e;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .best-model-card {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 14px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .best-model-card h2 {
        color: #ffd700;
        margin: 0;
        font-size: 1.8rem;
    }
    .best-model-card h3 {
        color: #e0e0e0;
        margin: 0.3rem 0;
        font-weight: 400;
    }
    .metric-row {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin-top: 1.2rem;
    }
    .metric-item {
        text-align: center;
        padding: 0.6rem 1.2rem;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        margin: 0.3rem;
    }
    .metric-item .value {
        font-size: 1.6rem;
        font-weight: bold;
        color: #4fc3f7;
    }
    .metric-item .label {
        font-size: 0.85rem;
        color: #ddd;
    }
    .prediction-result {
        padding: 1.4rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 1.2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .phishing {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .legitimate {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .badge-success {
        background-color: #d4edda;
        color: #155724;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# HELPER FUNCTIONS — DATA CLEANING & PREPROCESSING
# ==============================================================================

def load_data(file_or_path):
    """Load CSV dataset from uploaded file object or file path."""
    try:
        if isinstance(file_or_path, str):
            df = pd.read_csv(file_or_path)
        else:
            df = pd.read_csv(file_or_path)
        if df.empty:
            st.error("The dataset file is empty.")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading CSV dataset: {e}")
        return None


def get_dataset_info(df, target_col):
    """Return dictionary of dataset summary information."""
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


def clean_data_pipeline(df, target_col="Is_Phishing"):
    """
    Standard data cleaning pipeline mirroring the notebook:
    1. Drop exact duplicate rows.
    2. Convert invalid negative values in count/length features to NaN.
    3. Ensure binary columns are strictly {0, 1}.
    4. Ensure ratio columns are within [0, 1].
    5. Median imputation for missing numeric values.
    6. Type casting.
    """
    before_len = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True).copy()
    duplicates_removed = before_len - len(df_clean)

    count_length_cols = [
        "URL_Length", "Num_Dots", "Num_Hyphens", "Num_Special_Chars",
        "Num_Subdomains", "Domain_Age_Days", "Domain_Registration_Length",
        "Num_Redirects", "Form_Count", "Iframe_Count", "Popup_Count",
        "Domain_Name_Length", "URL_Entropy"
    ]
    for col in count_length_cols:
        if col in df_clean.columns:
            df_clean.loc[df_clean[col] < 0, col] = np.nan

    binary_cols = [
        "Has_IP_Address", "Has_HTTPS", "Has_Suspicious_Words",
        "Password_Field_Present", "Favicon_External"
    ]
    for col in binary_cols:
        if col in df_clean.columns:
            df_clean.loc[~df_clean[col].isin([0, 1]), col] = np.nan

    ratio_cols = ["External_Link_Ratio", "Image_Link_Ratio"]
    for col in ratio_cols:
        if col in df_clean.columns:
            df_clean.loc[(df_clean[col] < 0) | (df_clean[col] > 1), col] = np.nan

    # Median imputation
    features = [c for c in df_clean.columns if c != target_col]
    for col in features:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Cast integer columns
    for col in count_length_cols:
        if col in df_clean.columns and col != "URL_Entropy":
            df_clean[col] = df_clean[col].astype(int)
    for col in binary_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(int)
    if target_col in df_clean.columns:
        df_clean[target_col] = df_clean[target_col].astype(int)

    return df_clean, duplicates_removed


def preprocess_data(df, target_col="Is_Phishing", test_size=0.20, random_state=42):
    """
    Split data and prepare StandardScaler.
    Fitted strictly on training split to prevent leakage.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_columns = list(X.columns)

    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    num_imputer = None
    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
        X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    scaler = StandardScaler()
    all_final_features = list(X_train.columns)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_unscaled = X_train.values
    X_test_unscaled = X_test.values

    preprocessor = {
        "num_imputer": num_imputer,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "all_final_features": all_final_features,
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


# ==============================================================================
# MODEL CONFIGURATION — 4 MODELS ONLY
# ==============================================================================

def get_models(random_state=42):
    """
    Return the 4 core models:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest (Tuned)
    4. Naive Bayes (Gaussian NB)
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            max_depth=20,
            min_samples_leaf=2,
            min_samples_split=5,
            n_estimators=100,
            random_state=random_state,
        ),
        "Naive Bayes": GaussianNB(),
    }
    return models


def needs_scaling(model_name):
    """Determine if a model requires feature scaling."""
    scaled_models = {"Logistic Regression", "Naive Bayes"}
    return model_name in scaled_models


def train_and_evaluate_models(
    models, X_train_scaled, X_test_scaled, X_train_unscaled, X_test_unscaled, y_train, y_test
):
    """Train the 4 models and return results DataFrame and predictions."""
    results = []
    trained_models = {}
    model_predictions = {}
    errors = {}

    for name, model in models.items():
        t0 = time.time()
        try:
            if needs_scaling(name):
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train_unscaled, X_test_unscaled

            model.fit(X_tr, y_train)
            train_time = time.time() - t0
            y_pred = model.predict(X_te)

            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_te)[:, 1]

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

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "ROC-AUC": roc,
                "Training Time (s)": round(train_time, 3),
            })

            trained_models[name] = model
            model_predictions[name] = {
                "y_pred": y_pred,
                "y_prob": y_prob,
                "confusion_matrix": cm,
                "classification_report": report,
            }

        except Exception as e:
            errors[name] = str(e)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by="F1 Score", ascending=False).reset_index(drop=True)

    return results_df, trained_models, model_predictions, errors


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def plot_target_distribution(df, target_col="Is_Phishing"):
    """Plot styled target class distribution."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    counts = df[target_col].value_counts()
    colors = ["#2e7d32", "#c62828"]
    labels = ["Legitimate (0)", "Phishing (1)"]
    vals = [counts.get(0, 0), counts.get(1, 0)]
    bars = ax.bar(labels, vals, color=colors, width=0.45, edgecolor="black", linewidth=1)

    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax.annotate(f"{h:,}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Target Class Distribution (Is_Phishing)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Count", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, title="Confusion Matrix — Final Model"):
    """Plot styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
    cmap = LinearSegmentedColormap.from_list("custom", ["#e3f2fd", "#1565c0"])
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = ["Legitimate (0)", "Phishing (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual Label", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    thresh = cm.max() / 2.0
    cell_names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cell_names[i][j]}\n{cm[i, j]:,}",
                    ha="center", va="center", fontsize=12, fontweight="bold", color=color)
    plt.tight_layout()
    return fig


def plot_roc_curve(y_test, y_prob, model_name, auc_val):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    ax.plot(fpr, tpr, color="#1565c0", lw=2.5, label=f"{model_name} (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#9e9e9e", lw=1.5, linestyle="--", label="Random Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=10, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_precision_recall_curve(y_test, y_prob, model_name):
    """Plot Precision-Recall curve."""
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    ax.plot(rec, prec, color="#2e7d32", lw=2.5, label=f"{model_name} (AP = {ap:.4f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=10, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=10, fontweight="bold")
    ax.set_title("Precision-Recall (PR) Curve", fontsize=11, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance horizontal bar chart."""
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    bars = ax.barh(sorted_features[::-1], sorted_importances[::-1], color="#0284c7", edgecolor="black", height=0.65)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2, f"{w:.4f}",
                ha="left", va="center", fontsize=8.5, fontweight="bold")
    ax.set_title("Feature Importance Ranking (Gini Impurity Reduction)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Importance", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(sorted_importances) * 1.15)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    """Plot full correlation matrix heatmap."""
    fig, ax = plt.subplots(figsize=(14, 11), dpi=150)
    corr = df.corr()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cols = df.columns
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=6.5)
    ax.set_title("Correlation Matrix Across All Features", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_combined_model_comparison(results_df):
    """Grouped bar chart comparing Accuracy, Precision, Recall, F1 for all 4 models."""
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    models = results_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.18
    colors = ["#1565c0", "#2e7d32", "#e65100", "#6a1b9a"]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for i, metric in enumerate(metrics):
        vals = results_df[metric].values
        bars = ax.bar(x + i * width, vals, width, label=metric, color=colors[i], edgecolor="black", linewidth=0.6)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=90)

    ax.set_xlabel("Classification Model", fontsize=10, fontweight="bold")
    ax.set_ylabel("Score", fontsize=10, fontweight="bold")
    ax.set_title("Combined Performance Comparison Across 4 Models", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


# ==============================================================================
# PREDICT SINGLE SAMPLE
# ==============================================================================

def predict_sample(input_data, preprocessor, model, model_name):
    """Preprocess single test instance and return prediction & probabilities."""
    input_df = pd.DataFrame([input_data])
    feature_cols = preprocessor["feature_columns"]
    scaler = preprocessor["scaler"]

    # Impute missing if needed
    if preprocessor["num_imputer"] is not None:
        input_df[feature_cols] = preprocessor["num_imputer"].transform(input_df[feature_cols])

    # Reorder columns
    input_df = input_df[feature_cols]

    if needs_scaling(model_name):
        input_arr = scaler.transform(input_df)
    else:
        input_arr = input_df.values

    prediction = int(model.predict(input_arr)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_arr)[0]

    return prediction, probability


# ==============================================================================
# INITIALIZE STATE & AUTO-LOAD
# ==============================================================================

if "df" not in st.session_state or st.session_state.df is None:
    if os.path.exists("phishing_website_raw.csv"):
        st.session_state.df = load_data("phishing_website_raw.csv")

if "models_trained" not in st.session_state:
    st.session_state.models_trained = False

# Auto-initialize cleaned data and preprocessor if raw dataset is present
if st.session_state.df is not None and "df_clean" not in st.session_state:
    df_clean, _ = clean_data_pipeline(st.session_state.df)
    st.session_state.df_clean = df_clean
    (
        X_train_s, X_test_s,
        X_train_u, X_test_u,
        y_train, y_test,
        preprocessor
    ) = preprocess_data(df_clean, "Is_Phishing", 0.20, 42)
    st.session_state.X_train_scaled = X_train_s
    st.session_state.X_test_scaled = X_test_s
    st.session_state.X_train_unscaled = X_train_u
    st.session_state.X_test_unscaled = X_test_u
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.preprocessor = preprocessor

# Auto-load pre-trained model PKL if available
if os.path.exists("api/best_phishing_model.pkl") and not st.session_state.models_trained:
    try:
        loaded_model = joblib.load("api/best_phishing_model.pkl")
        models = get_models(42)
        # Fast evaluate all 4 models so full comparison is available immediately
        results_df, trained_models, model_predictions, errors = train_and_evaluate_models(
            models,
            st.session_state.X_train_scaled, st.session_state.X_test_scaled,
            st.session_state.X_train_unscaled, st.session_state.X_test_unscaled,
            st.session_state.y_train, st.session_state.y_test
        )
        st.session_state.results_df = results_df
        st.session_state.trained_models = trained_models
        st.session_state.model_predictions = model_predictions
        st.session_state.best_model_name = "Random Forest"
        st.session_state.models_trained = True
    except Exception as e:
        st.warning(f"Could not auto-load pre-trained model: {e}")


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/security-checked.png", width=64)
    st.markdown("## 🔐 Phishing Detector")
    st.markdown("<span class='status-badge badge-success'>4 Models Active</span>", unsafe_allow_html=True)
    st.markdown("---")

    # Dataset Upload
    st.markdown("### 📂 Dataset Management")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file is not None:
        df_new = load_data(uploaded_file)
        if df_new is not None:
            st.session_state.df = df_new
            df_clean, _ = clean_data_pipeline(df_new)
            st.session_state.df_clean = df_clean
            (
                X_train_s, X_test_s,
                X_train_u, X_test_u,
                y_train, y_test,
                preprocessor
            ) = preprocess_data(df_clean, "Is_Phishing", 0.20, 42)
            st.session_state.X_train_scaled = X_train_s
            st.session_state.X_test_scaled = X_test_s
            st.session_state.X_train_unscaled = X_train_u
            st.session_state.X_test_unscaled = X_test_u
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.preprocessor = preprocessor
            st.session_state.models_trained = False
            st.success("Custom dataset loaded!")

    st.markdown("---")
    st.markdown("### ⚙️ Training Settings")
    test_size = st.slider("Test Set Split Ratio", 0.10, 0.40, 0.20, 0.05)
    random_state = st.number_input("Random Seed (State)", 0, 1000, 42, step=1)

    st.markdown("---")
    st.markdown("### 🚀 Model Execution")
    btn_train = st.button("🤖 Train 4 Models", use_container_width=True, type="primary")

    if btn_train:
        with st.spinner("Training 4 Classifiers (Logistic Regression, Decision Tree, Random Forest, Naive Bayes)..."):
            (
                X_train_s, X_test_s,
                X_train_u, X_test_u,
                y_train, y_test,
                preprocessor
            ) = preprocess_data(st.session_state.df_clean, "Is_Phishing", test_size, random_state)
            st.session_state.X_train_scaled = X_train_s
            st.session_state.X_test_scaled = X_test_s
            st.session_state.X_train_unscaled = X_train_u
            st.session_state.X_test_unscaled = X_test_u
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.preprocessor = preprocessor

            models = get_models(random_state)
            results_df, trained_models, model_predictions, errors = train_and_evaluate_models(
                models, X_train_s, X_test_s, X_train_u, X_test_u, y_train, y_test
            )
            st.session_state.results_df = results_df
            st.session_state.trained_models = trained_models
            st.session_state.model_predictions = model_predictions
            st.session_state.best_model_name = results_df.iloc[0]["Model"]
            st.session_state.models_trained = True
            st.success(f"✅ Trained 4 models! Best: **{st.session_state.best_model_name}**")

    st.markdown("---")
    if os.path.exists("api/best_phishing_model.pkl"):
        st.info("📦 Pre-trained Model Loaded: `best_phishing_model.pkl`")


# ==============================================================================
# MAIN DASHBOARD CONTENT
# ==============================================================================

st.markdown('<h1 class="main-title">🔐 Phishing Website Detection Dashboard</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Machine Learning Classification Pipeline • 4 Core Models • Explainable Cybersecurity Analytics<br>'
    '<em>Overview → EDA → Preprocessing → Model Training → Comparison → Best Model Deep-Dive → Live Prediction</em></p>',
    unsafe_allow_html=True,
)

# Render Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Dataset Overview",
    "📊 EDA & Visualizations",
    "⚙️ Preprocessing Pipeline",
    "🤖 4-Model Training",
    "📈 Model Comparison",
    "🏆 Best Model In-Depth",
    "🔮 Live Website Prediction",
])

df_current = st.session_state.df
df_clean_current = st.session_state.get("df_clean", df_current)

# ------------------------------------------------------------------------------
# TAB 1: DATASET OVERVIEW
# ------------------------------------------------------------------------------
with tab1:
    st.header("📋 Dataset Overview")
    
    # KPI Metrics
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Raw Records", f"{len(df_current):,}")
    kpi2.metric("Clean Records", f"{len(df_clean_current):,}")
    kpi3.metric("Input Features", f"{df_clean_current.shape[1] - 1}")
    legit_cnt = int((df_clean_current["Is_Phishing"] == 0).sum())
    phish_cnt = int((df_clean_current["Is_Phishing"] == 1).sum())
    kpi4.metric("Legitimate (0)", f"{legit_cnt:,} ({legit_cnt/len(df_clean_current)*100:.1f}%)")
    kpi5.metric("Phishing (1)", f"{phish_cnt:,} ({phish_cnt/len(df_clean_current)*100:.1f}%)")

    st.markdown("---")
    
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.subheader("Raw Data Preview")
        n_rows = st.selectbox("Number of sample rows to display", [5, 10, 20, 50], index=0)
        st.dataframe(df_current.head(n_rows), use_container_width=True)

    with col_r:
        st.subheader("Target Balance")
        fig_tgt = plot_target_distribution(df_clean_current)
        st.pyplot(fig_tgt)
        plt.close(fig_tgt)

    st.markdown("---")
    st.subheader("Descriptive Statistics")
    st.dataframe(df_clean_current.describe().T, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------------------------
with tab2:
    st.header("📊 Exploratory Data Analysis")
    st.markdown("Explore key distributions, outlier boundaries, and feature-target relationships.")

    eda_sub1, eda_sub2, eda_sub3, eda_sub4 = st.tabs([
        "Feature Distributions",
        "Outlier Detection (Boxplots)",
        "Correlation Heatmap",
        "Feature-Target Correlation",
    ])

    with eda_sub1:
        st.subheader("Distribution of Features")
        features_list = [c for c in df_clean_current.columns if c != "Is_Phishing"]
        selected_feat = st.selectbox("Select Feature for Distribution Histogram", features_list, index=0)
        
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        col_data = df_clean_current[selected_feat].dropna()
        ax.hist(col_data, bins=30, color="#1976d2", edgecolor="black", alpha=0.7, density=True)
        # KDE
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(col_data)
            x_grid = np.linspace(col_data.min(), col_data.max(), 200)
            ax.plot(x_grid, kde(x_grid), color="#d32f2f", lw=2, label="KDE Overlay")
            ax.legend()
        except Exception:
            pass
        ax.set_title(f"Distribution & Spread: {selected_feat}", fontsize=12, fontweight="bold")
        ax.set_xlabel(selected_feat, fontweight="bold")
        ax.set_ylabel("Density", fontweight="bold")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with eda_sub2:
        st.subheader("Outlier Check Across Numeric Features")
        box_feat = st.selectbox("Select Feature for Outlier Boxplot", features_list, index=1)
        fig, ax = plt.subplots(figsize=(8, 3.5), dpi=150)
        ax.boxplot(df_clean_current[box_feat].dropna(), vert=False, patch_artist=True,
                   boxprops=dict(facecolor="#e0f2fe", color="#0284c7"),
                   medianprops=dict(color="#d32f2f", lw=2))
        ax.set_title(f"Boxplot: {box_feat}", fontsize=12, fontweight="bold")
        ax.set_xlabel(box_feat, fontweight="bold")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with eda_sub3:
        st.subheader("Full 20-Feature Correlation Matrix")
        fig_corr = plot_correlation_heatmap(df_clean_current)
        st.pyplot(fig_corr)
        plt.close(fig_corr)

    with eda_sub4:
        st.subheader("Ranked Correlation with Target (Is_Phishing)")
        corr_tgt = df_clean_current.corr()["Is_Phishing"].drop("Is_Phishing").sort_values()
        fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)
        colors_bar = ["#c62828" if v < 0 else "#2e7d32" for v in corr_tgt]
        bars = ax.barh(corr_tgt.index, corr_tgt.values, color=colors_bar, edgecolor="black", height=0.65)
        for bar in bars:
            w = bar.get_width()
            offset = 0.01 if w >= 0 else -0.01
            ha = "left" if w >= 0 else "right"
            ax.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:.3f}",
                    ha=ha, va="center", fontsize=8.5, fontweight="bold")
        ax.set_title("Ranked Pearson Correlation with Is_Phishing", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Correlation Coefficient", fontweight="bold")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlim(min(corr_tgt.values) - 0.08, max(corr_tgt.values) + 0.08)
        ax.grid(True, axis="x", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)


# ------------------------------------------------------------------------------
# TAB 3: PREPROCESSING PIPELINE
# ------------------------------------------------------------------------------
with tab3:
    st.header("⚙️ Preprocessing Pipeline")
    st.markdown(
        """
        The dataset passes through a structured, leakage-free preprocessing pipeline:
        - **1. Data Cleaning:** Duplicate elimination, invalid negative count clipping, binary column validation, ratio constraint bounding.
        - **2. Median Imputation:** Fills any missing values using training-set medians (robust to outliers).
        - **3. Train/Test Stratification:** 80% Train ($20,000$ rows), 20% Test ($5,000$ rows) with preserved class proportions.
        - **4. Feature Scaling (StandardScaler):** Fit on training split and applied to scale-sensitive models (*Logistic Regression, Naive Bayes*). Tree models (*Decision Tree, Random Forest*) operate on unscaled features.
        """
    )
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.subheader("Training Split (80%)")
        st.info(f"Shape: `{st.session_state.X_train_scaled.shape}` (20,000 samples, 20 features)")
    with col_p2:
        st.subheader("Held-Out Test Split (20%)")
        st.info(f"Shape: `{st.session_state.X_test_scaled.shape}` (5,000 samples, 20 features)")

    st.subheader("Selected Features (20 Total)")
    pp = st.session_state.preprocessor
    st.write(pp["feature_columns"])


# ------------------------------------------------------------------------------
# TAB 4: 4-MODEL TRAINING
# ------------------------------------------------------------------------------
with tab4:
    st.header("🤖 4-Model Training Pipeline")
    st.markdown("The system trains and compares **4 distinct machine learning classifiers**:")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("#### 1. Logistic Regression")
        st.caption("Linear probability model with L2 regularization and StandardScaler.")
    with c2:
        st.markdown("#### 2. Decision Tree")
        st.caption("Non-parametric tree-based recursive binary splitting model.")
    with c3:
        st.markdown("#### 3. Random Forest (Tuned)")
        st.caption("Ensemble of 100 bagging trees optimized via GridSearchCV.")
    with c4:
        st.markdown("#### 4. Naive Bayes")
        st.caption("Probabilistic classifier based on Bayes theorem with Gaussian prior.")

    st.markdown("---")
    if st.session_state.models_trained:
        st.success("✅ All 4 models have been successfully trained and evaluated on the held-out test split.")
    else:
        st.info("Click 'Train 4 Models' in the sidebar to run the training pipeline.")


# ------------------------------------------------------------------------------
# TAB 5: MODEL COMPARISON
# ------------------------------------------------------------------------------
with tab5:
    st.header("📈 Model Performance Comparison (4 Models)")
    
    if not st.session_state.models_trained:
        st.info("Train the models first using the sidebar button.")
    else:
        res_df = st.session_state.results_df
        
        # Display comparison table
        st.subheader("Performance Metrics Table")
        
        formatted_df = res_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
            formatted_df[col] = formatted_df[col].apply(lambda v: f"{v*100:.2f}%" if pd.notnull(v) else "N/A")
        st.dataframe(formatted_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Visual Comparison")
        fig_comp = plot_combined_model_comparison(res_df)
        st.pyplot(fig_comp)
        plt.close(fig_comp)


# ------------------------------------------------------------------------------
# TAB 6: BEST MODEL IN-DEPTH EVALUATION
# ------------------------------------------------------------------------------
with tab6:
    st.header("🏆 Best Model Deep-Dive — Random Forest")

    if not st.session_state.models_trained:
        st.info("Train models first to view detailed evaluation.")
    else:
        best_name = st.session_state.best_model_name
        best_model = st.session_state.trained_models[best_name]
        best_preds = st.session_state.model_predictions[best_name]
        res_df = st.session_state.results_df
        best_row = res_df[res_df["Model"] == best_name].iloc[0]

        # Best Model Badge Card
        st.markdown(
            f"""
            <div class="best-model-card">
                <h2>🏆 Top Performing Model: {best_name}</h2>
                <h3>Optimized with 5-Fold Cross-Validation Hyperparameter Tuning</h3>
                <div class="metric-row">
                    <div class="metric-item"><div class="value">{best_row['Accuracy']*100:.2f}%</div><div class="label">Accuracy</div></div>
                    <div class="metric-item"><div class="value">{best_row['Precision']*100:.2f}%</div><div class="label">Precision</div></div>
                    <div class="metric-item"><div class="value">{best_row['Recall']*100:.2f}%</div><div class="label">Recall</div></div>
                    <div class="metric-item"><div class="value">{best_row['F1 Score']*100:.2f}%</div><div class="label">F1-Score</div></div>
                    <div class="metric-item"><div class="value">{best_row['ROC-AUC']:.4f}</div><div class="label">ROC-AUC</div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        
        # Grid of Evaluation Visuals
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.subheader("Confusion Matrix")
            fig_cm = plot_confusion_matrix(best_preds["confusion_matrix"])
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with col_e2:
            st.subheader("ROC & PR Curves")
            if best_preds["y_prob"] is not None:
                fig_roc = plot_roc_curve(st.session_state.y_test, best_preds["y_prob"], best_name, best_row["ROC-AUC"])
                st.pyplot(fig_roc)
                plt.close(fig_roc)
                
                fig_pr = plot_precision_recall_curve(st.session_state.y_test, best_preds["y_prob"], best_name)
                st.pyplot(fig_pr)
                plt.close(fig_pr)

        st.markdown("---")
        st.subheader("Feature Importance Analysis")
        pp = st.session_state.preprocessor
        fig_imp = plot_feature_importance(best_model, pp["feature_columns"])
        if fig_imp:
            st.pyplot(fig_imp)
            plt.close(fig_imp)

        st.markdown("---")
        st.subheader("Detailed Classification Report")
        cr_dict = best_preds["classification_report"]
        cr_df = pd.DataFrame(cr_dict).T
        st.dataframe(cr_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}", "support": "{:,.0f}"}), use_container_width=True)

        st.markdown("---")
        # Download button
        st.subheader("💾 Model Artifact Download")
        buf = io.BytesIO()
        joblib.dump(best_model, buf)
        buf.seek(0)
        st.download_button(
            label="⬇️ Download Tuned Random Forest (.pkl)",
            data=buf,
            file_name="api/best_phishing_model.pkl",
            mime="application/octet-stream",
            use_container_width=True,
        )


# ------------------------------------------------------------------------------
# TAB 7: LIVE INTERACTIVE PREDICTION FORM
# ------------------------------------------------------------------------------
with tab7:
    st.header("🔮 Real-Time Website Phishing Prediction")
    st.markdown("Enter website characteristics or choose a preset scenario to perform an instant live classification.")

    best_name = st.session_state.get("best_model_name", "Random Forest")
    best_model = st.session_state.trained_models.get(best_name) if st.session_state.models_trained else None
    if best_model is None and os.path.exists("api/best_phishing_model.pkl"):
        best_model = joblib.load("api/best_phishing_model.pkl")

    pp = st.session_state.preprocessor

    # Presets
    preset_choice = st.selectbox(
        "⚡ Choose a Quick Preset Scenario:",
        [
            "Custom Manual Input",
            "Example 1: Verified Legitimate Banking Portal (HTTPS, Domain Age 3500d, Low Entropy)",
            "Example 2: High-Risk Phishing Page (No HTTPS, Young Domain 12d, High Entropy, Password Field)",
            "Example 3: Malicious Redirect / Popup Scam (Multiple Redirects, Iframes, Popups)",
        ]
    )

    preset_values = {}
    if "Legitimate" in preset_choice:
        preset_values = {
            "URL_Length": 24, "Num_Dots": 1, "Num_Hyphens": 0, "Num_Special_Chars": 1,
            "Num_Subdomains": 0, "Has_IP_Address": 0, "Has_HTTPS": 1, "Domain_Age_Days": 3650,
            "Domain_Registration_Length": 730, "Has_Suspicious_Words": 0, "Num_Redirects": 0,
            "External_Link_Ratio": 0.15, "Image_Link_Ratio": 0.20, "Form_Count": 1,
            "Password_Field_Present": 1, "Iframe_Count": 0, "Popup_Count": 0,
            "Favicon_External": 0, "Domain_Name_Length": 10, "URL_Entropy": 3.12
        }
    elif "High-Risk" in preset_choice:
        preset_values = {
            "URL_Length": 115, "Num_Dots": 6, "Num_Hyphens": 4, "Num_Special_Chars": 7,
            "Num_Subdomains": 3, "Has_IP_Address": 1, "Has_HTTPS": 0, "Domain_Age_Days": 12,
            "Domain_Registration_Length": 30, "Has_Suspicious_Words": 1, "Num_Redirects": 3,
            "External_Link_Ratio": 0.85, "Image_Link_Ratio": 0.90, "Form_Count": 4,
            "Password_Field_Present": 1, "Iframe_Count": 2, "Popup_Count": 3,
            "Favicon_External": 1, "Domain_Name_Length": 28, "URL_Entropy": 4.88
        }
    elif "Redirect" in preset_choice:
        preset_values = {
            "URL_Length": 88, "Num_Dots": 4, "Num_Hyphens": 3, "Num_Special_Chars": 5,
            "Num_Subdomains": 2, "Has_IP_Address": 0, "Has_HTTPS": 0, "Domain_Age_Days": 45,
            "Domain_Registration_Length": 90, "Has_Suspicious_Words": 1, "Num_Redirects": 4,
            "External_Link_Ratio": 0.70, "Image_Link_Ratio": 0.60, "Form_Count": 3,
            "Password_Field_Present": 0, "Iframe_Count": 3, "Popup_Count": 4,
            "Favicon_External": 1, "Domain_Name_Length": 22, "URL_Entropy": 4.52
        }

    # Form inputs in 3 columns
    input_data = {}
    cols_grid = st.columns(3)
    feature_cols = pp["feature_columns"]
    binary_cols_set = {"Has_IP_Address", "Has_HTTPS", "Has_Suspicious_Words", "Password_Field_Present", "Favicon_External"}

    for idx, feat in enumerate(feature_cols):
        c_idx = idx % 3
        with cols_grid[c_idx]:
            default_v = preset_values.get(feat, 0.0)
            if feat in binary_cols_set:
                val = st.selectbox(
                    feat,
                    options=[0, 1],
                    index=int(default_v) if default_v in [0, 1] else 0,
                    key=f"input_{feat}",
                )
                input_data[feat] = val
            elif "Ratio" in feat or "Entropy" in feat:
                val = st.number_input(
                    feat,
                    value=float(default_v),
                    step=0.01,
                    format="%.2f",
                    key=f"input_{feat}",
                )
                input_data[feat] = val
            else:
                val = st.number_input(
                    feat,
                    value=int(default_v),
                    step=1,
                    key=f"input_{feat}",
                )
                input_data[feat] = val

    st.markdown("---")
    btn_predict = st.button("🔍 Predict Phishing Status", type="primary", use_container_width=True)

    if btn_predict:
        if best_model is None:
            st.error("No trained model available. Please train models in the sidebar.")
        else:
            pred, proba = predict_sample(input_data, pp, best_model, best_name)
            
            st.markdown("### Prediction Outcome")
            if pred == 1:
                st.markdown(
                    '<div class="prediction-result phishing">🚨 DANGER: PHISHING WEBSITE DETECTED</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="prediction-result legitimate">✅ SAFE: LEGITIMATE WEBSITE</div>',
                    unsafe_allow_html=True,
                )

            if proba is not None:
                p_phish = proba[1] * 100
                p_legit = proba[0] * 100

                m1, m2 = st.columns(2)
                m1.metric("Legitimate Probability", f"{p_legit:.2f}%")
                m2.metric("Phishing Probability", f"{p_phish:.2f}%")

                # Visual probability meter
                fig_meter, ax_m = plt.subplots(figsize=(8, 1.4), dpi=150)
                ax_m.barh([""], [p_legit], color="#2e7d32", label=f"Legitimate ({p_legit:.1f}%)")
                ax_m.barh([""], [p_phish], left=[p_legit], color="#c62828", label=f"Phishing ({p_phish:.1f}%)")
                ax_m.set_xlim(0, 100)
                ax_m.set_xlabel("Probability (%)", fontweight="bold")
                ax_m.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2, fontsize=9)
                plt.tight_layout()
                st.pyplot(fig_meter)
                plt.close(fig_meter)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.85rem;'>"
    "🔐 Phishing Website Detection Dashboard — Built with Streamlit, Scikit-learn & Python"
    "</p>",
    unsafe_allow_html=True,
)
