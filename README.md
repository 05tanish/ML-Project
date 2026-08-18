# 🔐 Phishing Website Detection Using Machine Learning

An end-to-end machine learning project for detecting phishing websites using multiple classification algorithms. Built with Streamlit for an interactive dashboard experience.

## 🚀 Features

- **Interactive Web Dashboard** — Two Streamlit applications with rich UI components
- **Dataset Upload** — Upload your own CSV dataset for analysis
- **Exploratory Data Analysis (EDA)** — Visualize distributions, correlations, outliers, and class balance
- **6 Classification Models** — Train and compare multiple algorithms side-by-side:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Naive Bayes (Gaussian)
  - Support Vector Machine (SVM)
- **Rigorous Preprocessing** — Data leakage prevention with train-only fitted transformers
- **Cross-Validation** — Stratified K-Fold CV for robust model evaluation
- **Model Explainability** — Feature importance plots and confusion matrices
- **Single Sample Prediction** — Input custom feature values for real-time detection
- **Risk Level Assessment** — HIGH / MEDIUM / LOW risk categorization
- **Model Export** — Download the best trained model as a joblib bundle

## 📁 Project Structure

```
Ml-project/
├── app.py                           # Main Streamlit app (7-tab dashboard)
├── phishing_detector_app.py         # Enhanced Streamlit app (8-tab + history + CV)
├── phishing_website_raw.csv         # Sample phishing website dataset
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # Project documentation
```

## 🛠️ Tech Stack

| Category       | Libraries/Tools                                      |
|----------------|------------------------------------------------------|
| **Language**   | Python 3.8+                                          |
| **Framework**  | Streamlit                                            |
| **ML Core**    | scikit-learn                                         |
| **Data**       | pandas, NumPy                                        |
| **Visuals**    | matplotlib                                           |
| **Models**     | joblib (serialization)                               |

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/05tanish/ML-Project.git
cd ML-Project
```

2. **Create and activate a virtual environment (recommended)**
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Running the Applications

Choose one of the two Streamlit apps:

```bash
# App 1 — 7-tab dashboard
streamlit run app.py

# App 2 — 8-tab enhanced dashboard (recommended)
streamlit run phishing_detector_app.py
```

Your default browser will open automatically at `http://localhost:8501`.

### Workflow

1. **Upload Dataset** — Use the sidebar to upload a CSV file with a binary target column (e.g., `Is_Phishing` with 0 = Legitimate, 1 = Phishing).
2. **Configure Settings** — Adjust test size, random state, CV folds, and Random Forest tree count.
3. **Run EDA** — Explore the data quality, distributions, and correlations.
4. **Train Models** — Click "Train & Compare Models" to run all six classifiers.
5. **Compare Performance** — Rank models by F1 Score, Accuracy, Precision, Recall, ROC-AUC, and CV F1.
6. **Inspect Best Model** — View confusion matrix, ROC curve, precision-recall curve, and feature importances.
7. **Make Predictions** — Use the interactive form to classify a single website or download the model for production.

### Dataset Format

Your CSV should contain:
- One row per website sample
- Multiple feature columns (numerical and/or categorical)
- One binary target column (0 = Legitimate, 1 = Phishing). Common names: `Is_Phishing`, `Result`, `Label`.

A sample dataset `phishing_website_raw.csv` is provided to get started.

## 🧠 Preprocessing Pipeline

All preprocessors are **fitted exclusively on training data** to prevent data leakage:

| Step           | Method                                               |
|----------------|------------------------------------------------------|
| Duplicates     | Drop exact duplicate rows                            |
| Split          | Stratified train/test split (preserves class balance)|
| Imputation     | Numerical → Median; Categorical → Most Frequent      |
| Encoding       | One-Hot Encoder with `handle_unknown='ignore'`       |
| Scaling        | StandardScaler (applied only to LR, KNN, SVM)        |

## 📊 Evaluation Metrics

Models are evaluated and ranked on:

- **Accuracy** — Overall correctness
- **Precision** — Minimize false alarms (legit flagged as phish)
- **Recall** — Catch as many phishing sites as possible
- **F1 Score** — Harmonic mean of precision and recall (primary ranking metric)
- **ROC-AUC** — Discrimination ability across thresholds
- **CV F1** — Stratified K-Fold cross-validated F1 (stability check)

## 🤖 Models

| Model                  | Scaling Required | Use Case                                     |
|------------------------|------------------|----------------------------------------------|
| Logistic Regression    | ✅ Yes           | Baseline linear classifier                   |
| Decision Tree          | ❌ No            | Interpretable rule-based model               |
| Random Forest          | ❌ No            | Ensemble; typically highest performance      |
| K-Nearest Neighbors    | ✅ Yes           | Instance-based, distance-sensitive           |
| Naive Bayes            | ❌ No            | Fast probabilistic baseline                  |
| SVM                    | ✅ Yes           | High-dimensional, non-linear boundaries      |

## 💾 Model Export

After training, navigate to the **Best Model** tab and download a serialized bundle containing:
- Trained best model
- Fitted preprocessor pipeline
- Feature and target column metadata
- Performance metrics dictionary

Load in production:
```python
import joblib

bundle = joblib.load("best_model.joblib")
model = bundle["model"]
preprocessor = bundle["preprocessor"]
```

## ⚙️ Requirements

Full list in `requirements.txt`:

```
streamlit>=1.28.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
joblib>=1.3.0
```

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues and pull requests. Suggestions for additional models (XGBoost, LightGBM, Neural Networks) are welcome!
