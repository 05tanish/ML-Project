# Phishing Website Detection Project Documentation

This document provides a comprehensive explanation of the Jupyter Notebook (`phishing_classification.ipynb`), the feature engineering process, model selection, and common interview/viva questions.

---

## 1. Notebook Overview (`phishing_classification.ipynb`)

The Jupyter Notebook is the core research environment where the data was explored and machine learning models were trained. The flow of the notebook is as follows:

1. **Data Loading & Cleaning**: Importing `phishing_website_raw.csv` and checking for missing values, duplicates, and data types.
2. **Exploratory Data Analysis (EDA)**:
   - **Histograms & Boxplots**: To observe distributions and outliers (e.g., highly skewed URL lengths in phishing sites).
   - **Correlation Heatmap**: To identify which features correlate most strongly with the target variable `Is_Phishing` (e.g., `Has_HTTPS` is strongly negatively correlated, meaning having HTTPS reduces phishing probability).
3. **Data Preprocessing**:
   - Splitting the dataset into Training (80%) and Testing (20%) using stratified sampling.
   - Using `StandardScaler` to normalize the data for distance-based models (like Logistic Regression).
4. **Model Training & Evaluation**:
   - Training four classification models: **Logistic Regression**, **Decision Tree**, **Random Forest**, and **Naive Bayes**.
   - Comparing them using metrics like Accuracy, Precision, Recall, and F1-Score.
5. **Hyperparameter Tuning & Export**:
   - Selecting the best model and tuning it (using `GridSearchCV` conceptually).
   - Exporting the final trained model (`best_phishing_model.pkl`) and scaler (`phishing_scaler.pkl`) using `joblib` for deployment.

---

## 2. Feature Columns & Functionality

The dataset consists of 20 carefully selected features extracted from URLs and webpage content:

| Feature Name | Functionality & Relevance |
|:---|:---|
| **URL_Length** | Total character count of the URL. Phishers often use very long URLs to hide suspicious parts. |
| **Num_Dots** | Number of dots in the URL. Multiple subdomains (e.g., `login.paypal.secure.com`) increase this count. |
| **Num_Hyphens** | Number of dashes. Legitimate sites rarely use multiple hyphens; phishers use them to mimic real names (e.g., `apple-login-update.com`). |
| **Num_Special_Chars** | Symbols like `@`, `?`, `=`, `&`. Used heavily in phishing to inject scripts or obfuscate the true domain. |
| **Num_Subdomains** | Extracted from the domain. Phishers nest subdomains to look legitimate. |
| **Has_IP_Address** | Boolean (0 or 1). If the domain is just an IP (e.g., `http://192.168.1.1`), it is highly likely to be phishing. |
| **Has_HTTPS** | Boolean. Legitimate sites use HTTPS. However, some phishers now use free SSL, so it's not a standalone guarantee. |
| **Domain_Age_Days** | Time since the domain was registered. Phishing domains are usually very new (active for only a few days/weeks). |
| **Domain_Registration_Length** | Expiry time of the domain. Phishers buy domains for the shortest possible time (1 year). |
| **Has_Suspicious_Words** | Boolean. Checks for words like "login", "verify", "update", "banking" in the URL. |
| **Num_Redirects** | Number of times the page redirects. Phishers use redirects to evade detection scanners. |
| **External_Link_Ratio** | Percentage of links pointing to other domains. Phishing sites often rip HTML from legitimate sites, causing links to point externally. |
| **Image_Link_Ratio** | Similar to above; images loaded from external legitimate servers to look authentic. |
| **Form_Count** | Number of `<form>` tags. Phishing sites always contain forms to steal credentials. |
| **Password_Field_Present** | Boolean. Crucial indicator if the page asks for a password. |
| **Iframe_Count** | Number of hidden `<iframe>` tags used to load malicious content invisibly. |
| **Popup_Count** | Number of popups. Rare on modern legitimate sites, common in phishing to urgently ask for data. |
| **Favicon_External** | Boolean. If the small icon in the browser tab is loaded from a different domain. |
| **Domain_Name_Length** | Length of just the domain part. |
| **URL_Entropy** | Mathematical measure of randomness. Phishing URLs often consist of random hashes (e.g., `?token=xj902jd8u23d`), leading to high entropy. |
| **Is_Phishing** | Target Variable (1 = Phishing, 0 = Legitimate). |

---

## 3. Model Selection: Why Random Forest?

**We chose the Random Forest Classifier (Tuned, 100 Trees, max_depth=20).**

**Why this model?**
1. **High Accuracy (97.16%)**: Random Forest builds multiple decision trees and merges them together. This ensemble method drastically reduces the risk of overfitting compared to a single Decision Tree.
2. **Handles Non-Linearity**: Phishing indicators are highly non-linear and interactive. For example, a long URL is suspicious, but a long URL *combined* with an IP address instead of a domain name is a guaranteed phishing attempt. Random Forest captures these interactions perfectly.
3. **No Need for Scaling**: Unlike Logistic Regression or KNN, Random Forest doesn't require strict feature scaling (`StandardScaler`), making the pipeline simpler.
4. **Feature Importance**: Random Forest natively outputs which features are most important (e.g., Domain Age and URL Entropy), which is vital for cybersecurity interpretability.

---

## 4. Potential "Tricky" Interview/Viva Questions

**Q1: Why didn't you use Naive Bayes?**
> **Answer**: The Naive Bayes algorithm relies on the assumption that all features are statistically independent. In our dataset, features are highly correlated (e.g., `URL_Length` and `Num_Dots`, or `Domain_Age` and `Domain_Registration_Length`). Because this "naive" assumption is violated, Naive Bayes tends to perform worse than Random Forest.

**Q2: If Random Forest is so good, why not use an even heavier model like Support Vector Machines (SVM) or Deep Neural Networks?**
> **Answer**: Heavy models require significant computational resources, both for training and real-time inference. Since this project is deployed on a **Serverless Architecture (Vercel)**, we have strict memory constraints (500 MB limit) and execution timeouts. Random Forest strikes the perfect balance: it achieves >97% accuracy while remaining incredibly fast and lightweight.

**Q3: How did you fix the Vercel 500MB limit issue?**
> **Answer**: Data science libraries like Pandas and Scipy are massive. To deploy the API on Vercel without exceeding the 500MB limit, we created a `.vercelignore` file to decouple the Streamlit frontend dependencies from the Vercel backend. We then rewrote the API code to feed predictions into the model using a native Python/NumPy 2D array instead of a `pandas.DataFrame`.

**Q4: If a legitimate site has forms and asks for a password (like Facebook), won't your model flag it as phishing?**
> **Answer**: No, because the model doesn't look at features in isolation. While Facebook has a password field (`Password_Field_Present=1`), it also has a massive `Domain_Age_Days`, uses `HTTPS`, has low `URL_Entropy`, and `Has_IP_Address=0`. The Random Forest model considers the *combination* of all these features to make an accurate prediction.

**Q5: Why did you use `StratifiedShuffleSplit` or `stratify=y` during the train-test split?**
> **Answer**: To maintain the exact proportion of Phishing (1) and Legitimate (0) websites in both the training and testing sets. If we randomly split, one set might accidentally end up with too many legitimate sites, which would bias the model evaluation.

**Q6: What is URL Entropy and why is it useful?**
> **Answer**: Entropy is a concept from Information Theory that measures randomness or unpredictability. Phishers often use randomly generated strings in URLs to bypass security blacklists (e.g., `example.com/login.php?session=x8a9dfj34j9f8`). A high entropy score flags this randomness, which legitimate sites (using human-readable URLs like `/login`) usually don't have.
