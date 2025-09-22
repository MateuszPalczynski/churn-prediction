# Customer Churn Prediction Analysis

## Project Goal

In the dynamic and competitive telecommunications industry, customer retention is significantly more cost-effective than customer acquisition. The phenomenon of customers leaving for a competitor, known as "churn," is a key business challenge. The primary goal of this analysis is to proactively identify customers at the highest risk of churning. By building a precise predictive model, the company can implement targeted retention strategies (e.g., personalized offers, proactive support), thereby reducing the attrition rate, protecting the revenue base, and increasing Customer Lifetime Value (CLV).

## The Analytical Problem

From a machine learning perspective, the churn prediction problem is a **binary classification task**. Our objective is to build a model that assigns one of two labels to each customer: `Churn=1` (the customer will leave) or `Churn=0` (the customer will stay).

A key challenge in this project is the **imbalanced dataset**, where the number of loyal customers significantly outweighs the number of churning customers. This requires the use of specialized sampling techniques and evaluation metrics to avoid building a model that simply favors the majority class and fails to identify the customers who are most important from a business perspective.

## Dataset

The analysis is based on a historical dataset of a telecommunications company's customers. The dataset includes a wide range of attributes:
* **Demographic data:** Gender, senior citizen status, partner, and dependents.
* **Subscribed services:** Phone service, multiple lines, internet service type, and additional online services (e.g., online security, backup, streaming).
* **Contractual and financial data:** Contract type, tenure, monthly charges, total charges, and payment method.

## Project Workflow & Methodology

This project was executed following a rigorous, multi-stage analytical methodology to ensure the reliability and high quality of the final solution.

### 1. Data Cleaning and Preprocessing
* **Handling Missing Values:** Imputed missing values in the `TotalCharges` column, which appeared as empty strings in the raw data.
* **Data Type Conversion:** Ensured all features had appropriate numeric data types.
* **Categorical Feature Encoding:** Applied `One-Hot Encoding` for nominal features and `Ordinal Encoding` for ordinal features like `Contract`.
* **Duplicate Handling:** Identified and removed duplicate rows from the dataset.

### 2. Exploratory Data Analysis (EDA) & Feature Engineering
* **Distribution Analysis:** Visualized feature distributions using histograms and count plots to identify anomalies, such as low-variance features.
* **Feature Engineering:** Created new, informative features
* **Feature Transformation:** Analyzed and handled the long-tail distribution of the `TotalCharges` feature through **Box-Cox transformation** to prepare it for sensitive models.

### 3. Handling Class Imbalance
* Systematically tested a wide range of **sampling techniques** (including `RandomOverSampler`, `RandomUnderSampler`, `SMOTE`, `ADASYN`, `TomekLinks`) against a `None` baseline to find the optimal strategy for managing the imbalanced `Churn` variable.

### 4. Model Selection and Evaluation Strategy
* **Model Grouping:** Models were grouped into two categories based on their preprocessing needs:
    1.  **Robust Models:** Tree-based ensembles (`Random Forest`, `XGBoost`, `LightGBM`, `AdaBoost`,`Gradient Boosting`, `Decision Tree`) that do not require feature scaling.
    2.  **Sensitive Models:** Linear, distance-based, and probabilistic models (`Logistic Regression`, `k-NN`, `SVC`, `GaussianNB`) that require a fully preprocessed (scaled and transformed) dataset.
* **Validation Method:** **Stratified K-Fold Cross-Validation (k=4)** was used for robust and stable performance evaluation.
* **Evaluation Metric:** **Recall** was chosen as the primary business metric for optimization, while also monitoring `Precision`, `F1-Score`, and `ROC AUC` for a comprehensive performance view.

### 5. Hyperparameter Optimization
* A two-stage tuning strategy was implemented for efficiency:
    1.  **"Horse Race":** A quick comparison of all model-sampler combinations with default parameters to identify the most promising candidates.
    2.  **Bayesian Optimization (`BayesSearchCV`):** Targeted and efficient hyperparameter tuning for the top-performing combinations from the first stage.
 
### 6. Experiment tracking (MLflow)
* For repeatability and comparability of experiments, the project uses **MLflow**.  

## Results

After comprehensive experimentation, the combination that yielded the best performance in terms of the `recall` metric was:

* **Model:** **Support Vector Classifier (SVC)**
* **Sampling Technique:** **RandomUnderSampler**
* **Dataset:** Fully preprocessed data (`Box-Cox` transformed and `StandardScaler` scaled).

The final, optimized pipeline was evaluated on the held-out test set, achieving the following results:

| Metric    | Score          |
| :-------- | :------------- |
| **Recall**| **0.9793**     |
| Precision | 0.2979         |
| F1-Score  | 0.4568         |
| ROC AUC   | 0.3468         |

These results indicate that the model is capable of correctly identifying a significant majority of customers who are at risk of churning, providing a solid foundation for implementing effective retention campaigns.

## Technologies Used

This project leverages a comprehensive stack of Python libraries for data science and machine learning.

### Data Manipulation & Analysis
* **Pandas:** For data structuring, manipulation, and cleaning using its powerful DataFrame object.
* **NumPy:** For fundamental numerical operations, array manipulation, and mathematical functions.

### Data Visualization
* **Matplotlib & Seaborn:** For creating a wide range of static statistical plots, from histograms and box plots to heatmaps and count plots, to explore data distributions and relationships.
* **Plotly:** For creating interactive visualizations used for dynamic data exploration.

### Feature Preprocessing & Engineering
* **Scikit-learn (`sklearn.preprocessing`, `sklearn.impute`, `sklearn.decomposition`):** The core toolkit for essential preprocessing tasks, including feature scaling (`StandardScaler`), handling missing data (`IterativeImputer`), and dimensionality reduction (`PCA`).
* **SciPy (`scipy.stats`):** Specifically used for the `boxcox` function to transform skewed data distributions.

### Handling Imbalanced Datasets
* **imbalanced-learn (`imblearn`):** A crucial library for addressing class imbalance. Used to systematically test various over-sampling (`SMOTE`, `ADASYN`) and under-sampling (`RandomUnderSampler`, `TomekLinks`) techniques.

### Model Training & Machine Learning
* **Scikit-learn (`sklearn.linear_model`, `sklearn.ensemble`, etc.):** Provided the foundation for the majority of the machine learning models tested, including Logistic Regression, Random Forest, SVC, and more.
* **XGBoost:** For implementing the high-performance, gradient boosting eXtreme Gradient Boosting classifier.
* **LightGBM:** For implementing the fast and efficient Light Gradient Boosting Machine, another state-of-the-art gradient boosting framework.

### Model Evaluation & Hyperparameter Optimization
* **Scikit-learn (`sklearn.model_selection`, `sklearn.metrics`):** Used for robust model evaluation through `StratifiedKFold` cross-validation and calculating a comprehensive suite of performance metrics (`recall`, `precision`, `f1_score`, `roc_auc_score`, `confusion_matrix`).
* **Scikit-optimize (`skopt`):** For advanced hyperparameter tuning using `BayesSearchCV`, enabling an intelligent and efficient search for the optimal model parameters.
