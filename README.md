# Breast Cancer Classification Using Machine Learning

## Project Overview

This project uses machine learning classification algorithms to classify
breast cancer cases as **malignant** or **benign** using numerical
features extracted from breast cancer diagnostic measurements.

The project is implemented in a Jupyter Notebook 
using Python, pandas, NumPy, Matplotlib, Seaborn, and scikit-learn. It
explores the dataset, checks data quality, prepares the data, trains
multiple classification models, and compares their performance using
evaluation metrics and visualizations.


## Objectives

-   Explore and understand a breast cancer classification dataset.
-   Inspect feature types, summary statistics, duplicate rows, and
    missing values.
-   Examine possible outliers in numerical features.
-   Split the data into training and testing sets.
-   Standardize features for algorithms that are sensitive to feature
    scale.
-   Train and evaluate multiple classification models.
-   Compare models using accuracy, precision, recall, F1-score, and
    ROC-AUC.
-   Visualize confusion matrices and ROC curves to better understand
    model performance.
-   Identify a candidate model based on evaluation results.

## Dataset

The notebook loads the **Breast Cancer Wisconsin Diagnostic dataset**
directly from scikit-learn using `sklearn.datasets.load_breast_cancer`.
A separate CSV file is not required for this dataset-loading approach.

The dataset contains numerical measurements computed from digitized
images of breast mass samples. These measurements describe
characteristics such as radius, texture, perimeter, area, smoothness,
compactness, concavity, symmetry, and fractal dimension.

-   **Input features (`X`):** numerical diagnostic measurements.
-   **Target (`y`):** the class label.
-   **Number of features:** 30.
-   **Number of samples:** 569.
-   **Target classes:** malignant and benign.

In scikit-learn's built-in dataset, the target encoding is:

  Target value   Meaning
  -------------- -----------
  `0`            Malignant
  `1`            Benign

The notebook creates the feature DataFrame from `data.data` and the
target Series from `data.target`.

## Technologies Used

  -----------------------------------------------------------------------
  Technology                          Purpose
  ----------------------------------- -----------------------------------
  Python                              Main programming language

  Jupyter Notebook                    Development  and execution
                                      environment

  pandas                              DataFrame and tabular data
                                      operations

  NumPy                               Numerical operations

  Matplotlib                          Charts and visualizations

  Seaborn                             Statistical visualizations,
                                      including count plots and heatmaps

  scikit-learn                        Dataset loading, preprocessing,
                                      model training, and evaluation
  -----------------------------------------------------------------------

## Machine Learning Models

The notebook explores five classification algorithms.

### 1. Logistic Regression

Logistic Regression is a classification algorithm that estimates the
probability of a sample belonging to a class. In this notebook, it is
trained using standardized features and used as a baseline
classification model.

### 2. Decision Tree Classifier

A Decision Tree classifies samples through a series of feature-based
decisions. It is interpretable and can model non-linear relationships.
The notebook sets `max_depth=5` and `random_state=42`.

### 3. Random Forest Classifier

Random Forest combines predictions from multiple decision trees. It can
capture more complex patterns than a single tree and often provides
robust classification performance. The notebook uses 100 trees and
`random_state=42`.

### 4. Support Vector Machine (SVM)

The SVM model attempts to separate classes using a decision boundary.
The notebook uses an RBF kernel and enables probability estimates for
ROC-AUC evaluation. Standardized features are used for this model.

### 5. K-Nearest Neighbors (KNN)

KNN predicts a class based on the classes of nearby samples. Feature
scaling is important because KNN relies on distances between data
points. The notebook initially tests `k=5` and also evaluates candidate
values of `k` from 1 to 20.

## Project Workflow

### 1. Import libraries and load the dataset

The notebook imports the required Python and scikit-learn libraries and
loads the built-in breast cancer dataset.

### 2. Explore the data

- Feature names and dataset shape.
- Target values and class names.
- Data types and summary statistics.
- Sample rows from the feature and target data.
- Class distribution using a count plot.

### 3. Check data quality

Checks for duplicate feature rows and missing values. It
also calculates potential outliers using the Interquartile Range (IQR)
method.



### 4. Split the data

It uses `train_test_split` with: - Test size: 20%. - Training
size: 80%. - `random_state=42` for reproducibility. - `stratify=y` to
preserve the target-class proportions across the split.

### 5. Scale the features

 It use `StandardScaler` to standardize the training features
and then transforms the test features with the same fitted scaler. This
is used for Logistic Regression, SVM, and KNN.

Decision Tree and Random Forest are trained on the unscaled feature data
in the notebook.

### 6. Train the classification models

Each model is fitted on the training data and then used to predict the
labels of the test data.

### 7. Evaluate model performance

includes several evaluation techniques:

-   **Accuracy:** the proportion of all predictions that are correct.
-   **Precision:** the proportion of positive predictions that are
    correct.
-   **Recall:** the proportion of actual positive cases that the model
    identifies.
-   **F1-score:** the harmonic mean of precision and recall.
-   **Confusion matrix:** counts of correct and incorrect predictions
    for each class.
-   **ROC curve:** shows the relationship between true-positive rate and
    false-positive rate at different classification thresholds.
-   **ROC-AUC:** summarizes how well the model separates the two classes
    across thresholds.

For medical classification, accuracy alone may not be sufficient.
Recall, precision, confusion-matrix results, and the consequences of
false negatives and false positives should also be considered.

### 8. Compare model performance

Creates a model-comparison table and visualizations
intended to compare the classification models using evaluation metrics.
It also includes: 
- Classification reports for individual models.
- Confusion-matrix heatmaps for all models.
- ROC curves for model comparison.
- A bar chart comparing evaluation metrics.
- Code to display the highes
- lowest-ranked models based on the selected metric.

The final model should be chosen based on the evaluation results and the
purpose of the task, rather than accuracy alone.

## Visualizations



1.  **Class Distribution Plot** --- displays the number of samples in
    each target class.
2.  **Decision Tree Plot** --- visualizes the trained decision tree.
3.  **Individual ROC Curves** --- show the ROC curve and AUC for each
    model.
4.  **Confusion Matrix Heatmaps** --- compare actual and predicted
    classes for each model.
5.  **ROC Curve Comparison** --- plots the ROC curves of multiple models
    together.
6.  **Model Performance Bar Chart** --- compares metrics such as
    accuracy, precision, recall, F1-score, and ROC-AUC.













No license is specified in the notebook. If you publish this project on
GitHub, choose and add an appropriate license file if you want to define
how others may use, modify, or distribute your work.
