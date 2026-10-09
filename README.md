# Breast Cancer Classification Using Machine Learning

## Project Overview

This project uses machine learning classification algorithms to classify
breast cancer cases as **malignant** or **benign** using numerical
features extracted from breast cancer diagnostic measurements.

The project is implemented in a Jupyter Notebook (`final usee.ipynb`)
using Python, pandas, NumPy, Matplotlib, Seaborn, and scikit-learn. It
explores the dataset, checks data quality, prepares the data, trains
multiple classification models, and compares their performance using
evaluation metrics and visualizations.

> **Important:** This is an educational machine learning project. Its
> predictions are not a medical diagnosis and must not be used as a
> substitute for assessment by a qualified healthcare professional.

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

  Jupyter Notebook / Google Colab     Development and execution
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

The notebook examines: - Feature names and dataset shape. - Target
values and class names. - Data types and summary statistics. - Sample
rows from the feature and target data. - Class distribution using a
count plot.

### 3. Check data quality

The notebook checks for duplicate feature rows and missing values. It
also calculates potential outliers using the Interquartile Range (IQR)
method.

**Preprocessing note:** The notebook applies IQR-based clipping to the
feature data. Outlier handling should be reviewed carefully because
extreme values can be meaningful in medical datasets. Any preprocessing
decisions should be fitted using training data only to avoid data
leakage.

### 4. Split the data

The notebook uses `train_test_split` with: - Test size: 20%. - Training
size: 80%. - `random_state=42` for reproducibility. - `stratify=y` to
preserve the target-class proportions across the split.

### 5. Scale the features

The notebook uses `StandardScaler` to standardize the training features
and then transforms the test features with the same fitted scaler. This
is used for Logistic Regression, SVM, and KNN.

Decision Tree and Random Forest are trained on the unscaled feature data
in the notebook.

### 6. Train the classification models

Each model is fitted on the training data and then used to predict the
labels of the test data.

### 7. Evaluate model performance

The notebook includes several evaluation techniques:

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

The notebook creates a model-comparison table and visualizations
intended to compare the classification models using evaluation metrics.
It also includes: - Classification reports for individual models. -
Confusion-matrix heatmaps for all models. - ROC curves for model
comparison. - A bar chart comparing evaluation metrics. - Code to
display the highest- and lowest-ranked models based on the selected
metric.

The final model should be chosen based on the evaluation results and the
purpose of the task, rather than accuracy alone.

## Visualizations

The notebook includes or attempts to generate the following
visualizations:

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
7.  **KNN k-value Experiment** --- evaluates different neighbor counts
    from 1 to 20.

## Project Structure

The repository can be organized as follows:

``` text
Breast-Cancer-Classification/
│
├── final usee.ipynb
├── README.md
└── requirements.txt
```

-   `final usee.ipynb` --- notebook containing data exploration,
    preprocessing, model training, evaluation, and plots.
-   `README.md` --- project documentation.
-   `requirements.txt` --- Python packages required to run the notebook,
    if you choose to add this file.

The `requirements.txt` file is a suggested addition; it is not confirmed
to be included with the uploaded notebook.

## Installation and Setup

### Option 1: Run in Google Colab

1.  Open [Google Colab](https://colab.research.google.com/).
2.  Upload `final usee.ipynb`, or open the notebook from your Google
    Drive.
3.  Run the notebook cells in order, from top to bottom.
4.  Review the printed results, metric tables, and plots.

The notebook uses a dataset built into scikit-learn, so you do not need
to upload a separate dataset CSV for the dataset-loading step.

### Option 2: Run locally using Jupyter Notebook

#### Step 1: Install Python

Install a compatible Python version from
[python.org](https://www.python.org/downloads/), or use an existing
Python environment.

#### Step 2: Create a project folder

Create a folder named `Breast-Cancer-Classification` and place
`final usee.ipynb` inside it.

#### Step 3: Create and activate a virtual environment (recommended)

Create the environment:

``` bash
python -m venv .venv
```

Activate it on Windows:

``` bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

``` bash
source .venv/bin/activate
```

#### Step 4: Install dependencies

Run:

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

#### Step 5: Launch Jupyter Notebook

Run:

``` bash
jupyter notebook
```

Open `final usee.ipynb` in the browser tab and execute the cells in
order.

## Requirements File

You can create a `requirements.txt` file containing:

``` text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install the listed packages with:

``` bash
pip install -r requirements.txt
```

For reproducible results, consider recording the exact package versions
used in your working environment.

## How to Upload the Project to GitHub

1.  Create a new repository on [GitHub](https://github.com/).
2.  Choose a repository name, for example
    `Breast-Cancer-Classification`.
3.  Add `final usee.ipynb`, `README.md`, and optionally
    `requirements.txt`.
4.  Commit and push the files to the repository.
5.  Open the repository page and verify that the README renders
    correctly and that the notebook can be opened.

If using Git from the project folder, the basic commands are:

``` bash
git init
git add "final usee.ipynb" README.md requirements.txt
git commit -m "Add breast cancer classification project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the example repository URL with your actual GitHub repository
URL. If `origin` already exists, update it with
`git remote set-url origin YOUR-REPOSITORY-URL` rather than adding it
again. If you do not create `requirements.txt`, remove it from the
`git add` command.

## Important Notes and Recommended Improvements

This README describes the notebook's intended workflow. Before
presenting the project as a fully working result, run the notebook from
the first cell to the last and verify that all cells complete
successfully.

A few items deserve particular review:

-   **Model comparison cell:** the notebook's later comparison section
    references metric functions and a probability/score variable that
    may not be defined in that section. Ensure `precision_score`,
    `recall_score`, `f1_score`, and the correct model-specific
    probability scores are defined before running the comparison.
-   **KNN selection:** the notebook selects a `k` value by checking
    accuracy on the test set. For a more reliable evaluation, choose `k`
    using cross-validation on the training set, then evaluate the
    selected model on the test set only once.
-   **Preprocessing leakage:** calculate and apply outlier boundaries
    using training data only. Prefer a scikit-learn pipeline for scaling
    and other learned preprocessing.
-   **Target meaning:** in scikit-learn's built-in dataset, class `0` is
    malignant and class `1` is benign. Keep this mapping consistent in
    charts and classification reports.
-   **Medical context:** do not describe this model as a clinically
    validated diagnostic system. Real-world use would require clinical
    validation, careful risk assessment, and professional oversight.
-   **Best model:** do not claim that a specific algorithm is the best
    until the notebook runs successfully and the actual metrics have
    been checked.

## Results

The notebook is designed to produce model metrics and comparison charts
when executed. The actual accuracy, precision, recall, F1-score,
ROC-AUC, and best-performing model are not listed here because they
should be copied from a successful run rather than assumed or invented.

You can add a results table after running all cells:

  Model                   Accuracy   Precision   Recall   F1-score   ROC-AUC
  --------------------- ---------- ----------- -------- ---------- ---------
  Logistic Regression          ---         ---      ---        ---       ---
  Decision Tree                ---         ---      ---        ---       ---
  Random Forest                ---         ---      ---        ---       ---
  SVM                          ---         ---      ---        ---       ---
  KNN                          ---         ---      ---        ---       ---

Replace the dashes with the actual results from your completed notebook.

## Future Enhancements

-   Correct and validate the final model-comparison code.
-   Use cross-validation and systematic hyperparameter tuning.
-   Add a reusable preprocessing and classification pipeline.
-   Save the selected model and preprocessing object for later
    inference.
-   Add a user interface using Streamlit, if an interactive demo is
    desired.
-   Include automated tests and a reproducible environment.
-   Document limitations and evaluate model performance beyond a single
    train-test split.

## Disclaimer

This project is for learning and demonstration purposes only. It is not
intended to diagnose, treat, cure, or prevent any disease. Do not make
medical decisions based on its output. Consult a qualified healthcare
professional for medical advice and diagnosis.

## Author

**Author:** Add your name here.

## License

No license is specified in the notebook. If you publish this project on
GitHub, choose and add an appropriate license file if you want to define
how others may use, modify, or distribute your work.
