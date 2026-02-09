# Classification Models: Breast Cancer Detection

# a) Problem Statement

A comprehensive machine learning project for binary classification using multiple algorithms to detect breast cancer. This project includes model training and an interactive evaluation dashboard built with Streamlit.

# b) Dataset Description

The project uses the **Breast Cancer Wisconsin Dataset** from scikit-learn, which contains:
- **569 samples** with malignant and benign tumor data
- **30 features** derived from digitized images of cell nuclei
- **Binary classification**: Malignant (0) vs Benign (1)

# c) Models used

This project trains and evaluates six different classification models on the breast cancer dataset from scikit-learn:

- **Logistic Regression** - Linear model for binary classification
- **Decision Tree** - Tree-based classifier
- **K-Nearest Neighbors (KNN)** - Instance-based learning algorithm
- **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
- **Random Forest** - Ensemble of decision trees
- **XGBoost** - Gradient boosting framework

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---|---|---|---|---|---|
| Naive Bayes | 0.9737 | 0.9984 | 0.9595 | 1.0000 | 0.9793 | 0.9447 |
| Random Forest | 0.9649 | 0.9953 | 0.9589 | 0.9859 | 0.9722 | 0.9253 |
| Logistic Regression | 0.9561 | 0.9980 | 0.9459 | 0.9859 | 0.9655 | 0.9068 |
| KNN | 0.9561 | 0.9959 | 0.9342 | 1.0000 | 0.9660 | 0.9086 |
| XGBoost | 0.9561 | 0.9908 | 0.9583 | 0.9718 | 0.9650 | 0.9064 |
| Decision Tree | 0.9474 | 0.9440 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |

## Observations

| ML Model Name | Observation about model performance |
|---|---|
| Naive Bayes | Best overall performer with the highest F1 Score (0.9793) and accuracy (97.37%). Achieves perfect recall (1.0) and excellent AUC (0.9984), making it ideal for cases where missing positive cases is critical. |
| Random Forest | Second-best performer with strong, balanced results across all metrics. F1 Score of 0.9722 and MCC of 0.9253 indicate excellent generalization. Good choice for reliable predictions without perfect recall requirements. |
| Logistic Regression | Solid linear classifier with F1 Score of 0.9655 and high AUC (0.9980). Despite being simpler than ensemble methods, it performs comparably and offers good interpretability. Precision at 0.9459 is slightly lower. |
| KNN | Achieves perfect recall (1.0) and good F1 Score (0.9660), but with lower precision (0.9342). Works well for datasets where missing positive cases is unacceptable, though it may have higher false positive rates. |
| XGBoost | Good performer with F1 Score of 0.9650 and balanced precision-recall (0.9583 and 0.9718). Demonstrates that gradient boosting is effective but does not significantly outperform simpler methods on this dataset. |
| Decision Tree | Weakest performer with the lowest AUC (0.9440) and MCC (0.8880). Despite reasonable precision and recall, it shows signs of overfitting. The simple decision boundaries are less suitable for complex feature interactions in this dataset. |

## Project Structure

```
ModelClassification/
├── app.py                    # Streamlit web application for model evaluation
├── model/
│   ├── train_models.py       # Script to train and save all models
│   └── saved_models/         # Directory containing trained model files
│       ├── logistic_regression.pkl
│       ├── decision_tree.pkl
│       ├── knn.pkl
│       ├── naive_bayes.pkl
│       ├── random_forest.pkl
│       ├── xgboost.pkl
│       └── test_samples.csv  # Sample test data
├── requirements.txt          # Python dependencies
├── venv/                     # Optional virtual environment
└── README.md                 # This file
```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip or conda package manager

### Setup Steps

1. **Clone or download the project**
   ```bash
   cd ModelClassification
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv/Scripts/activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training Models

To train all models and save them to the `model/saved_models/` directory (run from project root):

```bash
python model/train_models.py
```

This script will:
- Load the breast cancer dataset
- Split data into training and testing sets (80/20 split)
- Train all six classification models
- Display training metrics
- Save trained models as pickle files

### Running the Evaluation Dashboard

Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard provides:
- **File Upload**: Upload your own test dataset (CSV format)
- **Target Column Selection**: Choose the target column for classification
- **Model Selection**: Select which models to evaluate
- **Performance Metrics**: View accuracy, precision, recall, F1-score, ROC-AUC, MCC
- **Confusion Matrices**: Visualize prediction patterns
- **Classification Reports**: Detailed evaluation statistics


## Dependencies

- `streamlit` - Web application framework
- `scikit-learn` - Machine learning library
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization
- `xgboost` - Gradient boosting framework
- `joblib` - Model serialization

## Model Performance

Each model is evaluated using the following metrics:
- **Accuracy** - Proportion of correct predictions
- **Precision** - Proportion of positive predictions that are correct
- **Recall** - Proportion of actual positives correctly identified
- **F1-Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under the ROC curve
- **Matthews Correlation Coefficient (MCC)** - Balanced measure for binary classification
- **Confusion Matrix** - Shows true positives, false positives, etc.

## How to Use the Web App

1. Run `streamlit run app.py`
2. Open the app in your browser (typically `http://localhost:8501`)
3. Select a model or compare multiple models
4. Upload a test CSV file with features and target column
5. Choose the target column name from the dropdown
6. View comprehensive evaluation metrics and visualizations

## Input CSV Format

Your test dataset CSV should contain:
- Feature columns (X variables) - same features as the breast cancer dataset
- Target column (y variable) - binary values (0 or 1)

Example:
```
feature1,feature2,...,feature30,target
1.5,2.3,...,4.2,1
2.1,1.8,...,3.5,0
...
```

## License

This project is for educational purposes as part of BITS Pilani coursework.

## Author

Created as a semester project for Machine Learning (Sem 1, BITS Pilani)

---

For questions or issues, please review the code comments or modify the scripts as needed for your specific use case.
