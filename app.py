import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Load models (cached)
# --------------------------------------------------
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("model/saved_models/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/saved_models/decision_tree.pkl"),
        "KNN": joblib.load("model/saved_models/knn.pkl"),
        "Naive Bayes": joblib.load("model/saved_models/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/saved_models/random_forest.pkl"),
        "XGBoost": joblib.load("model/saved_models/xgboost.pkl"),
    }

models = load_models()

st.title("📊 Classification Models: Evaluation Dashboard")

# --------------------------------------------------
# Test Dataset upload 
# -------------------------------------------------
st.header("📁 Upload Test Dataset (CSV)")

uploaded_file = st.file_uploader(
    "Upload test dataset (features + target column)",
    type=["csv"]
)
# Download sample test data
with open("model/saved_models/test_samples.csv", "rb") as f:
    st.download_button(
        label="Download Sample Test Data",
        data=f,
        file_name="test_samples.csv",
        mime="text/csv"
    )
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # --------------------------------------------------
    # Target column selection
    # --------------------------------------------------
    target_col = st.selectbox("Select target column [for the example csv shared, select 'target']", df.columns)

    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]

    # --------------------------------------------------
    # Model selection
    # --------------------------------------------------
    st.header("Select Model")
    model_name = st.selectbox("Choose a trained model", list(models.keys()))
    model = models[model_name]

    # --------------------------------------------------
    # Run Evaluation
    # --------------------------------------------------
    if st.button("Evaluate Model"):
        y_pred = model.predict(X_test)

        # For AUC
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_score)
        else:
            auc = None

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        # --------------------------------------------------
        # Display Metrics
        # --------------------------------------------------
        st.subheader("📈 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC", f"{auc:.4f}" if auc is not None else "N/A")
        col3.metric("Precision", f"{prec:.4f}")
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")
        

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        )

        st.dataframe(cm_df)

        # --------------------------------------------------
        # Classification Report
        # --------------------------------------------------
        st.subheader("📄 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df.round(4))

else:
    st.info("Please upload a CSV file containing test data.")
