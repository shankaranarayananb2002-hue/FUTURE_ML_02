import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

st.title(" Customer Churn Prediction Dashboard")
st.write("This dashboard shows predictions and insights for customer churn.")


model_path = "models/churn_model.pkl"

try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")


st.header(" Upload Customer Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data:")
    st.dataframe(df.head())


    label_col = "Churn Value"
    if label_col in df.columns:
        X = df.drop(label_col, axis=1)
    else:
        X = df.copy()

 
    from sklearn.preprocessing import LabelEncoder
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna("Unknown").astype(str)
            X[col] = LabelEncoder().fit_transform(X[col])


    if st.button("Run Prediction"):
        try:
            probs = model.predict_proba(X)[:, 1]
            preds = (probs >= 0.5).astype(int)

            df["Churn Probability"] = probs
            df["Churn Prediction"] = preds

            st.write("### Prediction Results:")
            st.dataframe(df.head())

            
            csv_download = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                " Download Full Predictions",
                csv_download,
                "predictions.csv",
                "text/csv",
                key="download-csv"
            )
        except Exception as e:
            st.error(f"Error during prediction: {e}")

st.header("Data Insights & Visualization")

if uploaded_file:

    st.subheader("1 Churn Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x=df["Churn Value"], ax=ax1)
    ax1.set_title("Churn Distribution")
    st.pyplot(fig1)


    if "Contract" in df.columns:
        st.subheader("2 Churn by Contract Type")
        fig2, ax2 = plt.subplots()
        sns.barplot(x=df["Contract"], y=df["Churn Value"], ax=ax2)
        ax2.set_title("Churn Rate by Contract Type")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    
    if "Monthly Charges" in df.columns:
        st.subheader("3️Monthly Charges vs Churn")
        fig3, ax3 = plt.subplots()
        sns.boxplot(x=df["Churn Value"], y=df["Monthly Charges"], ax=ax3)
        ax3.set_title("Monthly Charges vs Churn")
        st.pyplot(fig3)


    st.subheader("4 Feature Importance")
    try:
        importance = model.feature_importances_
        feat_names = X.columns

        imp_df = pd.DataFrame({
            "Feature": feat_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False).head(10)

        fig4, ax4 = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=imp_df, ax=ax4)
        ax4.set_title("Top 10 Important Features")
        st.pyplot(fig4)

    except Exception as e:
        st.error(f"Feature importance unavailable: {e}")

else:
    st.info("Upload a dataset above to view insights and charts.")
