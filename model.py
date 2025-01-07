# File: app.py
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
st.title("Breast Cancer Diagnosis Prediction")

# Sidebar for file upload
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Default file path
default_file_path = r"D:\breast cancer prediction\BreastCancer.csv"

if uploaded_file is not None:
    # If a file is uploaded, use it
    data = pd.read_csv(uploaded_file)
elif default_file_path:
    # Fallback to the default file path
    try:
        data = pd.read_csv(default_file_path)
        st.write("No file uploaded. Using default dataset from:")
        st.write(default_file_path)
    except FileNotFoundError:
        st.error(f"File not found at {default_file_path}. Please upload a file.")
        st.stop()
else:
    st.write("Please upload a CSV file or provide a valid default path.")
    st.stop()

# Display dataset preview
st.write("Dataset Preview:")
st.write(data.head())

# Check for target column
if 'diagnosis' not in data.columns:
    st.error("The dataset must contain a 'diagnosis' column!")
else:
    # Encode diagnosis column
    data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)  # Malignant=1, Benign=0

    # Feature selection
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Allow user input for predictions
    st.subheader("Make a Prediction")
    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(f"Enter value for {col}", float(X[col].min()), float(X[col].max()))

    user_input_df = pd.DataFrame([user_input])
    prediction = model.predict(user_input_df)
    prediction_result = "Malignant" if prediction[0] == 1 else "Benign"
    st.write(f"Prediction: {prediction_result}")
