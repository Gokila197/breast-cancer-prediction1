import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# App title
st.title("Breast Cancer Diagnosis Prediction")

# Local dataset path
dataset_path = "D:/breast cancer prediction/BreastCancer.csv"

try:
    # Read dataset
    data = pd.read_csv(dataset_path)
    st.write("Dataset Preview:")
    st.write(data.head())

    # Validate dataset
    if 'diagnosis' not in data.columns:
        st.error("The dataset must contain a 'diagnosis' column!")
    else:
        # Encode target column
        data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)  # Malignant=1, Benign=0

        # Feature selection
        X = data.drop(columns=['diagnosis'])
        y = data['diagnosis']

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest Classifier
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy:.2f}")

        # User input for predictions
        st.subheader("Make a Prediction")
        user_input = {}
        for col in X.columns:
            user_input[col] = st.number_input(f"Enter value for {col}", float(X[col].min()), float(X[col].max()))

        user_input_df = pd.DataFrame([user_input])
        prediction = model.predict(user_input_df)
        prediction_result = "Malignant" if prediction[0] == 1 else "Benign"
        st.write(f"Prediction: {prediction_result}")
except Exception as e:
    st.error(f"Error processing the file: {e}")
