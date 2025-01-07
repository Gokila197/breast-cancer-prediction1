# model.py
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set the title of the application
st.title("Breast Cancer Diagnosis Prediction")

# Sidebar for CSV file upload
st.sidebar.header("D:\breast cancer prediction\BreastCancer.csv")
uploaded_file = st.sidebar.file_uploader("D:\breast cancer prediction\BreastCancer.csv", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv("D:\breast cancer prediction\BreastCancer.csv")
    st.subheader("Dataset Overview")
    st.write(data.head())

    # Validate the dataset
    if 'diagnosis' not in data.columns:
        st.error("The dataset must contain a 'diagnosis' column!")
    else:
        st.success("Dataset successfully loaded!")
        # Encode the 'diagnosis' column
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})  # Malignant=1, Benign=0

        # Split features and target
        X = data.drop(columns=['diagnosis'])
        y = data['diagnosis']

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy:.2f}")

        # Section for user input
        st.subheader("Make a Diagnosis Prediction")
        st.write("Enter values for the following features:")

        # Dynamically create input fields based on feature names
        user_data = {col: st.number_input(f"{col}", float(X[col].min()), float(X[col].max())) for col in X.columns}
        user_data_df = pd.DataFrame([user_data])

        # Predict and display the result
        if st.button("Predict"):
            prediction = model.predict(user_data_df)[0]
            result = "Malignant" if prediction == 1 else "Benign"
            st.write(f"The predicted diagnosis is: **{result}**")
else:
    st.info("Please upload a CSV file to get started.")

# Footer
st.write("---")
st.write("Developed with ❤️ using Streamlit")
