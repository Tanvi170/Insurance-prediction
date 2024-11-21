import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
@st.cache
def load_data():
    data = pd.read_csv('insurance.csv')
    data.replace({'sex': {'male': 0, 'female': 1},
                  'smoker': {'yes': 0, 'no': 1},
                  'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)
    return data

# Preprocess data
def preprocess_data(data):
    X = data.drop(columns='charges', axis=1)
    y = data['charges']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

# Train model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Streamlit App
st.title("Medical Insurance Charge Prediction")

# Load and preprocess data
insurance_data = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(insurance_data)

# Train the model
model = train_model(X_train, y_train)

# Sidebar for user input
st.sidebar.header("Input Features")
age = st.sidebar.slider("Age", int(insurance_data["age"].min()), int(insurance_data["age"].max()), 30)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
bmi = st.sidebar.slider("BMI", float(insurance_data["bmi"].min()), float(insurance_data["bmi"].max()), 25.0)
children = st.sidebar.slider("Children", int(insurance_data["children"].min()), int(insurance_data["children"].max()), 0)
smoker = st.sidebar.selectbox("Smoker", ["Yes", "No"])
region = st.sidebar.selectbox("Region", ["Southeast", "Southwest", "Northeast", "Northwest"])

# Prepare input data
input_data = np.array([[
    age,
    1 if sex == "Female" else 0,
    bmi,
    children,
    0 if smoker == "Yes" else 1,
    {"Southeast": 0, "Southwest": 1, "Northeast": 2, "Northwest": 3}[region]
]])
input_data_scaled = scaler.transform(input_data)

# Predict
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data_scaled)
    st.subheader("Predicted Insurance Charges:")
    st.write(f"${prediction[0]:,.2f}")

# Display model performance
if st.checkbox("Show Model Performance"):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R-Squared Score: {r2:.2f}")
