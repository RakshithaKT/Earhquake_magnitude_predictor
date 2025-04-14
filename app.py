import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

st.title("🌍 Earthquake Magnitude Predictor")

# Load your dataset here (replace this path with your local dataset if needed)
@st.cache_data
def load_data():
    df = pd.read_csv("D:\\5th sem\\ML\\Earthquake\\Preprocessed_earthquake dataset (3).csv")
  # Replace this with your CSV file path if needed
    df = df.drop(columns=["Time", "Place"], errors="ignore")  # Drop unwanted columns
    return df

df = load_data()

# Check if 'Mag' column is present
if "Mag" not in df.columns:
    st.error("❌ Dataset must contain a 'Mag' column.")
else:
    # Split features and target
    X_raw = df[["Latitude", "Longitude", "Depth"]]
    y = df["Mag"]

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "Random Forest": RandomForestRegressor(),
        "Linear Regression": LinearRegression(),
        "SVR": SVR(),
        "Decision Tree": DecisionTreeRegressor(),
        "Gradient Boosting": GradientBoostingRegressor()
    }

    # Train and evaluate models
    best_model = None
    best_mse = float("inf")
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_name = name

    st.success(f"✅ Best Model: {best_name}")
    st.info(f"📉 MSE on Test Data: {best_mse:.4f}")

    # Take user input
    st.subheader("🔍 Predict Magnitude")
    lat = st.number_input("Latitude", value=0.0)
    lon = st.number_input("Longitude", value=0.0)
    depth = st.number_input("Depth", value=10.0)

    if st.button("Predict Magnitude"):
        user_input = pd.DataFrame([[lat, lon, depth]], columns=["Latitude", "Longitude", "Depth"])
        user_input_imputed = pd.DataFrame(imputer.transform(user_input), columns=user_input.columns)
        prediction = best_model.predict(user_input_imputed)[0]
        st.success(f"🌋 Predicted Magnitude: {prediction:.2f}")
