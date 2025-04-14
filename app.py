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

uploaded_file = st.file_uploader("Upload Earthquake CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if 'Time' in df.columns and 'Place' in df.columns:
        df = df.drop(['Time', 'Place'], axis=1)

    imputer = SimpleImputer(strategy="mean")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    X = df_imputed.drop("Mag", axis=1)
    y = df_imputed["Mag"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(),
        "Linear Regression": LinearRegression(),
        "SVR": SVR(),
        "Decision Tree": DecisionTreeRegressor(),
        "Gradient Boosting": GradientBoostingRegressor()
    }

    best_model = None
    best_mse = float('inf')
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

    st.subheader("🔍 Enter Input for Prediction")

    lat = st.number_input("Latitude", value=0.0)
    lon = st.number_input("Longitude", value=0.0)
    depth = st.number_input("Depth", value=10.0)

    if st.button("Predict Magnitude"):
        input_df = pd.DataFrame([[lat, lon, depth]], columns=["Latitude", "Longitude", "Depth"])
        input_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)
        prediction = best_model.predict(input_imputed)[0]
        st.success(f"🌋 Predicted Magnitude: {prediction:.2f}")
