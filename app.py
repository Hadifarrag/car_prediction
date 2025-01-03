import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import os

# Adding a sidebar
with st.sidebar:
    st.image("https://th.bing.com/th/id/OIP.O1SbADJtBced-4q58fh1sAHaE6?w=225&h=180&c=7&r=0&o=5&dpr=2.1&pid=1.7")
    st.title("Car Price Prediction")
    choice = st.radio("Navigation", ["Dataset", "Upload your information"])
    st.info("This project application helps you get the best price for your car")

# Helper functions to match training preprocessing
def convert_engine_to_num(x):
    try:
        return float(x)
    except:
        return None

def convert_maxpowers_to_num(x):
    try:
        return float(x)
    except:
        return None

# Load the dataset
@st.cache_data
def load_data():
    if os.path.exists('Car details v3.csv'):
        return pd.read_csv('Car details v3.csv')
    return None

df = load_data()

# Car information input form
if choice == "Upload your information":
    st.title('Add your car info')
    
    # Create input fields with proper validation
    year = st.number_input("Year", min_value=1900, max_value=2024, value=2003)
    km_driven = st.number_input("Kilometers driven", min_value=0, value=50000)
    fuel = st.selectbox('Fuel', ['Diesel', 'Petrol', 'CNG', 'LPG'])
    seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.selectbox('Transmission', ['Automatic', 'Manual'])
    owner = st.selectbox('Owner', [
        'First Owner',
        'Second Owner',
        'Third Owner',
        'Fourth & Above Owner',
        'Test Drive Car'
    ])
    # Changed to number_input instead of text_input
    engine = st.number_input("Engine (CC)", min_value=500, max_value=8000, value=1600)
    max_power = st.number_input("Max Power (bhp)", min_value=0, max_value=1200, value=80)
    seats = st.number_input("Seats", min_value=2, max_value=10, value=5)

    if st.button('Predict Price'):
        try:
            # Create initial DataFrame with one row
            input_data = {
                'km_driven': km_driven,
                'fuel': fuel,
                'seller_type': seller_type,
                'transmission': transmission,
                'owner': owner,
                'engine': engine,  # Now using direct numeric input
                'max_power': max_power,  # Now using direct numeric input
                'seats': seats
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Apply the same preprocessing as in training
            input_df['engine'] = input_df['engine'].apply(convert_engine_to_num)
            input_df['max_power'] = input_df['max_power'].apply(convert_maxpowers_to_num)
            
            # Calculate age
            input_df['age'] = 2024 - year
            
            # One-hot encode categorical variables
            categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
            input_df = pd.get_dummies(input_df, columns=categorical_cols)
            
            # Ensure all columns from training are present
            if df is not None:
                # Process training data to get column names
                train_df = df.copy()
                train_df['age'] = 2024 - train_df['year']
                train_df.drop('year', axis=1, inplace=True)
                train_df.drop(['mileage', 'name', 'torque', 'selling_price'], axis=1, inplace=True)
                train_df = pd.get_dummies(train_df, columns=categorical_cols)
                
                # Add missing columns with 0s
                for col in train_df.columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                # Reorder columns to match training data
                input_df = input_df[train_df.columns]

            # Load and use the model
            model = xgb.Booster()
            model.load_model('xgb_car_price_model.json')
            
            # Convert to DMatrix
            dmatrix = xgb.DMatrix(input_df)
            
            # Make prediction
            predicted_price = model.predict(dmatrix)[0]
            
            # Display prediction with proper formatting
            st.success(f"The predicted price for your car is: ₹{predicted_price:,.2f}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.info("Please check your input values and try again.")
            # Debug info
            st.write("Error details:", str(e))

elif choice == "Dataset":
    st.title("Model Dataset")
    if df is not None:
        st.dataframe(df)
        st.info("This is the original dataset before preprocessing")
    else:
        st.error("Dataset file not found!")