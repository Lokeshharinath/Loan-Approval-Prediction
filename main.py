import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import base64

# Function to encode the image as base64
def get_base64_image(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Function to add a background image
def add_background_image(image_path):
    base64_image = get_base64_image(image_path)
    st.markdown(
        f"""
        <style>
            body {{
                background-image: url("data:image/jpeg;base64,{base64_image}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
                color: white;
            }}
            .header {{
                text-align: center;
                font-size: 40px;
                font-weight: bold;
                color: #4CAF50;
            }}
            .input-container {{
                margin: 20px 0px;
                padding: 20px;
                background-color: rgba(255, 255, 255, 0.85);
                border-radius: 10px;
                color: black;
            }}
            .status-approved {{
                background-color: rgba(60, 179, 113, 0.9);
                padding: 10px;
                border-radius: 8px;
                color: white;
                font-weight: bold;
            }}
            .status-declined {{
                background-color: rgba(255, 69, 0, 0.9);
                padding: 10px;
                border-radius: 8px;
                color: white;
                font-weight: bold;
            }}
            button {{
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
                cursor: pointer;
            }}
            button:hover {{
                background-color: #45a049;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Load and preprocess the dataset
@st.cache_data
def load_data():
    file_path = 'loan_approval_dataset.csv'  # Ensure this file is in the same directory
    loan_data = pd.read_csv(file_path)

    # Clean column names
    loan_data.columns = loan_data.columns.str.strip().str.lower().str.replace(" ", "_")

    # Create a target variable influenced by multiple features
    loan_data['approval_chances'] = (
                                            0.5 * loan_data['credit_score'] / 850 +
                                            0.2 * loan_data['income_annum'] / loan_data['income_annum'].max() +
                                            0.15 * loan_data['loan_amount'] / loan_data['loan_amount'].max() +
                                            0.1 * loan_data['residential_assets_value'] / loan_data['residential_assets_value'].max() +
                                            0.05 * loan_data['bank_asset_value'] / loan_data['bank_asset_value'].max()
                                    ) * 100  # Scale to percentage
    return loan_data

# Train the regression model
@st.cache_data
def train_model(data):
    features = [
        'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
        'credit_score', 'residential_assets_value', 'commercial_assets_value',
        'luxury_assets_value', 'bank_asset_value'
    ]
    target = 'approval_chances'

    X = data[features]
    y = data[target]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Train Gradient Boosting Regressor with regularization
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,  # Enforce balanced splits
        random_state=42
    )
    model.fit(X_train, y_train)
    return model, scaler

# Function to validate inputs and apply logic
def validate_and_predict(inputs, model, scaler):
    # Unpack inputs
    (no_of_dependents, income_annum, loan_amount, loan_term, credit_score,
     residential_assets_value, commercial_assets_value, luxury_assets_value,
     bank_asset_value) = inputs

    # Validation rules
    decline_reasons = []
    if credit_score < 600:
        decline_reasons.append("Credit score is below 600.")
    if loan_amount > (0.5 * income_annum):
        decline_reasons.append("Debt-to-income ratio exceeds 50%.")
    total_assets = (residential_assets_value + commercial_assets_value +
                    bank_asset_value)
    if total_assets < (0.2 * loan_amount):
        decline_reasons.append("Combined asset value is less than 20% of the loan amount.")

    if decline_reasons:
        return None, None, decline_reasons

    # Scale the inputs
    inputs_scaled = scaler.transform([inputs])

    # Predict approval chances
    approval_chances = model.predict(inputs_scaled)[0]

    # Ensure approval chances are between 0 and 100
    approval_chances = max(0, min(100, approval_chances))

    # Map approval chances to days (higher chances = fewer days)
    predicted_days = max(6, min(25, 25 - (approval_chances / 100) * 19))
    return approval_chances, round(predicted_days), None

# Streamlit app
def main():
    add_background_image("./static/background.jpg")  # Set the background image

    st.markdown('<div class="header">Loan Approval Prediction</div>', unsafe_allow_html=True)

    # Load data and train model
    data = load_data()
    model, scaler = train_model(data)

    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.header("Enter Applicant Details")
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
    income_annum = st.number_input("Annual Income (in $)", min_value=20000, step=1000)
    loan_amount = st.number_input("Loan Amount (in $)", min_value=10000, step=1000)
    loan_term = st.number_input("Loan Term (in months)", min_value=12, max_value=360, step=1)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
    residential_assets_value = st.number_input("Residential Asset Value (in $)", min_value=0, step=1000)
    commercial_assets_value = st.number_input("Commercial Asset Value (in $)", min_value=0, step=1000)
    luxury_assets_value = st.number_input("Luxury Asset Value (in $)", min_value=0, step=1000)
    bank_asset_value = st.number_input("Bank Asset Value (in $)", min_value=5000, step=1000)
    st.markdown('</div>', unsafe_allow_html=True)

    # Buttons
    if st.button("Predict Loan Approval Status"):
        inputs = [
            no_of_dependents, income_annum, loan_amount, loan_term,
            credit_score, residential_assets_value, commercial_assets_value,
            luxury_assets_value, bank_asset_value
        ]
        approval_chances, predicted_days, decline_reasons = validate_and_predict(inputs, model, scaler)
        if decline_reasons:
            st.markdown('<div class="status-declined"><strong>Loan Status:</strong> Declined</div>', unsafe_allow_html=True)
            for reason in decline_reasons:
                st.error(f"- {reason}")
        else:
            st.markdown('<div class="status-approved"><strong>Loan Status:</strong> Approved</div>', unsafe_allow_html=True)
            st.markdown(f"<div class='status-approved'><strong>Estimated Loan Approval Time:</strong> {predicted_days} days</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
