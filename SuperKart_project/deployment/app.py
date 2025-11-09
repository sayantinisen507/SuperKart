import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ----------------------------
# Download model from Hugging Face
# ----------------------------
model_path = hf_hub_download(
    repo_id="Sayantini/SuperKart-model",
    filename="SuperKart_model_v1.joblib"
)

# Load model
model = joblib.load(model_path)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🛒 SuperKart Sales Prediction")
st.write("Enter product and store details to predict total sales value for the product.")

# ----------------------------
# Collect user input
# ----------------------------
Product_Weight = st.number_input("Product Weight (in kg)", min_value=0.0, value=10.0)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.slider("Product Allocated Area (fraction of store area)", 0.0, 1.0, 0.05)
Product_Type = st.selectbox("Product Type", [
    "Frozen Foods", "Dairy", "Canned", "Baking Goods", "Health and Hygiene", 
    "Snack Foods", "Soft Drinks", "Meat", "Household", "Others"
])
Product_MRP = st.number_input("Product MRP (₹)", min_value=1.0, value=100.0)
Store_Id = st.text_input("Store ID (e.g., OUT001)")
Store_Establishment_Year = st.number_input("Store Establishment Year", min_value=1980, max_value=2025, value=2005)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", [
    "Supermarket Type1", "Supermarket Type2", "Supermarket Type3", 
    "Departmental Store", "Food Mart"
])

# ----------------------------
# Prepare input data
# ----------------------------
input_data = pd.DataFrame([{
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_Type": Product_Type,
    "Product_MRP": Product_MRP,
    "Store_Id": Store_Id,
    "Store_Establishment_Year": Store_Establishment_Year,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type
}])

# ----------------------------
# Predict button
# ----------------------------
if st.button("Predict Sales"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Product Sales: ₹{prediction:,.2f}")
