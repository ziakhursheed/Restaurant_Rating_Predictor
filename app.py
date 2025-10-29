# app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset to get mapping info
df = pd.read_csv("zomato.csv")

# Clean dataset same way as during training
df = df.dropna(subset=['rate', 'approx_cost(for two people)'])
df = df[df['rate'] != 'NEW']
df = df[df['rate'] != '-']
df['rate'] = df['rate'].apply(lambda x: str(x).split('/')[0]).astype(float)
df['online_order'] = df['online_order'].map({'Yes': 1, 'No': 0})
df['book_table'] = df['book_table'].map({'Yes': 1, 'No': 0})
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].replace(',', '', regex=True).astype(float)

# Encode categories
df['location_code'] = df['location'].astype('category').cat.codes
df['rest_type_code'] = df['rest_type'].astype('category').cat.codes

# Create mapping dictionaries
location_map = dict(zip(df['location'], df['location_code']))
rest_type_map = dict(zip(df['rest_type'], df['rest_type_code']))

# Reverse maps for display
location_names = sorted(location_map.keys())
rest_type_names = sorted(rest_type_map.keys(), key=lambda x: str(x))


st.set_page_config(page_title="Restaurant Rating Predictor", layout="centered")

st.title("🍽️ Restaurant Rating Predictor")
st.write("Enter restaurant details below to predict its average rating:")

# Input fields
online_order = st.selectbox("Online Order Available?", ["Yes", "No"])
book_table = st.selectbox("Book Table Facility?", ["Yes", "No"])
location = st.selectbox("Select Location", location_names)
rest_type = st.selectbox("Select Restaurant Type", rest_type_names)
cost = st.number_input("Approx Cost (for two people)", min_value=100, max_value=10000, step=50)

# Convert inputs to model format
online_order = 1 if online_order == "Yes" else 0
book_table = 1 if book_table == "Yes" else 0
location_code = location_map[location]
rest_type_code = rest_type_map[rest_type]

# Predict button
if st.button("Predict Rating"):
    features = np.array([[online_order, book_table, location_code, rest_type_code, cost]])
    pred = model.predict(features)[0]
    st.success(f"⭐ Predicted Rating: {pred:.2f} / 5.0")

st.markdown("---")
st.write("Built with ❤️ using Python and Streamlit.")
