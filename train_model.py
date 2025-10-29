# train_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
df = pd.read_csv("zomato.csv")

# Basic cleaning (you can modify later)
df = df.dropna(subset=['rate', 'approx_cost(for two people)'])
df = df[df['rate'] != 'NEW']
# Clean the 'rate' column
df = df[df['rate'].notna()]
df = df[df['rate'] != '-']
df['rate'] = df['rate'].apply(lambda x: str(x).split('/')[0]).astype(float)


# Simplify features
df['online_order'] = df['online_order'].map({'Yes': 1, 'No': 0})
df['book_table'] = df['book_table'].map({'Yes': 1, 'No': 0})
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].replace(',', '', regex=True).astype(float)

# Encode 'location' and 'rest_type'
df['location'] = df['location'].astype('category').cat.codes
df['rest_type'] = df['rest_type'].astype('category').cat.codes

# Define X and y
X = df[['online_order', 'book_table', 'location', 'rest_type', 'approx_cost(for two people)']]
y = df['rate']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n✅ Model trained successfully!")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n💾 Model saved as 'model.pkl' in your project folder.")
