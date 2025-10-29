# Restaurant Rating Predictor

A sleek Machine Learning web app built with Python and Streamlit that predicts restaurant ratings based on user inputs.  
The model is pre-trained, so users can get instant predictions without needing to upload any dataset.

---

## Features

- **Instant Predictions** – Enter details and get predicted ratings immediately.  
- **No Dataset Upload Needed** – The model is already trained and ready to use.  
- **Interactive UI** – Built using Streamlit for a modern and responsive user experience.  
- **Lightweight and Fast** – Works entirely in the browser with no complex setup.  

---

## How It Works

The app takes a few restaurant details such as:

- Online order availability  
- Table booking facility  
- Location  
- Restaurant type  
- Approximate cost for two people  

and predicts the restaurant’s average user rating using a pre-trained machine learning model.

---

## Tech Stack

- **Python 3.9+**  
- **Streamlit** – For building the web interface  
- **Scikit-learn** – For training and predicting with the ML model  
- **Pandas and NumPy** – For data processing and analysis  

---

## Project Structure

Restaurant_Rating_Predictor/
│
├── app.py # Main Streamlit app
├── model.pkl # Pre-trained ML model
├── restaurant_data.csv # Dataset used for model training
├── requirements.txt # Project dependencies
└── README.md # Project documentation

---

## Run Locally

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/Restaurant_Rating_Predictor.git
   ```

2. **Navigate into the project folder**
   ```bash
   cd Restaurant_Rating_Predictor
   ```

3. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

6. Open the local URL displayed in your terminal (typically `http://localhost:8501`).

---


## Deployment

This app can be deployed on:

- Hugging Face Spaces
 (recommended)
- Streamlit Cloud
- Render or Vercel

Simply upload app.py, model.pkl, and requirements.txt — the app will launch automatically.

---

## Future Enhancements

- Add cuisine-based recommendations
- Display model insights and feature importance
- Include user review sentiment analysis

---

## Author

Developed by **Zia Khursheed**  

A passionate learner exploring the intersection of Machine Learning and Practical Web Deployment using Python and Streamlit.


---

> **Note:** The dataset file `restaurant_data.csv` is not included in this repository due to GitHub size limitations.  
> You can download it from [Google Drive](https://drive.google.com/file/d/1ug2-ka6VMmBH_GQ-yyOh28gu7HfaN2Z3/view?usp=sharing).
