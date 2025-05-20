# 🐾 Pet Adoption Prediction App

This repository hosts a predictive application that estimates the adoption speed of pets listed on PetFinder. Built with Streamlit and powered by scikit-learn, the app uses structured features and text descriptions to predict how quickly a pet might be adopted and also offers suggestions to improve pet descriptions.

## 🔍 Project Purpose

The goal of this application is to:

- Use a trained RandomForestClassifier pipeline to predict AdoptionSpeed categories (0–4).

- Provide keyword analysis and actionable suggestions to improve pet descriptions.

- Offer a simple, browser-based interface for uploading pet data and viewing predictions.

## 📁 Files in This Repo

- app.py – Streamlit frontend for user interaction and prediction.

- pipeline_rf_protocol4.pkl – Trained machine learning pipeline.

- train.csv, breed_labels.csv, color_labels.csv, state_labels.csv – Data used to train the model.

- predictive_app.py – Source notebook with full model training and pipeline export.

## ▶️ How to Run the App

1. Clone the Repository

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/amerchant23/MSDS-498-Capstone.git)
cd your-repo-name

2. Create a Virtual Environment (optional but recommended)

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. Install the Required Packages

pip install -r requirements.txt

4. Run the App

streamlit run app.py
Then open the provided local URL (usually http://localhost:8501) in your browser.

## 💡 How It Works

The app loads a trained pipeline that:

- Processes structured pet features (age, breed, colors, etc.).

- Extracts TF-IDF features from descriptions.

- Makes predictions using a RandomForestClassifier.

- Analyzes descriptions and gives tailored suggestions for improvement.


