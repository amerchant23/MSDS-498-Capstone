# 🐾 Pet Adoption Prediction App

This repository hosts a predictive application that estimates the adoption speed of pets listed on PetFinder. Built with **Streamlit** and powered by **scikit-learn**, the app uses structured features and text descriptions to predict how quickly a pet might be adopted and offers suggestions to improve pet descriptions.

---

## 🔍 Project Purpose

The goal of this application is to:

- Predict **AdoptionSpeed** categories (0–4) using a trained `RandomForestClassifier` pipeline.
- Provide **keyword analysis** and **actionable suggestions** to enhance pet descriptions.
- Offer a **simple, browser-based interface** for uploading pet data and viewing predictions.

---

## 📁 Files in This Repository

- `app.py` – Streamlit frontend for user interaction and predictions.  
- `pipeline_rf_protocol4.pkl` – Trained machine learning pipeline.  
- `train.csv`, `breed_labels.csv`, `color_labels.csv`, `state_labels.csv` – Data files used to train the model.  
- `predictive_app.py` – Source notebook containing model training and pipeline export.

---

## ▶️ How to Run the App

### 1. Clone the Repository
```bash
git clone https://github.com/amerchant23/MSDS-498-Capstone.git
cd MSDS-498-Capstone
```

### 2. Create a Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install the Required Packages
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app.py
```

Then open the provided local URL (usually http://localhost:8501) in your browser.

## 💡 How It Works

The app loads a trained pipeline that:

- Processes structured pet features (age, breed, colors, etc.).
- Extracts TF-IDF features from descriptions.
- Makes predictions using a RandomForestClassifier.
- Analyzes descriptions and gives tailored suggestions for improvement.


