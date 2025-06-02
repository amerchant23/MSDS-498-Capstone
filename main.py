from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import re

# Load the model and training columns
pipeline, training_columns = joblib.load("pipeline_rf_protocol4.pkl")

# FastAPI app instance
app = FastAPI(title="Pet Adoption Predictor API")

# --------- Utility Functions ---------
def clean_description(text):
    text = re.sub(r"<.*?>", "", str(text))  # Remove HTML
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\\s]", "", text)  # Remove special characters
    return text

def analyze_description(text):
    keywords = re.findall(r"[a-zA-Z0-9]+", text.lower())
    stop_words = set(['the', 'a', 'is', 'in', 'it', 'and', 'of', 'to', 'be', 'with', 'for'])
    filtered_keywords = [word for word in keywords if word not in stop_words and len(word) > 2]
    return {"keywords": filtered_keywords, "sentiment": 0.0}

def generate_suggestions(pet_info, description, analysis_results):
    suggestions = []
    keywords = analysis_results["keywords"]

    positive_keywords = ["friendly", "playful", "loving", "gentle", "loyal", "sweet", "happy"]
    missing_keywords = [k for k in positive_keywords if k not in keywords]
    if missing_keywords:
        suggestions.append(f"Consider adding positive words like: {', '.join(missing_keywords)}")

    if pet_info.Type.lower() == "dog":
        if "good with kids" not in description.lower():
            suggestions.append("If the dog is good with children, mention 'good with kids'.")
        if "loves walks" not in description.lower():
            suggestions.append("If the dog enjoys walks, mention 'loves walks'.")
    elif pet_info.Type.lower() == "cat":
        if "affectionate" not in description.lower():
            suggestions.append("If the cat is affectionate, mention 'affectionate'.")
        if "clean" not in description.lower():
            suggestions.append("Cats are typically clean, you can mention that.")

    if pet_info.MainBreed and pet_info.MainBreed.lower() not in description.lower():
        suggestions.append(f"Consider highlighting the breed: {pet_info.MainBreed}.")

    if pet_info.Age < 6:
        suggestions.append("Emphasize that this young pet is playful and energetic.")
    elif pet_info.Age > 72:
        suggestions.append("Emphasize that this senior pet is calm and loving.")

    return suggestions

# --------- Request Schema ---------
class PetInput(BaseModel):
    Type: str
    Gender: str
    MaturitySize: str
    FurLength: str
    Vaccinated: str
    Dewormed: str
    Sterilized: str
    Health: str
    StateName: str
    Breed1Type: str
    Breed2Type: Optional[str] = None
    MainBreed: str
    SecondBreed: Optional[str] = None
    ColorName1: str
    ColorName2: Optional[str] = None
    ColorName3: Optional[str] = None
    Age: int
    Quantity: int
    Fee: float
    VideoAmt: int
    PhotoAmt: int
    Name: Optional[str] = None
    Description: str

# --------- Prediction Endpoint ---------
@app.post("/predict")
def predict(pet: PetInput):
    data = pet.dict()
    data["HasName"] = 0 if not data["Name"] or data["Name"].strip() == "" else 1
    data["IsMixBreed"] = 0 if not data["SecondBreed"] else 1
    data["DescriptionLength"] = len(data["Description"])
    cleaned_description = clean_description(data["Description"])
    data["Description"] = cleaned_description

    df = pd.DataFrame([data])

    for col in training_columns:
        if col not in df.columns:
            df[col] = np.nan

    df = df[training_columns]

    prediction = pipeline.predict(df)[0]

    analysis_results = analyze_description(cleaned_description)
    suggestions = generate_suggestions(pet, data["Description"], analysis_results)

    return {
        "predicted_adoption_speed": int(prediction),
        "suggestions": suggestions
    }
