from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Optional
import os

app = FastAPI(title="Pet Adoption Speed Predictor", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and columns
pipeline_rf = None
saved_columns = None

@app.on_event("startup")
async def load_model():
    global pipeline_rf, saved_columns
    try:
        # Load your model (make sure to upload pipeline_rf_protocol4.pkl to your repo)
        pipeline_rf, saved_columns = joblib.load("pipeline_rf_protocol4.pkl")
        print("Model loaded successfully!")
        print(f"Expected columns: {saved_columns}")
    except Exception as e:
        print(f"Error loading model: {e}")

class PetData(BaseModel):
    Type: str
    Name: Optional[str] = ""
    Age: int
    Gender: str
    MaturitySize: str
    FurLength: str
    Vaccinated: str
    Dewormed: str
    Sterilized: str
    Health: str
    Quantity: int = 1
    Fee: int
    RescuerID: str
    VideoAmt: int = 0
    Description: str
    PetID: str
    PhotoAmt: float
    StateName: str
    Breed1Type: Optional[float] = None
    MainBreed: str
    Breed2Type: Optional[float] = None
    SecondBreed: Optional[str] = None
    ColorName1: str
    ColorName2: Optional[str] = None
    ColorName3: Optional[str] = None
    DescriptionLength: int
    HasName: int
    IsMixBreed: int

@app.get("/")
async def root():
    return {"message": "Pet Adoption Speed Predictor API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": pipeline_rf is not None,
        "expected_features": len(saved_columns) if saved_columns else 0
    }

@app.post("/predict")
async def predict_adoption_speed(pet_data: PetData):
    global pipeline_rf, saved_columns
    
    if pipeline_rf is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert PetData to dictionary
        data_dict = pet_data.dict()
        
        # Create DataFrame with the expected columns
        df = pd.DataFrame([data_dict])
        
        # Ensure all expected columns are present
        for col in saved_columns:
            if col not in df.columns:
                df[col] = None
        
        # Reorder columns to match training data
        df = df[saved_columns]
        
        # Make prediction
        prediction = pipeline_rf.predict(df)[0]
        prediction_proba = pipeline_rf.predict_proba(df)[0]
        
        # Convert prediction to readable format
        adoption_speed_map = {
            0: "Same Day",
            1: "Within Week", 
            2: "Within Month",
            3: "2-3 Months",
            4: "No Adoption"
        }
        
        adoption_speed = adoption_speed_map.get(prediction, "Unknown")
        confidence = float(max(prediction_proba))
        
        # Estimate days based on category
        days_map = {
            0: np.random.randint(1, 2),      # Same Day
            1: np.random.randint(2, 7),      # Within Week  
            2: np.random.randint(8, 30),     # Within Month
            3: np.random.randint(31, 90),    # 2-3 Months
            4: np.random.randint(91, 365)    # No Adoption
        }
        
        estimated_days = int(days_map.get(prediction, 30))
        
        return {
            "adoptionSpeed": adoption_speed,
            "confidence": confidence,
            "daysToAdoption": estimated_days,
            "prediction_class": int(prediction),
            "class_probabilities": prediction_proba.tolist()
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
