from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load your model (replace with MLflow loading if needed)
model = joblib.load("model.pkl")

app = FastAPI(title="Housing Price Predictor")

# Define input schema
class PredictionInput(BaseModel):
    area: float

# Simple GET endpoint to check server
@app.get("/")
def home():
    return {"message": "FastAPI server is running!"}

# POST endpoint for prediction
@app.post("/predict")
def predict(data: PredictionInput):
    area = data.area
    prediction = model.predict([[area]])[0]
    return {"prediction": prediction}
