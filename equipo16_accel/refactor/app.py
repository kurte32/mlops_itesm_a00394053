# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import List
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Predicción de configuración",
    description="API para predecir la clase de vibración-Equipo 16",
    version="1.0.0"
)

# Load the model at startup
model = None

@app.on_event("startup")
def load_model():
    """
    Load the model during startup
    """
    global model
    model_path = "best_model.joblib"  # Adjust path if necessary
    if not os.path.exists(model_path):
        logger.error(f"Model file '{model_path}' does not exist.")
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading the model: {e}")
        raise e  # Re-raise to prevent the app from starting without a model

# Define the input data schema
class PredictionRequest(BaseModel):
    """
    Input data schema
    x: float
    y: float
    z: float
    pctid: float
    vibration_magnitude: float
    """
    x: float = Field(..., example=1.23, description="Feature x")
    y: float = Field(..., example=4.56, description="Feature y")
    z: float = Field(..., example=7.89, description="Feature z")
    pctid: float = Field(..., example=0.12, description="Feature pctid")
    vibration_magnitude: float = Field(..., example=3.45, description="Feature vibration_magnitude")

class PredictionResponse(BaseModel):
    """
    Output data schema
    prediction: int
    probability: float
    """
    prediction: int
    probability: float

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Make a prediction",
    description="Provide feature values to get a prediction."
)
def predict(request: PredictionRequest):
    """
    Make a prediction using the input data
    """
    if model is None:
        logger.error("Model is not loaded.")
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        # Create a DataFrame from the input
        input_data = pd.DataFrame([request.dict()])
        logger.debug(f"Input DataFrame: {input_data}")

        # Make prediction
        prediction = model.predict(input_data.values)[0]
        logger.info(f"Prediction: {prediction}")

        # Get prediction probability if supported
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data.values)[0]
            probability = float(np.max(probabilities))
            logger.info(f"Prediction Probability: {probability}")
        else:
            probability = 1.0  # Default probability if not available
            logger.info("Model does not support probability estimates. Setting probability to 1.0.")

        return PredictionResponse(prediction=int(prediction), probability=probability)

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")
