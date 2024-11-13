# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import List
import logging
import os
from logging.handlers import RotatingFileHandler

# Create a logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logger = logging.getLogger("equipo16_accel_app")
logger.setLevel(logging.INFO)  # Set the logging level as needed

# Create handlers
c_handler = logging.StreamHandler()  # Console handler
f_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "app.log"),
    maxBytes=5*1024*1024,  # 5 MB
    backupCount=3
)  # File handler with rotation

c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.INFO)

# Create formatters and add them to handlers
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

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
    model_path = os.path.join(os.path.dirname(__file__), "best_model.joblib")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file '{model_path}' does not exist.")
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    try:
        logger.info(f"Loading model from: {model_path}")
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
