from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
from typing import List
from pathlib import Path

from src.model import AdvancedModelTrainer, RealTimePredictor
from src.utils import load_config

# --- Configuration and Model Loading ---
config = load_config("config/settings.yaml")
app = FastAPI(
    title="ML-TA API",
    description="API for the Machine Learning Technical Analysis trading system.",
    version="0.1.0"
)

# Global objects for holding the loaded model and predictor
predictor_registry = {}

@app.on_event("startup")
def load_model():
    """Load the trained model and feature engineer at startup."""
    model_path = Path(config.model.get('save_path', 'models/')) / "advanced_model.joblib"
    if not model_path.exists():
        raise RuntimeError(f"Model file not found at {model_path}. Please run tests to train and save the model.")
        
    try:
        trainer = AdvancedModelTrainer(config)
        trainer.load_models(model_path)
        
        # Create a predictor for each model in the trainer
        for label_name, model in trainer.models.items():
            if label_name in trainer.feature_engineers:
                feature_engineer = trainer.feature_engineers[label_name]
                predictor_registry[label_name] = RealTimePredictor(config, model, feature_engineer)
        
        print(f"✓ Models loaded successfully. Available predictors: {list(predictor_registry.keys())}")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        # Exit if model loading fails, as the API is not usable
        raise RuntimeError(f"Could not load model: {e}") from e


# --- API Models ---
class Candle(BaseModel):
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

class PredictionRequest(BaseModel):
    candles: List[Candle]
    label_name: str = "label_class_1"  # Default label, can be overridden

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Welcome to the ML-TA API!"}

@app.get("/predictors")
async def get_predictors():
    """Return a list of available predictors (trained models)."""
    return {"available_predictors": list(predictor_registry.keys())}

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make a real-time prediction based on candlestick data."""
    if not predictor_registry:
        raise HTTPException(status_code=503, detail="Predictor not available. Model may not be loaded.")

    predictor = predictor_registry.get(request.label_name)
    if not predictor:
        raise HTTPException(status_code=404, detail=f"Predictor for label '{request.label_name}' not found.")

    try:
        # Convert Pydantic models to a pandas DataFrame
        df = pd.DataFrame([c.dict() for c in request.candles])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        # Get prediction
        result = predictor.predict(df)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
