from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import pandas as pd
from typing import List, Dict
from pathlib import Path
import time

from src.model import AdvancedModelTrainer, RealTimePredictor
from src.utils import load_config

# --- Application State ---
class AppState:
    def __init__(self):
        self.is_model_loaded = False
        self.predictors: Dict[str, RealTimePredictor] = {}
        self.load_error = None

app_state = AppState()

# --- Configuration and App Initialization ---
config = load_config("config/settings.yaml")
app = FastAPI(
    title="ML-TA API",
    description="API for the Machine Learning Technical Analysis trading system.",
    version="0.1.0"
)

# --- Background Model Loading ---
def do_load_model(state: AppState):
    """The actual model loading logic to be run in the background."""
    try:
        model_path = Path(config.model.get('save_path', 'models/')) / "advanced_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}.")

        print("Background task: Starting model loading...")
        start_time = time.time()
        
        trainer = AdvancedModelTrainer(config)
        trainer.load_models(model_path)
        
        for label_name, model in trainer.models.items():
            if label_name in trainer.feature_engineers:
                feature_engineer = trainer.feature_engineers[label_name]
                state.predictors[label_name] = RealTimePredictor(config, model, feature_engineer)
        
        state.is_model_loaded = True
        duration = time.time() - start_time
        print(f"✓ Background task: Models loaded successfully in {duration:.2f}s. Available predictors: {list(state.predictors.keys())}")

    except Exception as e:
        print(f"✗ Background task: Error loading model: {e}")
        state.load_error = str(e)

@app.on_event("startup")
async def startup_event(background_tasks: BackgroundTasks):
    """On startup, trigger the background task to load the model."""
    background_tasks.add_task(do_load_model, app_state)

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
    label_name: str = "label_class_1"

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Welcome to the ML-TA API!"}

@app.get("/status")
async def get_status():
    """Check the status of the model loading process."""
    if app_state.load_error:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {app_state.load_error}")
    if not app_state.is_model_loaded:
        return {"status": "loading", "message": "Model is currently loading in the background."}
    return {
        "status": "ready",
        "available_predictors": list(app_state.predictors.keys())
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make a real-time prediction based on candlestick data."""
    if not app_state.is_model_loaded:
        raise HTTPException(status_code=503, detail="Model is not ready. Please check /status for updates.")
    if app_state.load_error:
        raise HTTPException(status_code=500, detail=f"Model could not be loaded: {app_state.load_error}")

    predictor = app_state.predictors.get(request.label_name)
    if not predictor:
        raise HTTPException(status_code=404, detail=f"Predictor for label '{request.label_name}' not found.")

    try:
        df = pd.DataFrame([c.dict() for c in request.candles])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        result = predictor.predict(df)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
