import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

# 1. Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="Backend API for California House Price Prediction using ANN model",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load trained model, encoder, and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")
ENCODER_PATH = os.path.join(BASE_DIR, "ocean_proximity_encoder.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

try:
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessor artifacts: {str(e)}")


# Define Pydantic schema for prediction input
class HouseInput(BaseModel):
    longitude: float = Field(..., description="Longitude coordinate")
    latitude: float = Field(..., description="Latitude coordinate")
    housing_median_age: float = Field(..., ge=0, description="Median age of houses in block")
    total_rooms: float = Field(..., ge=0, description="Total number of rooms")
    total_bedrooms: float = Field(..., ge=0, description="Total number of bedrooms")
    population: float = Field(..., ge=0, description="Total population in block")
    households: float = Field(..., ge=0, description="Total number of households")
    median_income: float = Field(..., ge=0, description="Median income of households (in tens of thousands USD)")
    ocean_proximity: str = Field(..., description="Proximity to the ocean")


@app.get("/")
def serve_frontend():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "House Price Prediction API is running. Frontend file not found."}


# 3. Create POST /predict endpoint
@app.post("/predict")
def predict(data: HouseInput):
    try:
        # 4. Convert input to a pandas DataFrame
        user_data = pd.DataFrame([{
            "longitude": data.longitude,
            "latitude": data.latitude,
            "housing_median_age": data.housing_median_age,
            "total_rooms": data.total_rooms,
            "total_bedrooms": data.total_bedrooms,
            "population": data.population,
            "households": data.households,
            "median_income": data.median_income,
            "ocean_proximity": data.ocean_proximity
        }])

        # 5. Use encoder.transform() to encode ocean_proximity (NEVER fit or fit_transform)
        encoded_features = encoder.transform(user_data[["ocean_proximity"]])
        if hasattr(encoded_features, "toarray"):
            encoded_features = encoded_features.toarray()

        # 6. Keep encoded columns in exact training order
        encoded_df = pd.DataFrame(
            encoded_features,
            columns=encoder.get_feature_names_out(["ocean_proximity"]),
            index=user_data.index
        )

        # 7. Combine numerical + encoded features in exact training order
        processed_df = pd.concat(
            [user_data.drop(columns=["ocean_proximity"]), encoded_df],
            axis=1
        )

        # 8. Use scaler.transform() (NEVER fit or fit_transform)
        scaled_data = scaler.transform(processed_df)

        # 9. Pass scaled data to model.predict()
        raw_prediction = model.predict(scaled_data)
        predicted_value = float(raw_prediction[0][0])

        # 10. Return prediction JSON response
        return {"prediction": predicted_value}

    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))
