from fastapi import FastAPI, HTTPException
import pandas as pd
from challenge.model import DelayModel
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI()

# Train the model at startup
model = DelayModel()
data = pd.read_csv("data/data.csv") 
X, y = model.preprocess(data, target_column="delay")
model.fit(X, y)  # Train the model

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}

class FlightFeatures(BaseModel):
    OPERA: str
    MES: int = Field(..., ge=1, le=12, description="Month must be between 1 and 12")
    TIPOVUELO: str = Field(..., regex="^(I|N)$", description="TIPOVUELO must be 'I' (International) or 'N'")
    Fecha_I: Optional[str] = Field(None, alias="Fecha-I")
    Fecha_O: Optional[str] = Field(None, alias="Fecha-O")

@app.post("/predict", status_code=200)
async def post_predict(data: dict):
    """
    Receives flight information and returns delay predictions.
    """
    try:
        flights = data.get("flights", None)
        if not flights:
            raise HTTPException(status_code=400, detail="Missing 'flights' key in request data.")

        # Ensure missing fields have default values
        for flight in flights:
            flight.setdefault("Fecha-I", "2024-01-01 00:00:00")
            flight.setdefault("Fecha-O", "2024-01-01 00:00:00")

        # Validate input data
        processed_flights = [FlightFeatures(**flight) for flight in flights]

        # Convert input to DataFrame
        input_data = pd.DataFrame([flight.dict(by_alias=True) for flight in processed_flights])

        # Preprocess input data
        X = model.preprocess(input_data)

        # Predict using trained model
        predictions = model.predict(X)

        return {"predict": predictions}

    except Exception as e:
        print(f"Error in /predict: {e}")
        raise HTTPException(status_code=400, detail=str(e))
