#api.py
import fastapi
import pandas as pd
from model import DelayModel
from pydantic import BaseModel
from typing import List

app = fastapi.FastAPI()

# Initialize model on startup
model = DelayModel()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


# Define input data schema
class FlightFeatures(BaseModel):
    OPERA: str
    MES: int
    TIPOVUELO: str
    Fecha_I: str
    Fecha_O: str


@app.post("/predict", status_code=200)
async def post_predict(data: List[FlightFeatures]) -> dict:
    """
    Receives flight information and returns delay predictions.
    """
    # Convert input to DataFrame
    input_data = pd.DataFrame([item.dict() for item in data])

    # Preprocess input data
    X = model.preprocess(input_data)

    # Predict
    predictions = model.predict(X)

    return {"predictions": predictions}
