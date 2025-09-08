from fastapi import FastAPI
from pydantic import BaseModel
import joblib


app = FastAPI()
model = joblib.load("regression.joblib")

class PredictInput(BaseModel):
    size: float
    nb_rooms: int
    garden: int

@app.post("/predict")
async def predict(data: PredictInput):
    predicted_price = model.predict([[data.size, data.nb_rooms, data.garden]])
    return {"price_pred": predicted_price[0]}