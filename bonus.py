from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

mlm = pipeline("fill-mask", model="bert-base-uncased")

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "BERT Masked LM API is running!"}

@app.post("/predict")
def predict(data: TextInput):
    """
    Expects input like: "The quick brown [MASK] jumps over the lazy dog."
    Returns top predictions for the masked token.
    """
    predictions = mlm(data.text)
    output = [{"token": p["token_str"], "score": p["score"]} for p in predictions]
    return {"input": data.text, "predictions": output}
