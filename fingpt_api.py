from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class PredictRequest(BaseModel):
    ticker: str
    date: str
    n_weeks: int
    use_basics: bool


@app.post("/predict")
async def predict(req: PredictRequest):
    # TODO: Replace this with your real FinGPT prediction logic
    info = f"Predicted info for {req.ticker} on {req.date}"
    answer = "0.5"  # Example sentiment score
    return {"info": info, "answer": answer}
