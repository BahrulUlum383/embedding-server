from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from model_loader import load_model

app = FastAPI()
model = load_model()

class EmbedRequest(BaseModel):
    texts: List[str]

@app.post("/embed")
def embed(req: EmbedRequest):
    embeddings = model.encode(req.texts).tolist()
    return {"embeddings": embeddings}
