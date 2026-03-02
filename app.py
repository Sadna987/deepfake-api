from fastapi import FastAPI
from pydantic import BaseModel
import requests
from PIL import Image
from io import BytesIO

from model import predict_image

app = FastAPI()


class ImageRequest(BaseModel):
    image_url: str


@app.get("/")
def home():
    return {"message":"Deepfake detection API running"}


@app.post("/predict")
def predict(req: ImageRequest):

    response = requests.get(req.image_url)

    image = Image.open(BytesIO(response.content)).convert("RGB")

    label, confidence = predict_image(image)

    return {
        "prediction": label,
        "confidence": confidence
    }
