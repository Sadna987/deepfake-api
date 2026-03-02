from fastapi import FastAPI
from pydantic import BaseModel
import requests
from io import BytesIO

app = FastAPI()

class ImageRequest(BaseModel):
    image_url: str


@app.post("/predict")
def predict(data: ImageRequest):

    response = requests.get(data.image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        prob = torch.softmax(output, dim=1)

    confidence = prob.max().item()
    prediction = prob.argmax().item()

    label = "Real" if prediction == 0 else "Fake"

    return {
        "prediction": label,
        "confidence": round(confidence, 4)
    }
