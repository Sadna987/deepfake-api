from fastapi import FastAPI
from pydantic import BaseModel
import requests
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms

from model import CLoGNet   # import your model

app = FastAPI()

# -----------------------------
# Load Model
# -----------------------------
device = torch.device("cpu")

model = CLoGNet()
model.load_state_dict(torch.load("clognet_weights.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# -----------------------------
# Request Schema
# -----------------------------
class ImageRequest(BaseModel):
    image_url: str


# -----------------------------
# Prediction Function
# -----------------------------
def run_model(image):

    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

    confidence, pred = torch.max(probs, dim=1)

    label = "Fake" if pred.item() == 1 else "Real"

    return label, confidence.item()


# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/predict")
def predict(req: ImageRequest):

    response = requests.get(req.image_url)

    image = Image.open(BytesIO(response.content)).convert("RGB")

    label, confidence = run_model(image)

    return {
        "prediction": label,
        "confidence": confidence
    }
