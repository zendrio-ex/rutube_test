from flask import Flask, request
from utils import preparation
import model
import torch

model_obj = model.install_model()
labels = model.labels

app = Flask(__name__)


@app.post('/get_label')
def get_label() -> dict:
    image = request.json['image']
    image = torch.tensor(image, dtype=torch.uint8)
    image = preparation.pipeline(image)
    label = model_obj(image.unsqueeze(dim=0)).argmax(dim=-1).item()
    label = labels[label]

    return {"label": label}
