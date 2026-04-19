from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import torch
import json

MODEL_DIR = "Fatima1412/ingredients-distilbert-classifier"

tokenizer = None
model = None
id2label = None

app = FastAPI(title="Ingredients Classifier API")

@app.on_event("startup")
def load_model():
    global tokenizer, model, id2label

    try:
        # Load tokenizer + model from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()

        # Download labels_mapping.json from Hugging Face
        mapping_path = hf_hub_download(
            repo_id=MODEL_DIR,
            filename="labels_mapping.json"
        )

        with open(mapping_path, "r") as f:
            id2label = {int(k): v for k, v in json.load(f).items()}

        print("Model and label mapping loaded successfully.")

    except Exception as e:
        print("⚠ WARNING: Model not loaded. Check Hugging Face repo.")
        print(e)


class IngredientsRequest(BaseModel):
    ingredients: str


class PredictionResponse(BaseModel):
    label_id: int
    label_name: str
    probabilities: List[float]


def predict_ingredients(text):
    inputs = tokenizer(text, return_tensors="pt")

    # DistilBERT does NOT accept token_type_ids
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).tolist()[0]
    label_id = logits.argmax(dim=-1).item()
    label_name = id2label[label_id]

    return label_id, label_name, probs


@app.post("/predict", response_model=PredictionResponse)
def predict(req: IngredientsRequest):
    label_id, label_name, probs = predict_ingredients(req.ingredients)
    return PredictionResponse(
        label_id=label_id,
        label_name=label_name,
        probabilities=probs
    )
