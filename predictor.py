from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from labels import id2label


class EmotionPredictor:

    def __init__(self):

        model_name = "ShahdAbdelnasser1/egybert-emotion-classifier"

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)

        idx = probs.argmax(dim=1).item()
        confidence = probs[0][idx].item()

        return {
            "emotion": id2label[idx],
            "confidence": round(confidence, 4),
        }
