from transformers import pipeline
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=True,
    device=device,
)


def emotion_scores(sample):
    emotion = classifier(sample)
    return emotion[0]
