from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=True,
)


def emotion_scores(sample):
    emotion = classifier(sample)
    return emotion[0]
