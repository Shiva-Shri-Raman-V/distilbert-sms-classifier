from transformers import AutoModelForSequenceClassification
from config import MODEL_CHECKPOINT,NUM_LABELS,ID2LABEL,LABEL2ID


def load_model():

    model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID
    )

    return model







