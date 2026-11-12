import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from config import MODEL_OUTPUT_DIR


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_OUTPUT_DIR
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_OUTPUT_DIR
)

model.to(device)

model.eval()







def predict_sms(text):

    if not text or not text.strip():
        raise ValueError("SMS message cannot be empty")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

    confidence, predicted_class = probabilities.max(
        dim=-1
    )

    predicted_id = predicted_class.item()

    label = model.config.id2label[predicted_id]

    return {
        "prediction": label,
        "confidence": round(confidence.item(), 4)
    }









if __name__ == "__main__":

    test_messages = [
        "Congratulations! You have won a cash prize. Click here.",
        "Hey, are we still meeting at 6 pm?",
        "Your bank account will be blocked. Verify immediately."
    ]

    for message in test_messages:

        result = predict_sms(message)

        print(f"\nMessage: {message}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}")
