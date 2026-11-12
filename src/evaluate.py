import numpy as np
import os
import json
import matplotlib.pyplot as plt

from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments,DataCollatorWithPadding


from sklearn.metrics import classification_report,ConfusionMatrixDisplay,accuracy_score,f1_score,precision_score,recall_score


from src.data_processing import load_and_split_dataset,tokenize_dataset


from config import MODEL_OUTPUT_DIR,TRAINING_OUTPUT_DIR,RESULTS_DIR,EVAL_BATCH_SIZE















data = load_and_split_dataset()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_OUTPUT_DIR
)

data_encoded = tokenize_dataset(
    data,
    tokenizer
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_OUTPUT_DIR
)



data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

eval_args = TrainingArguments(
    output_dir=TRAINING_OUTPUT_DIR,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=eval_args,
    data_collator=data_collator,
    processing_class=tokenizer
)


predictions = trainer.predict(
    data_encoded["test"]
)

y_true = predictions.label_ids

y_pred = np.argmax(
    predictions.predictions,
    axis=1
)

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["HAM", "SPAM"],
    zero_division=0
)

print(report)


os.makedirs(RESULTS_DIR, exist_ok=True)

display = ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    labels=[0, 1],
    display_labels=["HAM", "SPAM"]
)

plt.title("SMS Spam Classification - Test Confusion Matrix")

plt.savefig(
    f"{RESULTS_DIR}/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

metrics = {
    "accuracy": accuracy_score(y_true, y_pred),

    "precision_weighted": precision_score(
        y_true, y_pred, average="weighted"
    ),

    "recall_weighted": recall_score(
        y_true, y_pred, average="weighted"
    ),

    "weighted_f1": f1_score(
        y_true, y_pred, average="weighted"
    ),

    "macro_f1": f1_score(
        y_true, y_pred, average="macro"
    ),

    "spam_recall": recall_score(
        y_true, y_pred, pos_label=1
    )
}

with open(
    f"{RESULTS_DIR}/test_metrics.json",
    "w"
) as file:

    json.dump(metrics, file, indent=4)

with open(
    f"{RESULTS_DIR}/classification_report.txt",
    "w"
) as file:

    file.write(report)

print(metrics)