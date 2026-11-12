
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from config import (
    TRAINING_OUTPUT_DIR,
    MODEL_OUTPUT_DIR,
    NUM_EPOCHS,
    LEARNING_RATE,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    WEIGHT_DECAY,
    RANDOM_SEED
)

from src.data_processing import prepare_data
from src.model import load_model


def compute_metrics(pred):

    labels = pred.label_ids
    predictions = pred.predictions.argmax(axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),

        "precision": precision_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "recall": recall_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "weighted_f1": f1_score(
            labels,
            predictions,
            average="weighted"
        ),

        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro"
        ),

        "spam_recall": recall_score(
            labels,
            predictions,
            pos_label=1,
            average="binary",
            zero_division=0
        )
    }


def main():

    
    data_encoded, tokenizer = prepare_data()

    
    model = load_model()

    
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    
    training_args = TrainingArguments(
        output_dir=TRAINING_OUTPUT_DIR,

        num_train_epochs=NUM_EPOCHS,

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,

        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        weight_decay=WEIGHT_DECAY,

        eval_strategy="epoch",

        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="eval_macro_f1",

        greater_is_better=True,

        logging_strategy="epoch",

        save_total_limit=2,

        seed=RANDOM_SEED,

        data_seed=RANDOM_SEED,

        report_to="none"
    )

    
    trainer = Trainer(
        model=model,

        args=training_args,

        train_dataset=data_encoded["train"],

        eval_dataset=data_encoded["validation"],

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics
    )

    
    trainer.train()

    
    trainer.save_model(MODEL_OUTPUT_DIR)

    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)


if __name__ == "__main__":
    main()
