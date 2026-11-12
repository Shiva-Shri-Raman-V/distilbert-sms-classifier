MODEL_CHECKPOINT = "distilbert-base-uncased"

NUM_LABELS = 2

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
NUM_EPOCHS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

RANDOM_SEED = 42

MODEL_OUTPUT_DIR = "models/distilbert-sms"

LABEL2ID = {
    "HAM": 0,
    "SPAM": 1
}



ID2LABEL = {
    0: "HAM",
    1: "SPAM"
}


TRAINING_OUTPUT_DIR = "models/checkpoints"

RESULTS_DIR = "results"