# SMS Spam Classification using DistilBERT

A binary text classification project that fine-tunes a pretrained DistilBERT model to classify SMS messages as **HAM (legitimate)** or **SPAM (unwanted)** using Hugging Face Transformers and PyTorch.

## 1. Project Overview

This project uses transfer learning to build an SMS spam classifier.

Instead of training a language model from scratch, it uses the pretrained `distilbert-base-uncased` model and fine-tunes it on an SMS spam dataset.

The model learns to distinguish between legitimate messages and spam messages based on their textual content.

**Example:**

| SMS Message | Prediction |
|---|---|
| Hey, are we still meeting at 6 pm? | HAM |
| Congratulations! You have won a cash prize. Click here. | SPAM |
| Can you send me yesterday's notes? | HAM |
| Your bank account will be blocked. Verify immediately. | SPAM |

## 2. Project Objectives

- Understand and explore an SMS classification dataset.
- Perform exploratory data analysis (EDA).
- Preprocess and tokenize text using a pretrained DistilBERT tokenizer.
- Fine-tune DistilBERT for binary classification.
- Evaluate the model using multiple classification metrics.
- Generate a confusion matrix and classification report.
- Build a reusable prediction function for new SMS messages.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | Pretrained DistilBERT and Trainer |
| Hugging Face Datasets | Dataset loading and processing |
| Scikit-learn | Evaluation metrics |
| Pandas | Data analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |

## 4. Dataset

**Dataset:** `sms_spam` from Hugging Face Datasets.

The dataset contains 5,574 SMS messages with two classes:

- `0` – HAM
- `1` – SPAM

### Dataset distribution

| Class | Number of messages | Percentage |
|---|---:|---:|
| HAM | 4,827 | 86.5% |
| SPAM | 747 | 13.5% |
| **Total** | **5,574** | **100%** |

### Train, validation and test split

The dataset is divided using stratified sampling to preserve class proportions.

| Dataset | Percentage | Records |
|---|---:|---:|
| Training | 80% | 4,459 |
| Validation | 10% | 558 |
| Testing | 10% | 557 |
| **Total** | **100%** | **5,574** |

Random seed: `42`

### Exploratory Data Analysis

The following observations were obtained from the training split.

**Class distribution**

- HAM: 3,861 messages (86.59%).
- SPAM: 598 messages (13.41%).

**Message length analysis**

| Statistic | HAM | SPAM |
|---|---:|---:|
| Mean word count | 14.39 | 23.98 |
| Median word count | 11 | 25 |
| 75th percentile | 19 | 28 |
| Maximum word count | 171 | 35 |

Spam messages are longer on average than legitimate messages in this dataset.

The class distribution also reveals an imbalance, making macro F1 and spam recall important evaluation metrics.

## 5. Project Structure

```text
distilbert-sms-classifier/
│
├── models/
│   ├── checkpoints/
│   └── distilbert-sms/
│
├── results/
│   ├── confusion_matrix.png
│   ├── test_metrics.json
│   └── classification_report.txt
│
├── src/
│   ├── __init__.py
│   ├── data_exploration.py
│   ├── data_processing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── config.py
├── requirements.txt
└── README.md
```

### File responsibilities

| File | Responsibility |
|---|---|
| `data_exploration.py` | Dataset inspection, class distribution and message length analysis |
| `data_processing.py` | Dataset splitting, tokenizer loading and tokenization |
| `model.py` | Loading the pretrained DistilBERT classification model |
| `train.py` | Model fine-tuning and checkpoint selection |
| `evaluate.py` | Test evaluation, metrics and confusion matrix |
| `predict.py` | Prediction on new SMS messages |
| `config.py` | Centralized configuration |
| `requirements.txt` | Project dependencies |

## 6. Model Architecture

The project uses `distilbert-base-uncased`.

DistilBERT is a smaller, distilled version of BERT that retains much of its language understanding capability while having fewer parameters.

The classification architecture consists of:

1. **Tokenizer:** Converts SMS text into numerical token representations.
2. **DistilBERT encoder:** Processes tokens and generates contextual representations.
3. **Classification head:** Maps the encoded representation to two output classes.
4. **Output logits:** Produces scores for HAM and SPAM.

The classification head is initialized for the downstream task and learns during fine-tuning.

### Architecture flow

```text
SMS Message
     |
     v
DistilBERT Tokenizer
     |
     v
Input IDs + Attention Mask
     |
     v
DistilBERT Encoder
     |
     v
Classification Head
     |
     v
Output Logits
     |
     v
Argmax / Softmax
     |
     v
HAM or SPAM
```

## 7. Data Processing

The preprocessing pipeline performs the following operations:

1. Load the SMS spam dataset.
2. Divide the dataset into training, validation and test sets.
3. Load the pretrained DistilBERT tokenizer.
4. Tokenize SMS messages.
5. Apply truncation to handle long sequences.
6. Remove the original SMS text column after tokenization while preserving labels.

### Tokenization

The tokenizer converts text into:

- `input_ids`: Numerical token identifiers.
- `attention_mask`: Identifies actual tokens and padding positions.

### Dynamic Padding

`DataCollatorWithPadding` dynamically pads messages within each batch to the length of the longest message in that batch.

This avoids unnecessary padding across the entire dataset.

## 8. Model Training

The model is fine-tuned using Hugging Face's `Trainer` API.

### Training configuration

| Parameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Number of classes | 2 |
| Epochs | 2 |
| Learning rate | 2e-5 |
| Training batch size | 8 |
| Evaluation batch size | 8 |
| Weight decay | 0.01 |
| Random seed | 42 |
| Evaluation strategy | Every epoch |
| Checkpoint saving | Every epoch |
| Best model selection | Validation macro F1 |

### Training workflow

1. Load and tokenize the dataset.
2. Initialize the pretrained DistilBERT classification model.
3. Configure training arguments.
4. Initialize the Hugging Face Trainer.
5. Fine-tune the model on the training dataset.
6. Evaluate after every epoch using the validation dataset.
7. Select the checkpoint with the highest validation macro F1.
8. Save the best model and tokenizer.

The best checkpoint is saved in `models/distilbert-sms/`.

### Training environment

- Training device: CPU
- Approximate training time: 70 minutes

Training time depends on the hardware and environment.

## 9. Model Evaluation

The trained model is evaluated on the separate test dataset containing 557 SMS messages.

The following metrics are calculated:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Weighted Precision | Precision weighted by class frequency |
| Weighted Recall | Recall weighted by class frequency |
| Weighted F1 | F1 weighted by class frequency |
| Macro F1 | Average F1 across both classes |
| Spam Recall | Proportion of actual spam messages correctly detected |

### Evaluation Results

| Metric | Test result |
|---|---:|
| Accuracy | 99.28% |
| Weighted Precision | 99.28% |
| Weighted Recall | 99.28% |
| Weighted F1 | 99.28% |
| Macro F1 | 98.42% |
| Spam Recall | 95.95% |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| HAM | 0.99 | 1.00 | 1.00 | 483 |
| SPAM | 0.99 | 0.96 | 0.97 | 74 |
| **Accuracy** | | | **0.99** | **557** |
| Macro average | 0.99 | 0.98 | 0.98 | 557 |
| Weighted average | 0.99 | 0.99 | 0.99 | 557 |

### Results Interpretation

- The model correctly classified approximately 553 out of 557 test messages.
- It correctly detected 71 out of 74 spam messages.
- It missed 3 spam messages.
- The macro F1 score of 98.42% indicates strong performance across both classes.

### Confusion Matrix

A confusion matrix is generated to examine correct and incorrect predictions for both HAM and SPAM.

The evaluation outputs are saved in the `results/` directory:

- `confusion_matrix.png`
- `test_metrics.json`
- `classification_report.txt`

## 10. Prediction

The project provides a reusable `predict_sms()` function.

It accepts an SMS message and returns:

- Predicted class.
- Prediction confidence.

### Example usage

```python
from src.predict import predict_sms

result = predict_sms(
    "Congratulations! You have won a cash prize."
)

print(result)
```

### Example output

```python
{
    "prediction": "SPAM",
    "confidence": 0.98
}
```

The confidence shown is illustrative. Actual predictions depend on the trained model. Confidence is not necessarily a calibrated probability.

## 11. Installation and Execution

### Prerequisites

- Python 3.10 or later.
- Internet connection for downloading the dataset and pretrained model.
- CPU or compatible GPU.

### Step 1: Clone the repository

```bash
git clone https://github.com/ML-DL-Mastery

cd distilbert-sms-classifier
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Explore the dataset

```bash
python -m src.data_exploration
```

### Step 5: Train the model

```bash
python -m src.train
```

### Step 6: Evaluate the model

```bash
python -m src.evaluate
```

### Step 7: Run predictions

```bash
python -m src.predict
```

**Note:** Training downloads the pretrained model and dataset if they are not already cached. The training process may take considerably longer on a CPU.

### Hardware and Device Selection

The project automatically selects the available computation device.

- **GPU:** Uses CUDA when a compatible GPU and CUDA-enabled PyTorch installation are available.
- **CPU:** Falls back to CPU when CUDA is unavailable.

Device selection is handled automatically during training, evaluation and prediction.

## 12. Limitations

- **Class imbalance:** The dataset contains significantly more HAM messages (86.5%) than SPAM messages (13.5%), which can affect classification performance across classes.
- **Dataset generalization:** The model is trained on a specific SMS dataset and may not generalize equally well to newer spam patterns, different languages, or messages with different writing styles.
- **Confidence calibration:** The softmax confidence returned during prediction is not necessarily a calibrated probability of correctness.
- **Limited evaluation:** Performance is measured using a single held-out test split. Results may vary on different datasets and real-world SMS messages.
- **Text-only classification:** The model relies on message text and does not consider sender information, URLs, or other external signals that could help identify spam.

## 13. Future Improvements

- Experiment with additional hyperparameter configurations.
- Compare DistilBERT with other pretrained transformer models.
- Improve spam detection through additional training data.
- Add batch prediction support.
- Develop a FastAPI endpoint for model inference.


## 14. Acknowledgements

- Hugging Face Transformers
- Hugging Face Datasets
- PyTorch
- Scikit-learn

---

**Project:** SMS Spam Classification using DistilBERT  
**Task:** Binary Text Classification  
**Model:** DistilBERT  
**Framework:** PyTorch and Hugging Face Transformers