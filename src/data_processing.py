from transformers import AutoTokenizer  
from datasets import load_dataset, DatasetDict 

from config import MODEL_CHECKPOINT , RANDOM_SEED


def load_and_split_dataset():
  dataset = load_dataset("sms_spam")  

  train = dataset["train"]

  first_split = train.train_test_split(test_size = 0.2 , seed = RANDOM_SEED, stratify_by_column = 'label')

  second_split = first_split["test"].train_test_split(test_size = 0.5 , seed = RANDOM_SEED, stratify_by_column = 'label' )

  data = DatasetDict({"train":first_split['train'] , "test":second_split['train'] , "validation":second_split['test']})  

  return data



def load_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CHECKPOINT
    )

    return tokenizer











def tokenize_dataset(data, tokenizer):
  
  def tokenize(batch):
    return tokenizer(batch['sms'], truncation = True) 
  
  data_encoded = data.map(tokenize, batched=True,remove_columns=["sms"])

  return data_encoded






def prepare_data():

    data = load_and_split_dataset()

    tokenizer = load_tokenizer()

    data_encoded = tokenize_dataset(
        data,
        tokenizer
    )

    return data_encoded, tokenizer

