import numpy as np
from datasets import load_dataset
from datasets import IterableDataset
#from datasets.IterableDataset import shuffle
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification

from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM

print("\n Downloading Dataset \n")
dataset = load_dataset('amazon_polarity')
print("\n Dataset downloaded \n")

print('\n Token setup \n')
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
def tokenize_function(examples):
    return tokenizer(examples["content"], padding="max_length", truncation=True)
print('\n Token initialized \n')


small_train_dataset = dataset["train"].shuffle(seed=42).select(range(50)) 
small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(50)) 
big_train_dataset = dataset["train"].shuffle(seed=42).select(range(3000)) 
big_eval_dataset = dataset["test"].shuffle(seed=42).select(range(3000)) 

tokenized_small_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
tokenized_small_eval_dataset = small_eval_dataset.map(tokenize_function, batched=True)
tokenized_big_train_dataset = big_train_dataset.map(tokenize_function, batched=True)
tokenized_big_eval_dataset = big_eval_dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

training_args = TrainingArguments("test_trainer")

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_small_train_dataset, eval_dataset=tokenized_small_eval_dataset)

print('\n Starting training \n')
trainer.train()
print('\n Training Done ! \n')


from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

predictions = trainer.predict(tokenized_small_eval_dataset)
print("Prediction shape ", predictions.predictions.shape, " Prediction label shape ", predictions.label_ids.shape)

preds = np.argmax(predictions.predictions, axis=-1)
metric = load_metric("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
