
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from torch.optim import Adam
from torch.utils.data import DataLoader

raw_datasets = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=2)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"].shuffle(
    seed=42).select(range(1000))
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)

optimizer = Adam(model.parameters())

for epoch in range(3):
    for batch in tqdm(train_dataloader):
        outputs = model(**batch)

        optimizer.zero_grad()
        outputs.loss.backward()        
        optimizer.step() 