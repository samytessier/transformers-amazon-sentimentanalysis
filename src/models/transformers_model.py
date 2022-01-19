from transformers import AutoModelForSequenceClassification

bert_base = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
