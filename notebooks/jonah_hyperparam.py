import wandb
from datasets import load_dataset, load_metric
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
wandb.login()
wandb.init(project='first-run',entity='mlops-group9')

metric = load_metric('glue', 'mrpc')
dataset = load_dataset('amazon_polarity')
print("\n Dataset downloaded \n")

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
def tokenize_function(examples):
    return tokenizer(examples["content"], padding="max_length", truncation=True)


small_train_dataset = dataset["train"].shuffle(seed=42).select(range(300)) 
small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(50)) 

tokenized_small_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
tokenized_small_eval_dataset = small_eval_dataset.map(tokenize_function, batched=True)

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', return_dict=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Evaluate during training and a bit more often
# than the default to be able to prune bad trials early.
# Disabling tqdm is a matter of preference.
training_args = TrainingArguments(
    "test", evaluation_strategy="steps", eval_steps=50, disable_tqdm=True)
trainer = Trainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=tokenized_small_train_dataset,
    eval_dataset=tokenized_small_eval_dataset,
    model_init=model_init,
    compute_metrics=compute_metrics,
)

# Default objective is the sum of all metrics
# when metrics are provided, so we have to maximize it.
trainer.hyperparameter_search(
    direction="maximize", 
    backend="ray", 
    n_trials=3,
    resources_per_trial={"cpu": 8} # number of trials
)