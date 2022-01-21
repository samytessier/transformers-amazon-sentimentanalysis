from datasets import load_from_disk, load_metric
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
from test import _PATH_DATA
import pytest

valset = load_from_disk(_PATH_DATA + '/processed/eval_processed_size_1700')
valset.features.pop('content', None)

model = AutoModelForSequenceClassification.from_pretrained(_PATH_DATA + '/checkpoint-6000',local_files_only=True)
training_args = TrainingArguments("test_trainer",
    evaluation_strategy="epoch",
    logging_steps=100,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model, 
    args=training_args,
    train_dataset=valset, ## dummy value
    eval_dataset=valset,
    tokenizer=tokenizer,
)

def test_fails_if_missing_arg():
    """
    make sure the model fails if it doesn't have the right type of data
    this is kind of a poo-poo test but I can't think of anything better
    """
    with pytest.raises(KeyError):
        predictions = trainer.predict(valset)



