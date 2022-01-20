# -*- coding: utf-8 -*-
from numpy.lib.type_check import imag
import numpy as np
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import wandb
from hydra.utils import get_original_cwd

from datasets import load_from_disk, load_metric
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer


def eval_transformer(C):
    print("Predict day and night")
    og_fpath = get_original_cwd()
    cfg = C.eval_transformer
    data_filepath, model_filepath = og_fpath+cfg.data_filepath,\
     og_fpath+cfg.model_filepath
     
    size_val = C.common.size_val if C.same_data_size_everywhere else cfg.size_val
    size_train = C.common.size_train if C.same_data_size_everywhere else cfg.size_train

    train_dataset = load_from_disk(data_filepath + '/train_processed_size_%s' % size_train)
    eval_dataset = load_from_disk(data_filepath + '/eval_processed_size_%s' % size_val)

    print("config ok, setting up W&B...")
    wandb.init(project='huggingface',entity='TheJproject')

    model = AutoModelForSequenceClassification.from_pretrained(model_filepath + '/checkpoint-6000',local_files_only=True)
    training_args = TrainingArguments("test_trainer",
        report_to="wandb",  # enable logging to W&B
        run_name="bert-test"  # name of the W&B run (optional)
        )
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    predictions = trainer.predict(eval_dataset)
    print(predictions.predictions.shape, predictions.label_ids.shape)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = load_metric("glue", "mrpc")
    metric.compute(predictions=preds, references=predictions.label_ids)
    trainer.evaluate()


def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

