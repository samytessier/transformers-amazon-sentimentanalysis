from datasets import load_from_disk, load_metric
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer

# -*- coding: utf-8 -*-
from numpy.lib.type_check import imag
import numpy as np
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import wandb
wandb.init(project='huggingface',entity='TheJproject')



@click.command()
@click.option('--size_train', required=True, type=int)
@click.option('--size_val', required=True, type=int)
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
def main(data_filepath, model_filepath, size_train, size_val):
    print("Predict day and night")
    #parser = argparse.ArgumentParser(description='Training arguments')
    #parser.add_argument('--lr', default=0.1)
    # add any additional argument that you want
    #args = parser.parse_args(sys.argv[2:])
    #print(args)
    
    #path
    train_dataset = load_from_disk(data_filepath + '/train_processed_size_%s' % size_train)
    eval_dataset = load_from_disk(data_filepath + '/eval_processed_size_%s' % size_val)

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


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    # not used in this stub but often useful for finding various files


    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
