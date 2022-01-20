from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer

# -*- coding: utf-8 -*-
from numpy.lib.type_check import imag
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import wandb

def train_transformer(C):
    cfg = C.train_transformer
    og_fpath = get_original_cwd()
    input_filepath, output_filepath, size_train, size_val = og_fpath+cfg.input_filepath,\
     og_fpath+cfg.output_filepath,\
     cfg.size_train,\
     cfg.size_val
    wandb.init(project='huggingface',entity='TheJproject')
    print("Training transformers day and night")
    train_dataset = load_from_disk(input_filepath + '/train_processed_size_%s' % size_train)
    eval_dataset = load_from_disk(input_filepath + '/eval_processed_size_%s' % size_val)

    # TODO: Implement training loop here
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2 )

    training_args = TrainingArguments("test_trainer",
        report_to="wandb",  # enable logging to W&B
        run_name="bert-test"  # name of the W&B run (optional)
        )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
        )
    trainer.train()
    trainer.save_model(output_filepath + 'trained_model')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    # not used in this stub but often useful for finding various files


    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
