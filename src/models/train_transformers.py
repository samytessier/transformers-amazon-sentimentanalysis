from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from hydra.utils import get_original_cwd

# -*- coding: utf-8 -*-
from numpy.lib.type_check import imag
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import wandb

def train_transformer(C):
    log = logging.getLogger(__name__)
    cfg = C.train_transformer
    og_fpath = get_original_cwd()
    input_filepath, output_filepath = og_fpath+cfg.input_filepath,\
     og_fpath+cfg.output_filepath

    size_val = C.common.size_val if C.same_data_size_everywhere else cfg.size_val
    size_train = C.common.size_train if C.same_data_size_everywhere else cfg.size_train
    try:
        wandb.init(project='trial-run',entity='mlops-group9')
        use_wandb = True
    except AssertionError:
        log.warning("wandb not logged-in, proceeding withtout.\n")
        use_wandb = False

    log.info("Training transformers day and night")
    train_dataset = load_from_disk(input_filepath + '/train_processed_size_%s' % size_train)
    eval_dataset = load_from_disk(input_filepath + '/eval_processed_size_%s' % size_val)

    # TODO: Implement training loop here
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2 )

    report_to="none"
    if use_wandb:
        report_to="wandb"

    training_args = TrainingArguments("test_trainer",
        report_to=report_to,  # enable logging to W&B
        run_name="bert-test"  # name of the W&B run (optional)
        )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
        )
    trainer.train()
    trainer.save_model(output_filepath + 'trained_model')
    log.info("succesfully save mode to {}".format(output_filepath + 'trained_model'))
if __name__ == '__main__':
    pass #idk