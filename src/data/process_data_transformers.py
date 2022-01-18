import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.option('--size_train', required=True, type=int)
@click.option('--size_val', required=True, type=int)
@click.argument('input_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(size_train, size_val, input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Tokenize data set from transformers data')
    train_dataset = load_from_disk(input_filepath + '/train_dataset_size_%s' % size_train)
    eval_dataset = load_from_disk(input_filepath + '/eval_dataset_size_%s' % size_val)
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    tokenized_train_dataset.save_to_disk(output_filepath + '/train_processed_size_%s' % size_train)
    tokenized_eval_dataset.save_to_disk(output_filepath + '/eval_processed_size_%s' % size_val)

def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    tokenized = tokenizer(examples["content"], padding="max_length", truncation=True)
    return tokenized

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()