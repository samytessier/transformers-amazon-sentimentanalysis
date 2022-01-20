from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from hydra.utils import get_original_cwd

def process_data(C):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    og_fpath = get_original_cwd()
    cfg = C.process_data
    size_train, size_val, input_filepath, output_filepath = cfg.size_train,\
     cfg.size_val,\
     og_fpath+cfg.input_filepath,\
     og_fpath+cfg.output_filepath
    logger = logging.getLogger(__name__)
    logger.info('Tokenize data set from transformers data')

    try:
        train_dataset = load_from_disk(input_filepath + '/train_dataset_size_%s' % size_train)
        eval_dataset = load_from_disk(input_filepath + '/eval_dataset_size_%s' % size_val)
    except FileNotFoundError:
        print("file not found - has make data_download been run already? \n check logs for more info")
        logger.error("aborted because file not found. likely because `make data_download` has not yet been run or bad path and sizes provided")
    
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    tokenized_train_dataset.save_to_disk(output_filepath + '/train_processed_size_%s' % size_train)
    tokenized_eval_dataset.save_to_disk(output_filepath + '/eval_processed_size_%s' % size_val)

def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    tokenized = tokenizer(examples["content"], padding="max_length", truncation=True)
    return tokenized
