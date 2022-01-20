from os.path import exists

import logging
from hydra.utils import get_original_cwd

from datasets import load_dataset


def run_download(C):
    """ Download dataset from transformers into
         data ready to be processed (saved in ../raw).
    """
    original_path = get_original_cwd()
    cfg = C.run_download
    size_train, size_val, output_filepath = cfg.size_train, cfg.size_val, cfg.output_filepath
    
    logger = logging.getLogger(__name__)
    logger.info("params ok. download starting...")
    #call data part now
    dataset = load_dataset('amazon_polarity')
    train_dataset = dataset["train"].shuffle(seed=42).select(range(size_train)) 
    eval_dataset = dataset["test"].shuffle(seed=42).select(range(size_val))

    train_dataset.save_to_disk(original_path+output_filepath + '/train_dataset_size_%s' % size_train)
    eval_dataset.save_to_disk(original_path+output_filepath + '/eval_dataset_size_%s' % size_val)
    
    logger.info("successfully saved train and test sets in {}".format(output_filepath))


#from pathlib import Path
#from dotenv import find_dotenv, load_dotenv

# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()
