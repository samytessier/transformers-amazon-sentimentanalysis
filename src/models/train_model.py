from os.path import abspath, dirname
from pathlib import Path
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers_model import bert_base

from datasets import load_dataset
#data_path = dirname(abspath(__file__))
data_path = str(Path(abspath(__file__)).resolve().parents[1]) + "/data/raw/test.ft.txt.bz2"

#requires python setup.py install locally to import
from src.data.load_bz2_nlpdata import amzreview_dataset 
train, test = amzreview_dataset(data_path)


#dataset = load_dataset('amazon_polarity')


def train_model():
    
    
    model = bert_base()

    training_args = TrainingArguments("test_trainer")

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train, eval_dataset=test
    )
    trainer.train()


train_model()