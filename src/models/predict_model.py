from os.path import abspath, dirname
import numpy as np
from transformers import TrainingArguments, Trainer
from datasets import load_metric, load_dataset
from src.data.load_bz2_nlpdata import amzreview_dataset
from transformers_model import bert_base
# loading the trained model based on the path of the file rather than where the file is run
save_path = dirname(abspath(__file__))
trained_model = save_path + "/savedmodels/checkpoint-500"

train, test = amzreview_dataset()

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def predict_model():

    training_args = TrainingArguments("test_trainer")

    trainer = Trainer(
        model=bert_base, args=training_args, train_dataset=train, eval_dataset=test
    )
    predictions = trainer.predict(test)
    print(predictions.predictions.shape, predictions.label_ids.shape)
    compute_metrics(predictions)



predict_model()