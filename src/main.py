import argparse
import sys
import os

import numpy as np
import torch
#import matplotlib.pyplot as plt

from data.load_bz2_nlpdata import amzreview_dataset
from models.cnn_model import CNNclassifier
#from checkpoint_mgmt import load_checkpoint, save_checkpoint

import hydra
from hydra.utils import get_original_cwd
import logging
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

def train(C):
    cfg =  C.train
    print("workdir in train: ", os.getcwd(), "\n")
    log.info("Training homemade CNN classifier...")
    
    model = CNNclassifier()
    train_set, _ = amzreview_dataset(get_original_cwd()+'/data/raw/test.ft.txt.bz2') 
    
    try:
        criterion = getattr(torch.nn, cfg.criterion)()
    except AttributeError:
        criterion= torch.nn.BCELoss() #BINARY cross entropy loss
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparameters.lr)
    train_losses = []

    #training loop:

    for epoch in range(cfg.hyperparameters.epochs):
        loss = 0
        for images, labels in train_set: #gradients, update weights on
        # -this observation
            optimizer.zero_grad()
            log_ps = model(torch.transpose(images, 1, 2).float())
            loss = criterion(log_ps.view(64), labels.float()) #resize from (64, 1) to 64
            loss.backward()
            optimizer.step()

            loss += loss.item()
        train_losses.append(loss.item())
        log.info("Epoch: {}, Loss: {}".format(epoch, loss))
        #print("Epoch: {}, Loss: {}".format(epoch, loss))
    print("final loss: {}".format(loss))    
    torch.save(model, get_original_cwd() + '/src/models/CNN_classifier.pt') #[COFIG] put in configfile
    print("saved model at model/CNN_classifier.pt")

  
def evaluate(C):
    cfg = C.evaluate
    log.info("Evaluating until hitting the ceiling")

    model = torch.load(get_original_cwd()+"/models/CNN_classifier.pt")
    _, test_loader = amzreview_dataset(get_original_cwd()+'/data/raw/test.ft.txt.bz2')
    with torch.no_grad():
        model.eval()
        correct_preds, n_samples = 0, 0
        for images, labels in test_loader:
            ps = torch.exp(model.forward(torch.transpose(images, 1, 2)))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            correct_preds += torch.sum(equals).item()
            n_samples += images.shape[0]

        accuracy = correct_preds / n_samples

    log.info('Accuracy of classifier: {}%'.format(accuracy*100))

@hydra.main(config_path="config", config_name='config_CNN.yaml')
def main(cfg):
    """ Helper class that will to launch train and test functions
    expects there to be a "command" field in the config file
    """
    try: 
        globals()[cfg.command]
    except AttributeError:
        print('Unrecognized command \'{}\''.format(cfg.command))
        exit(1)
    globals()[cfg.command](cfg)
    #log = logging.getLogger(__name__)
    #print("log, config: ", self.log, " ", self.config)
    #only needed here, as we can't execute a train then a test in a single run

if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    
