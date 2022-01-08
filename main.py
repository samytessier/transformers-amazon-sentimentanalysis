import argparse
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt


from src.data.load_bz2_nlpdata import amzreview_dataset
from cnn_model import CNNclassifier
from checkpoint_mgmt import load_checkpoint, save_checkpoint

#model based on: https://nextjournal.com/gkoehler/pytorch-mnist

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
        #if args.command == "train":
        #    train()
        if args.command == "test":
            print("not yet implemented sorry man :/")
    


    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        parser.add_argument('--epochs', default=100) #no type declaration in this-> is it safe?
        
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        model = CNNclassifier()
        train_set, _ = amzreview_dataset() #mnist() returns already a dataloader
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_losses = []

        #training loop:
        for epoch in range(args.epochs):
            model.train()
            loss = 0
            for images, labels in train_set: #gradients, update weights on this observation
                optimizer.zero_grad()
                log_ps = model(images.float())
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                loss += loss.item()
            train_losses.append(loss.item())
            print("Epoch: {}, Loss: {}".format(epoch, loss))

        #checkpoint should be created here
        #

        #plot loss:
        # plt.plot(np.arange(args.epochs), train_losses, label="training accuracy")
        # plt.ylabel("Loss")
        # plt.xlabel("epochs")
        # plt.legend()
        # plt.show()


    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        model = torch.load(args.load_model_from)
        _, test_set = mnist()


            

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    
