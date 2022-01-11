import argparse
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

from data.load_bz2_nlpdata import amzreview_dataset
from models.cnn_model import CNNclassifier
#from checkpoint_mgmt import load_checkpoint, save_checkpoint


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
    


    def train(self): #train_cnn currently
        print("Training homemade CNN classifier")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        parser.add_argument('--epochs', default=24) #no type declaration in this-> is it safe?
        
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        model = CNNclassifier()
        train_set, _ = amzreview_dataset() 
        
        criterion = torch.nn.BCELoss() #BINARY cross entropy loss
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_losses = []

        #training loop:

        for epoch in range(args.epochs):
            model.train()
            loss = 0
            for images, labels in train_set: #gradients, update weights on
            # -this observation
                optimizer.zero_grad()
                log_ps = model(torch.transpose(images, 1, 2).float())
                #print("log_ps size: ",log_ps[0].size())
                loss = criterion(log_ps.view(64), labels.float()) #resize from (64, 1) to 64
                loss.backward()
                optimizer.step()

                loss += loss.item()
            train_losses.append(loss.item())
            print("Epoch: {}, Loss: {}".format(epoch, loss))

        #checkpoint should be created here
        torch.save(model, 'models/CNN_classifier.pt') #[COFIG] put in configfile
        print(
            "sucesfully trained & saved model at model/CNN_classifier.pt"
        )
        #plot loss:
        # plt.plot(np.arange(args.epochs), 
        # train_losses, label="training accuracy")
        # plt.ylabel("Loss")
        # plt.xlabel("epochs")
        # plt.legend()
        # plt.show()


    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--model_path', default='models/CNN_classifier.pt', type=str)
        parser.add_argument('--mb_size', default=64, type=int)
        args = parser.parse_args(sys.argv[2:])

        model = torch.load("models/CNN_classifier.pt")

        _, test_loader = amzreview_dataset()
        #test_loader = torch.utils.data.DataLoader(
        #    test_set, batch_size=args.mb_size, shuffle=True)

        with torch.no_grad():
            model.eval()
            correct_preds, n_samples = 0, 0
            for images, labels in test_loader:
                #print("\n made it here \n")
                ps = torch.exp(model.forward(torch.transpose(images, 1, 2)))
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)

                correct_preds += torch.sum(equals).item()
                n_samples += images.shape[0]

            accuracy = correct_preds / n_samples

        print(f'Accuracy of classifier: {accuracy*100}%')

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    
