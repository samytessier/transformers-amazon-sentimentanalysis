from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class CNNclassifier(nn.Module):
    def __init__(self):
        #super().__init__()
        super(CNNclassifier, self).__init__() #dim of embedding vectors returned by fasttext: 300
        self.M = nn.Sequential(
        nn.Conv1d(in_channels=300, out_channels=65, kernel_size=6), #65: to not confuse with batch size
        nn.BatchNorm1d(num_features=65),
        nn.MaxPool1d(3),
        nn.ReLU(),
        nn.Conv1d(in_channels=65, out_channels=5, kernel_size=4),
        nn.BatchNorm1d(num_features=5),
        nn.MaxPool1d(3),
        nn.Conv1d(in_channels=5, out_channels=5, kernel_size=6),
        nn.MaxPool1d(2),
        nn.Flatten(),
        nn.Linear(in_features=40, out_features=1),
        nn.Sigmoid()
        )

    def forward(self, x):
        return self.M(x)
