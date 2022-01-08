from torch import nn
import torch.nn.functional as F
import torch.optim as optim
# taken from https://nextjournal.com/gkoehler/pytorch-mnist

class CNNclassifier(nn.Module):
    def __init__(self):
        #super().__init__()
        super(CNNclassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=3, kernel_size=6)
        self.conv1_bn= nn.BatchNorm1d(num_features=64)
        #maxpool here
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=5, kernel_size=4)
        self.conv2_bn= nn.BatchNorm1d(num_features=64)
        #maxpool here
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=5, kernel_size=6)
        #globalmaxpool
        self.ft1 = nn.Flatten()
        self.fc1 = nn.Linear(in_features=100, out_features=1)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 1))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 1))
        x = F.relu(F.max_pool2d(self.conv3(x),1))
        x = F.dropout(x, training=self.training)
        x = self.ft1(F.relu(self.fc1(x)))
        return F.log_softmax(x)
