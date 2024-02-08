import torch
import torch.nn as nn
import torch.nn.functional as F

from dbn import DBN

class DNN(nn.Module):
    def __init__(self, layer_sizes, n_classes):
        """
        Deep Neural Network (DNN) constructor.
        
        Args:
        - layer_sizes (list): List of integers representing the sizes of layers.
        - n_classes (int): Number of classes for classification.
        """
        super(DNN, self).__init__()
        self.layer_sizes = layer_sizes
        self.n_classes = n_classes
        self.dbn = DBN(layer_sizes)
        self.classification = nn.Linear(layer_sizes[-1], n_classes)
        self.device = torch.device('cpu')
        
    def forward(self, input):
        """
        Forward pass of the DNN.
        
        Args:
        - input (Tensor): Input tensor.
        
        Returns:
        - Tensor: Output tensor after passing through the DNN.
        """
        out = self.dbn(input)
        out = self.classification(out)
        out = F.softmax(out, dim=1)
        return out
