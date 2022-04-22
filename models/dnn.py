import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__ (self):
        super(DNN, self).__init__()

        #The deep neural network is initalized as mentioned in the paper
        #Use the Adam optimizer
        #Learning rate = 0.00005
        #Use batchnorm
        #Input layer dim = 484 (4 concatenated quarters), hidden layers dim = (100, 50, 33) 
        #Activation funciton = Exponential Linear Unit (ELU)
        #Batch size = 256
        #Epochs = 10

        self.LinearLayer1 = nn.Linear(484, 100)
        self.BN1 = nn.BatchNorm2d(100)
        self.LinearLayer2 = nn.Linear(100, 50)
        self.BN2 = nn.BatchNorm2d(50)
        self.LinearLayer3 = nn.Linear(50, 33)
        self.BN3 = nn.BatchNorm2d(33)
        self.LinearLayer4 = nn.Linear(33, 1)
        self.ELUActivation = nn.ELU()

    def forward(self, x):
        z1 = self.LinearLayer1(x)
        bn1 = self.BN1(z1)
        a1 = self.ELUActivation(bn1)

        z2 = self.LinearLayer2(a1)
        bn2 = self.BN2(z2)
        a2 = self.ELUActivation(bn2)

        z3 = self.LinearLayer3(a2)
        bn3 = self.BN3(z3)
        a3 = self.ELUActivation(bn3)

        z4 = self.LinearLayer3(a3)

        return z4