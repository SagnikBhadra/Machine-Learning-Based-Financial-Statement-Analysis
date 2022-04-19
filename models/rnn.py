import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, device):
        super(RNN, self).__init__()

        #input size = 121
        #Use Gated Recurrent Unit (GRU)
        #Initialize h and n to 0s
        #Hidden state dim = 20
        #Stacking GRU cells = 10?
        #Hidden state of the top most GRU is linked to a FCL
        #RMSProp optimizer
        #Learning rate = 0.001
        #Epochs = 5
        #Batch size = 128
        
        self.device = device
        self.input_size = 121
        self.num_layers = 4
        self.hidden_size = 20
        self.num_classes = 1
        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first = True)
        # x -> (batch_size, sequence_size, input_size)
        self.linear_layer = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        initial_hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.rnn(self, initial_hidden_state)
        # out: batch_size, sequence_length, hidden_size

        #out: (batch_size, hidden_size)
        out = out[:, -1, :]
        out = self.linear_layer(out)

        return out

