import torch
from torch import nn

class EOTNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm_0 = nn.LSTM(input_size=4, hidden_size=64, bidirectional=True, batch_first=True)
        self.lstm_1 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True, batch_first=True)

        fc_0 = nn.Linear(in_features=128, out_features=32)
        fc_1 = nn.Linear(in_features=32, out_features=32)
        fc_2 = nn.Linear(in_features=32, out_features=32)
        fc_3 = nn.Linear(in_features=32, out_features=1)

        self.fc_network = nn.Sequential(
            fc_0,
            nn.LeakyReLU(negative_slope=0.01),
            fc_1,
            nn.LeakyReLU(negative_slope=0.01),
            fc_2,
            nn.LeakyReLU(negative_slope=0.01),
            fc_3,
            nn.Sigmoid()
        )

    def forward(self, x): # x is (N, L, 4)
        lstm_out_0, _ = self.lstm_0(x) # (N, L, 128)
        lstm_out_1, _ = self.lstm_1(lstm_out_0) # (N, L, 128)
        lstm_out_2, _ = self.lstm_2(lstm_out_1) # (N, L, 128)

        probs = self.fc_network(lstm_out_0 + lstm_out_2) # (N, L, 1); add residual from lstm_0
        
        return probs
        

