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

        epsilons = self.fc_network(lstm_out_0 + lstm_out_2) # (N, L, 1); add residual from lstm_0
        
        return epsilons  # (N, L, 1)
        

class HeightDeltaLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm_0 = nn.LSTM(input_size=6, hidden_size=64, batch_first=True)
        self.lstm_1 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)

        fc_0 = nn.Linear(in_features=64, out_features=32)
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
            fc_3
        )

    def forward(self, x_reduced): # x_reduced has shape (N, L, 5)
        N, L, _ = x_reduced.shape

        h = torch.zeros((N, 1))

        heights = []

        state_0 = None
        state_1 = None
        state_2 = None

        for t in range(L):
            x_t = torch.concatenate((x_reduced[:, t, :], h), dim=-1) # (N, 6)
            x_t = x_t.unsqueeze(dim=1) # (N, 1, 6); add sequence dimension

            x_t, state_0 = self.lstm_0(x_t, state_0)
            x_t, state_1 = self.lstm_1(x_t, state_1)
            x_t, state_2 = self.lstm_2(x_t, state_2) # x_t is now (B, 1, 64)

            delta_h_t = self.fc_network(x_t) # (N, 1, 1)
            delta_h_t = delta_h_t.squeeze(1) # (N, 1)

            heights.append(h) # Assumes the initial height is 0
            h += delta_h_t

        heights_tensor = torch.stack(heights, dim=1) # (N, L, 1)

        return heights_tensor


class HeightLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm_0 = nn.LSTM(input_size=5, hidden_size=64, bidirectional=True, batch_first=True)
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
            fc_3
        )
    
    def forward(self, x): # x has shape (N, L, 5)
        lstm_out_0, _ = self.lstm_0(x) # (N, L, 128)
        lstm_out_1, _ = self.lstm_1(lstm_out_0) # (N, L, 128)
        lstm_out_2, _ = self.lstm_2(lstm_out_1) # (N, L, 128)

        h_refined = self.fc_network(lstm_out_0 + lstm_out_2) # (N, L, 1); add residual from lstm_0
        
        return h_refined  # (N, L, 1)
    

class HeightNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm_f = HeightDeltaLSTM()
        self.lstm_b = HeightDeltaLSTM()
        self.lstm_height = HeightLSTM()

    def forward(self, p, delta_p, epsilon): # p is (N, L, 4), delta_p is (N, L, 4), epsilon is (N, L, 1)
        L = p.shape[1]
        times = torch.arange(L, dtype=p.dtype).view(1, L, 1) # zero-indexed, so perform t - 1 in weight computation isn't necessary

        lstm_f_input = torch.concatenate((delta_p, epsilon), dim=-1) # (N, L, 5); initial height feature added inside implmentation of HeightDeltaLSTM
        lstm_b_input = torch.flip(lstm_f_input, dims=1)

        h_f = self.lstm_f(lstm_f_input) # (N, L, 1)
        h_b = self.lstm_b(lstm_b_input) # Sequence is currently reversed and unable to be weighted properly to form h_t
        h_b = torch.flip(h_b, dims=1) # (N, L, 1)

        fb_weights = times / (L - 1) # (1, L, 1)
        h = (1 - fb_weights) * h_f + fb_weights * h_b # (N, L, 1)

        lstm_height_input = torch.concatenate((p, h), dim=-1)
        h_refined = self.lstm_height(lstm_height_input) # (N, L, 1)

        return h_refined  # (N, L, 1)

