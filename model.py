import torch
import torch.nn as nn

class toy_LSTM(nn.Module):
    def __init__(self, seq_len):
        super(toy_LSTM, self).__init__()
        self.seq_len = seq_len
        self.lstm_hidden_size = 6
        self.num_layers = 1
        self.num_classes = 8
        self.input_dim = 3 #单个数据的特征维度
        self.bidirectional = False
        self.lstm_input_size = 6
        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=self.bidirectional)
        self.lstm_out_size = self.lstm_hidden_size * 2 if self.bidirectional else self.lstm_hidden_size
        self.dropout = nn.Dropout(p=0.25)
        self.fc = nn.Linear(self.lstm_out_size, self.num_classes)
        self.head = nn.Sequential(nn.Linear(self.input_dim, 4),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.25),
                                 nn.Linear(4, self.lstm_input_size),
                                 nn.ReLU())
        self.project1 = nn.Sequential(nn.Linear(self.input_dim, self.lstm_input_size),
                                                nn.ReLU(),
                                                )
        self.project2 = nn.Sequential(nn.Linear(self.lstm_out_size, self.lstm_out_size),
                                                nn.ReLU(),
                                                )
        self.MLP = self.project2 = nn.Sequential(nn.Linear(self.input_dim, 6),
                                                 nn.ReLU(),
                                                 nn.Linear(6, self.num_classes),
                                                 )

    def forward(self, feats):
        out = feats

        ########## MLP model
        # out = out.view(-1, self.input_dim)
        # out = self.MLP(out)

        ########## LSTM model
        out = self.project1(out)
        out, (_, _) = self.lstm(out, None)
        out = self.fc(out)
        return out