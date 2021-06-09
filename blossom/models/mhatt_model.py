import torch
import torch.nn as nn
import torch.nn.functional as F

class MHAttKWS(nn.Module):
    def __init__(
        self,
        num_classes: int=None,
        in_channel: int=1,
        hidden_dim: int=128,
        n_head: int=4,
        dropout: float=0.1
    ):
        super(MHAttKWS, self).__init__()

        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        if self.num_classes == 2:
            output_dim = 1
        else:
            output_dim = self.num_classes
        

        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(in_channel, 10, (5, 1), stride=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 1, (5, 1), stride=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2)
        )

        self.rnn = nn.LSTM(1, self.hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.q_emb = nn.Linear(self.hidden_dim << 1, (self.hidden_dim << 1) * self.n_head)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)

        x = self.cnn_extractor(x)
        x = x.reshape(x.size(0), -1, x.size(1))
        x, _ = self.rnn(x)

        middle = x.size(1) // 2
        mid_feature = x[:, middle, :]

        multiheads = []
        queries = self.q_emb(mid_feature).view(self.n_head, batch_size, -1, self.hidden_dim << 1)

        for query in queries:
            att_weights = torch.bmm(query, x.transpose(1, 2))
            att_weights = F.softmax(att_weights, dim=-1)
            multiheads.append(torch.bmm(att_weights, x).view(batch_size, -1))

        x = torch.cat(multiheads, dim=-1)
        x = self.dropout(x)
        x = self.fc(x)

        return x