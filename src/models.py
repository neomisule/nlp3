import torch
import torch.nn as nn
import torch.nn.functional as F

ACT_MAP = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid()
}

class BaseRNNClassifier(nn.Module):
    def __init__(self, arch="bilstm", vocab_size=10000, embed_dim=100, hidden_size=64,
                 num_layers=2, dropout=0.3, activation="relu", bidirectional=False):
        super().__init__()
        self.arch = arch.lower()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional

        if self.arch == "rnn":
            self.rnn = nn.RNN(embed_dim, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True, nonlinearity="tanh")
        elif self.arch == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        elif self.arch == "bilstm":
            self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
            self.bidirectional = True
        else:
            raise ValueError("Unknown arch: " + arch)

        fc_in = hidden_size * (2 if self.bidirectional else 1)
        self.fc = nn.Linear(fc_in, 1)
        self.activation_name = activation
        self.activation = ACT_MAP.get(activation, nn.ReLU())

    def forward(self, x):
        emb = self.embedding(x)  #[batch, seq, embed_dim]
        out, hn = self.rnn(emb)  #out: [batch, seq, hidden*directions]
        #For LSTM, out already contains hidden states for every time so we'll take out[:, -1, :]
        final = out[:, -1, :]  #[batch, hidden*directions]
        final = self.dropout(final)
        final = self.activation(final)
        logits = self.fc(final).squeeze(1)
        probs = torch.sigmoid(logits)
        return probs, logits
