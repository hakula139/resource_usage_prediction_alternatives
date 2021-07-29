from torch import nn, Tensor
from torch.nn import init


class GruNet(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        seq_len: int,
        n_layers: int,
        dropout: float = 0.2,
    ) -> None:
        '''
        Args:
            `hidden_size`: the dimension of the hidden state
            `output_size`: the number of output data
            `seq_len`: the size of each batch data (sequence length)
            `n_layers`: the depth of recurrent layers
            `dropout`: the dropout rate of each dropout layer
        '''

        super().__init__()

        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.hidden: Tensor = None

        self.gru = nn.GRU(
            1,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        # Initialize weights
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)

        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: Tensor) -> Tensor:
        '''
        Args:
            `input`: shape(batch_size, seq_len)

        Returns:
            shape(batch_size, output_size)
        '''

        batch_size = input.shape[0]
        input = input.reshape(self.seq_len, batch_size, -1)
        output, hidden_n = self.gru(input, self.hidden)
        self.hidden = hidden_n
        output = self.fc(output)
        return output.mean(0)

    def init_hidden(self, batch_size: int) -> None:
        '''
        Initialize hidden state.

        Args:
            `batch_size`: batch size
        '''

        weight = next(self.parameters())
        self.hidden = weight.new_zeros(
            self.n_layers * 2, batch_size, self.hidden_size,
        )
