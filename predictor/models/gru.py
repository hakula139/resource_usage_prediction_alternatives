from typing import Tuple
from torch import nn, Tensor
from torch.nn import init


class GruEncoder(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        seq_len: int,
        n_layers: int = 1,
        dropout: float = 0.0,
        bidirectional=False,
    ) -> None:
        '''
        Args:
            `hidden_size`: the dimension of the hidden state
            `seq_len`: the size of each batch data (sequence length)
            `n_layers`: the depth of recurrent layers
            `dropout`: the dropout rate of each recurrent layer
        '''

        super().__init__()

        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.rnn_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            1,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Initialize weights
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        '''
        Args:
            `input`: shape(batch_size, seq_len)

        Returns:
            `output`: shape(batch_size, seq_len, hidden_size)
            `hidden`: shape(batch_size, hidden_size)
        '''

        batch_size: int = input.shape[0]
        hidden: Tensor = self.init_hidden(batch_size)

        output: Tensor
        output, hidden = self.gru(
            input.unsqueeze(2),
            hidden,
        )

        n_layers = self.n_layers * self.rnn_directions
        if n_layers > 1:
            if self.rnn_directions > 1:
                output = output.reshape(
                    batch_size, self.seq_len, self.rnn_directions, self.hidden_size,
                )
                output = output.sum(2)
            hidden = hidden.reshape(
                self.n_layers, self.rnn_directions, batch_size, self.hidden_size,
            )
            hidden = hidden[-1].sum(0)
        else:
            hidden = hidden.squeeze(0)

        return output, hidden

    def init_hidden(self, batch_size: int) -> Tensor:
        '''
        Initialize hidden state.

        Args:
            `batch_size`: batch size

        Returns:
            shape(n_layers, batch_size, hidden_size)
        '''

        weight = next(self.parameters())
        return weight.new_zeros(
            self.n_layers * self.rnn_directions,
            batch_size,
            self.hidden_size,
        )


class GruDecoder(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        seq_len: int,
        n_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        '''
        Args:
            `hidden_size`: the dimension of the hidden state
            `seq_len`: the size of each batch data (sequence length)
            `n_layers`: the depth of recurrent layers
            `dropout`: the dropout rate of each dropout layer
        '''

        super().__init__()

        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.n_layers = n_layers

        self.gru = nn.GRUCell(1, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input: Tensor, hidden: Tensor) -> Tensor:
        '''
        Args:
            `input`: shape(batch_size)
            `hidden`: shape(1, batch_size, hidden_size)

        Returns:
            `output`: shape(batch_size, 1)
            `hidden`: shape(1, batch_size, hidden_size)
        '''

        hidden = self.gru(
            input.unsqueeze(1),
            hidden,
        )
        output: Tensor = self.fc(hidden)
        hidden = self.dropout(hidden)
        return output, hidden


class GruNet(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        seq_len: int,
        n_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        '''
        Args:
            `hidden_size`: the dimension of the hidden state
            `seq_len`: the size of each batch data (sequence length)
            `n_layers`: the depth of recurrent layers
            `dropout`: the dropout rate of each dropout layer
        '''

        super().__init__()

        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.n_layers = n_layers

        self.encoder = GruEncoder(
            hidden_size,
            seq_len,
            n_layers,
            dropout,
            bidirectional=False,
        )

        self.decoder = GruDecoder(
            hidden_size,
            seq_len,
            1,
            dropout,
        )

        self.relu = nn.ReLU()

    def forward(self, input: Tensor) -> Tensor:
        '''
        Args:
            `input`: shape(batch_size, seq_len)

        Returns:
            shape(batch_size, output_size)
        '''

        _, enc_hidden = self.encoder(input)
        dec_output, _ = self.decoder(input[:, -1], enc_hidden)
        return dec_output
