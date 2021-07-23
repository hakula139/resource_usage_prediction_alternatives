from torch import nn, Tensor


class GruNet(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        batch_size: int,
        n_layers: int,
        dropout: float = 0.2,
    ) -> None:
        '''
        Args:
            `input_size`: the dimension of input data
            `hidden_size`: the dimension of the hidden state
            `output_size`: the dimension of output data
            `batch_size`: the size of each batch data
            `n_layers`: the depth of recurrent layers
            `dropout`: the dropout rate for each dropout layer
        '''

        super().__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.hidden: Tensor = None

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: Tensor) -> Tensor:
        '''
        Args:
            `input`: shape(batch_size)

        Returns:
            shape(batch_size)
        '''

        input = input.reshape(self.batch_size, 1, -1)
        output, hidden_n = self.gru(input, self.hidden)
        self.hidden = hidden_n
        output = output.reshape(self.batch_size, -1)
        output = self.fc(output)
        return output[0]

    def init_hidden(self, batch_size: int) -> None:
        '''
        Initialize hidden state.

        Args:
            `batch_size`: the size of each batch data
        '''

        weight = next(self.parameters())
        self.hidden = weight.new_zeros(
            self.n_layers, batch_size, self.hidden_size,
        )
