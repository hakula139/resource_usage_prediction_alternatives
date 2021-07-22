from typing import Any, List
from abc import ABC, abstractmethod
from torch import float32, nn, optim, Tensor, tensor
from common.config import *
from predictor.models.arima import Arima
from predictor.models.gru import GruNet


class BasePredictor(ABC):

    @abstractmethod
    def __init__(self) -> None:

        super().__init__()

    @abstractmethod
    def train(self, batch_data: List[int], expected: List[int]) -> float:
        '''
        Args:
            `batch_data`: batch data for training
            `expected`: expected output

        Returns:
            The training loss.
        '''

        pass

    @abstractmethod
    def predict(self, batch_data: List[int]) -> Any:
        '''
        Args:
            `batch_data`: batch data for predicting

        Returns:
            The predicted values.
        '''

        pass


class ArimaPredictor(BasePredictor):

    def __init__(self) -> None:

        super().__init__()

    def train(self, batch_data: List[int], expected: List[int] = None) -> float:

        self.model = Arima(batch_data, (ARIMA_P, ARIMA_D, ARIMA_Q))
        self.model_fit = self.model.fit()
        return self.model_fit.mse

    def predict(self, batch_data: List[int] = None) -> List[float]:

        return self.model_fit.forecast(OUTPUT_SIZE)


class GruPredictor(BasePredictor):

    def __init__(self) -> None:

        super().__init__()

        self.model = GruNet(
            INPUT_SIZE,
            HIDDEN_SIZE,
            OUTPUT_SIZE,
            BATCH_SIZE,
            N_LAYERS,
            DROPOUT,
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), LEARNING_RATE)

    def train(self, batch_data: List[int], expected: List[int]) -> float:

        self.model.train()
        self.model.zero_grad()
        self.model.init_hidden(BATCH_SIZE)

        input = tensor(batch_data, dtype=float32) / MAX_SIZE
        target = tensor(expected, dtype=float32)
        output = self.model.forward(input) * MAX_SIZE

        loss = self.loss(output, target)
        cur_loss = loss.item()

        loss.backward()
        self.optimizer.step()
        return cur_loss

    def predict(self, batch_data: List[int]) -> Tensor:

        self.model.eval()
        self.model.init_hidden(BATCH_SIZE)

        input = tensor(batch_data, dtype=float32) / MAX_SIZE
        output = self.model.forward(input) * MAX_SIZE
        return self.model.relu(output)

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        '''
        Args:
            `output`: actual output
            `target`: expected output

        Returns:
            The loss (MSE) between output and target.
        '''

        return self.criterion(output, target)
