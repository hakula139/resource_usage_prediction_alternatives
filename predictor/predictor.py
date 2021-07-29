from typing import Any, List
from collections import deque
from abc import ABC, abstractmethod
import math
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

        self.model = Arima((ARIMA_P, ARIMA_D, ARIMA_Q))
        self.model_fit = None

    def train(self, batch_data: List[int], expected: List[int] = None) -> float:

        if self.model_fit is None:
            self.model_fit = self.model.fit(batch_data)
        else:
            self.model_fit = self.model.append(batch_data[-1:])

        return math.sqrt(self.model_fit.mse)

    def predict(self, batch_data: List[int] = None) -> List[float]:

        return self.model_fit.forecast(OUTPUT_SIZE)


class GruPredictor(BasePredictor):

    def __init__(self) -> None:

        super().__init__()

        self.model = GruNet(
            HIDDEN_SIZE,
            OUTPUT_SIZE,
            SEQ_LEN,
            N_LAYERS,
            DROPOUT,
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.8,
            patience=500,
            min_lr=2e-3,
            verbose=True,
        )

        self.cached_batch_data = deque([])
        self.cached_expected = deque([])

    def train(self, batch_data: List[int], expected: List[int]) -> float:

        self.cached_batch_data.append(batch_data)
        self.cached_expected.append(expected)
        if len(self.cached_batch_data) > BATCH_SIZE:
            self.cached_batch_data.popleft()
            self.cached_expected.popleft()

        self.model.train()
        self.model.zero_grad()
        self.model.init_hidden(SEQ_LEN)

        input = tensor(self.cached_batch_data, dtype=float32) / MAX_SIZE
        target = tensor(self.cached_expected, dtype=float32)
        output = self.model.forward(input) * MAX_SIZE

        loss = self.loss(output, target)
        cur_loss = loss.item()

        loss.backward()
        self.optimizer.step()
        return cur_loss

    def predict(self, batch_data: List[int]) -> Tensor:

        self.model.eval()
        self.model.init_hidden(SEQ_LEN)

        input = tensor([batch_data], dtype=float32) / MAX_SIZE
        output: Tensor = self.model.forward(input) * MAX_SIZE
        output = self.model.relu(output)
        return output.squeeze(0)

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        '''
        Args:
            `output`: actual output
            `target`: expected output

        Returns:
            The loss (MSE) between output and target.
        '''

        mse: Tensor = self.criterion(output, target)
        return mse.sqrt()
