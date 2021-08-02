from typing import Any, List
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
    def train(self, batch_data: List[int]) -> float:
        '''
        Args:
            `batch_data`: batch data for training

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

    def train(self, batch_data: List[int]) -> float:

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
            patience=200,
            min_lr=5e-3,
            verbose=True,
        )

    def train(self, batch_data: List[int]) -> float:

        self.model.train()
        self.model.zero_grad()

        batch_size = len(batch_data) - SEQ_LEN - OUTPUT_SIZE
        train_data = tensor([
            batch_data[i:i + SEQ_LEN]
            for i in range(0, batch_size)
        ], dtype=float32)
        expected = tensor([
            batch_data[i + SEQ_LEN:i + SEQ_LEN + OUTPUT_SIZE]
            for i in range(0, batch_size)
        ], dtype=float32)

        output: Tensor = self.model.forward(train_data)
        loss = self.loss(output, expected)
        cur_loss = loss.item()

        loss.backward()
        self.optimizer.step()
        return cur_loss

    def predict(self, batch_data: List[int]) -> Tensor:

        self.model.eval()

        predict_data = tensor([
            batch_data
        ], dtype=float32)

        output: Tensor = self.model.forward(predict_data)
        output = self.model.relu(output)
        return output

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
