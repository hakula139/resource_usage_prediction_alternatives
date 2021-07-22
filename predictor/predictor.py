from typing import List, Tuple
from predictor.models.arima import Arima


class ArimaPredictor:

    def __init__(self) -> None:
        pass

    def train(self, data: List, params: Tuple[int, int, int]) -> float:
        '''
        Args:
            `data`: a list of history data
            `params`: (`p`, `d`, `q`)

        Returns:
            The training loss (MSE).
        '''

        self.model = Arima(data, params)
        self.model_fit = self.model.fit()
        return self.model_fit.mse

    def predict(self, output_size: int = 1) -> List[float]:
        '''
        Args:
            `output_size`: the number of steps to predict

        Returns:
            The predicted values.
        '''

        return self.model_fit.forecast(output_size)
