from typing import List, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults


class Arima:

    def __init__(self, params: Tuple[int, int, int]) -> None:
        '''
        Args:
            `params`: (`p`, `d`, `q`)
                `p`: autoregressive model parameter
                `d`: integrated model parameter
                `q`: moving average model parameter
        '''

        self.params = params
        self.result: SARIMAXResults = None

    def fit(self, batch_data: List[int]) -> SARIMAXResults:
        '''
        Args:
            `data`: a list of history data

        Returns:
            The training result.
        '''

        self.arima = SARIMAX(
            batch_data,
            order=self.params,
            initialization='approximate_diffuse',
        )
        self.result = self.arima.fit(
            disp=False,
            warn_convergence=False,
        )
        return self.result

    def append(
        self, batch_data: List[int], refit: bool = False
    ) -> SARIMAXResults:
        '''
        Append new data to current model.

        Args:
            `batch_data`: a list of new observed data
            `refit`: perform refit or not

        Returns:
            The new training result.
        '''

        self.result = self.result.extend(batch_data)
        return self.result
