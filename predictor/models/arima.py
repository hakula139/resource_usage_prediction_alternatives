from typing import List, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults


class Arima:

    def __init__(self, data: List, params: Tuple[int, int, int]) -> None:
        '''
        Args:
            `data`: a list of history data
            `params`: (`p`, `d`, `q`)
                `p`: autoregressive model parameter
                `d`: integrated model parameter
                `q`: moving average model parameter
        '''

        self.arima = SARIMAX(
            data,
            order=params,
            initialization='approximate_diffuse',
        )

    def fit(self) -> SARIMAXResults:
        '''
        Returns:
            The training result.
        '''

        return self.arima.fit(
            disp=False,
            warn_convergence=False,
        )
