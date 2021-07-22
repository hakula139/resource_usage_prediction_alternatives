from typing import List
import os
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from utils import read_data, plot_predictions, plot_train_loss


if __name__ == '__main__':
    # Parameters

    INPUT_SIZE = 1
    OUTPUT_SIZE = 1
    BATCH_SIZE = 50
    ARIMA_P = 5  # autoregressive model parameter
    ARIMA_D = 1  # integrated model parameter
    ARIMA_Q = 1  # moving average model parameter
    LOSS_THRESHOLD = 20

    DATA_DIR = 'data'
    INPUT_PATH = os.path.realpath(os.path.join(DATA_DIR, 'input.txt'))
    OUTPUT_PATH = os.path.realpath(os.path.join(DATA_DIR, 'output.txt'))
    FIGURE_DIR = 'figures'
    PREDICTIONS_FIGURE_PATH = os.path.realpath(os.path.join(
        FIGURE_DIR, 'predictions.svg'
    ))
    TRAIN_LOSS_FIGURE_PATH = os.path.realpath(os.path.join(
        FIGURE_DIR, 'train_loss.svg'
    ))

    try:
        if not os.path.isdir(FIGURE_DIR):
            os.makedirs(FIGURE_DIR)

        print('Server started.')

        dataset = read_data(INPUT_PATH)
        predictions: List[int] = []

        # Points for plotting
        expected_x, prediction_x = range(1, len(dataset) + 1), []
        expected_y, prediction_y = dataset, []
        train_loss_x, naive_loss_x = [], []
        train_loss_y, naive_loss_y = [], []

        with open(OUTPUT_PATH, 'w') as output_file:
            for i in range(BATCH_SIZE, len(dataset)):
                epoch = i + 1

                data = dataset[i - BATCH_SIZE:i]
                model = ARIMA(data, order=(ARIMA_P, ARIMA_D, ARIMA_Q))
                model_fit: ARIMAResults = model.fit(
                    method_kwargs={"warn_convergence": False}
                )
                train_loss: float = model_fit.mse / ARIMA_P
                train_loss_x.append(epoch)
                train_loss_y.append(train_loss)

                predictions: List[float] = model_fit.forecast()

                naive_pred = round(data[-OUTPUT_SIZE])
                prediction = (
                    max(round(predictions[0]), 0)
                    if train_loss < LOSS_THRESHOLD else naive_pred
                )
                prediction_x.append(epoch + 1)
                prediction_y.append(prediction)

                print('> {} (naive: {})   \tLoss: {:10.5f} (train)'.format(
                    prediction, naive_pred, train_loss
                ))
                output_file.write(f'{prediction} ')

        plot_predictions(
            PREDICTIONS_FIGURE_PATH,
            expected_x, expected_y,
            prediction_x, prediction_y,
        )
        plot_train_loss(
            TRAIN_LOSS_FIGURE_PATH,
            train_loss_x, train_loss_y,
        )

    except KeyboardInterrupt:
        print('\nAborted.')
