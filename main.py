from typing import List
import os
from time import process_time_ns
from predictor.predictor import ArimaPredictor
from common.config import *
from common.utils import read_data, plot_predictions, plot_train_loss

if __name__ == '__main__':

    try:
        if not os.path.isdir(FIGURE_DIR):
            os.makedirs(FIGURE_DIR)

        print('Server started.')

        dataset = read_data(INPUT_PATH)
        predictor = ArimaPredictor()
        predictions: List[int] = []

        # Points for plotting
        expected_x, prediction_x = range(1, len(dataset) + 1), []
        expected_y, prediction_y = dataset, []
        train_loss_x, naive_loss_x = [], []
        train_loss_y, naive_loss_y = [], []

        total_time, max_time, time_count = 0.0, 0.0, 0

        with open(OUTPUT_PATH, 'w') as output_file:
            for epoch in range(BATCH_SIZE, len(dataset)):
                start_time = process_time_ns()

                data = dataset[:epoch]
                train_loss: float = predictor.train(
                    data, (ARIMA_P, ARIMA_D, ARIMA_Q)
                )
                train_loss_x.append(epoch)
                train_loss_y.append(train_loss)

                predictions: List[float] = predictor.predict()

                naive_pred = round(data[-OUTPUT_SIZE])
                prediction = (
                    max(round(predictions[0]), 0)
                    if train_loss < LOSS_THRESHOLD else naive_pred
                )
                prediction_x.append(epoch + 1)
                prediction_y.append(prediction)

                end_time = process_time_ns()
                time = (end_time - start_time) / 1e6
                total_time += time
                max_time = max(max_time, time)
                time_count += 1

                print(
                    f'> {prediction} (naive: {naive_pred})   \t'
                    f'Loss: {train_loss:9.5f} (train) \t'
                    f'Time: {time:.4f} ms'
                )
                output_file.write(f'{prediction} ')

        avg_time = total_time / time_count
        print(f'Time: {avg_time:.4f} ms (average) | {max_time:.4f} ms (max)')

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
