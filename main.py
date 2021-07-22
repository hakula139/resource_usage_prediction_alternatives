from typing import Callable, Dict, List, NamedTuple, Type
import os
from time import process_time_ns
from predictor.predictor import ArimaPredictor, BasePredictor, GruPredictor
from common.config import *
from common.utils import read_data, plot_predictions, plot_train_loss


class PredictorOptions(NamedTuple):

    predictor_class: Type[BasePredictor]
    epoch_start: int
    data_start: Callable[[int], int]


if __name__ == '__main__':

    predictor_map: Dict[str, PredictorOptions] = {
        'arima': PredictorOptions(
            ArimaPredictor,
            epoch_start=BATCH_SIZE,
            data_start=lambda end: 0,
        ),
        'gru': PredictorOptions(
            GruPredictor,
            epoch_start=BATCH_SIZE + OUTPUT_SIZE,
            data_start=lambda end: end - epoch_start,
        ),
    }

    try:
        if not os.path.isdir(FIGURE_DIR):
            os.makedirs(FIGURE_DIR)

        print('Server started.')

        dataset = read_data(INPUT_PATH)
        options = predictor_map[MODEL]
        predictor = options.predictor_class()

        # Points for plotting
        expected_x, prediction_x = range(1, len(dataset) + 1), []
        expected_y, prediction_y = dataset, []
        train_loss_x, naive_loss_x = [], []
        train_loss_y, naive_loss_y = [], []

        total_time, max_time, time_count = 0.0, 0.0, 0

        with open(OUTPUT_PATH, 'w') as output_file:
            epoch_start = options.epoch_start

            for i in range(epoch_start, len(dataset)):
                start_time = process_time_ns()

                data_start = options.data_start(i)
                data = dataset[data_start:i]

                train_input = data[:BATCH_SIZE]
                expected = data[BATCH_SIZE:]
                train_loss: float = predictor.train(train_input, expected)
                if train_loss < LOSS_THRESHOLD:
                    train_loss_x.append(i)
                    train_loss_y.append(train_loss)

                valid_input = data[OUTPUT_SIZE:]
                predictions = predictor.predict(valid_input)

                prediction = predictions[0]
                prediction: int = max(round(
                    prediction
                    if type(prediction) == List[float]
                    else prediction.item()
                ), 0)
                prediction_x.append(i + 1)
                prediction_y.append(prediction)

                end_time = process_time_ns()
                time = (end_time - start_time) / 1e6
                total_time += time
                max_time = max(max_time, time)
                time_count += 1

                print(
                    f'> {prediction} \t'
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
