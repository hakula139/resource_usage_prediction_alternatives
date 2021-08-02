from typing import Dict, List, NamedTuple, Type
import os
from time import process_time_ns
from predictor.predictor import ArimaPredictor, BasePredictor, GruPredictor
from common.config import *
from common.utils import *


class PredictorOptions(NamedTuple):

    predictor_class: Type[BasePredictor]
    window_size: int
    seq_len: int


if __name__ == '__main__':

    predictor_map: Dict[str, PredictorOptions] = {
        'arima': PredictorOptions(
            ArimaPredictor,
            window_size=ARIMA_WINDOW_SIZE,
            seq_len=ARIMA_WINDOW_SIZE,
        ),
        'gru': PredictorOptions(
            GruPredictor,
            window_size=RNN_WINDOW_SIZE,
            seq_len=SEQ_LEN,
        ),
    }

    try:
        if not os.path.isdir(FIGURE_DIR):
            os.makedirs(FIGURE_DIR)

        print('Server started.')

        dataset = read_data(INPUT_PATH)
        normalized_dataset, max_value = normalize(dataset)
        options = predictor_map[MODEL]
        predictor = options.predictor_class()

        # Points for plotting
        expected_x, prediction_x = range(1, len(dataset) + 1), []
        expected_y, prediction_y = dataset, []
        train_loss_x, naive_loss_x = [], []
        train_loss_y, naive_loss_y = [], []

        total_time, max_time, time_count = 0.0, 0.0, 0
        total_loss, loss_count, avg_loss = 0.0, 0, -1.0

        start_plotting = False

        with open(OUTPUT_PATH, 'w') as output_file:
            for i in range(options.window_size, len(dataset)):
                start_time = process_time_ns()

                data = normalized_dataset[i - options.window_size:i]

                train_input = data
                train_loss: float = predictor.train(train_input)

                # Waiting for an acceptable training loss
                if train_loss < 0:
                    # Not training
                    pass
                elif train_loss < LOSS_THRESHOLD:
                    start_plotting = True

                if start_plotting and train_loss >= 0:
                    train_loss_x.append(i)
                    train_loss_y.append(train_loss)
                    total_loss += train_loss
                    loss_count += 1

                if hasattr(predictor, 'scheduler') and train_loss >= 0:
                    predictor.scheduler.step(train_loss)

                valid_input = data[-options.seq_len:]
                predictions = predictor.predict(valid_input)

                prediction: float = (
                    predictions[0]
                    if type(predictions) == List
                    else predictions[0].item()
                )
                prediction = denormalize(prediction, max_value)

                if start_plotting:
                    prediction_x.append(i + 1)
                    prediction_y.append(prediction)

                end_time = process_time_ns()
                time = (end_time - start_time) / 1e6
                total_time += time
                max_time = max(max_time, time)
                time_count += 1

                avg_loss = total_loss / loss_count if loss_count > 0 else -1.0
                print(
                    f'#{i + 1:<6} > {prediction}'
                    f' \tLoss:' + (
                        f' {train_loss:9.5f} (current)'
                        if train_loss >= 0 else ' ' * 20
                    ) + (
                        f' | {avg_loss:9.5f} (average)'
                        if avg_loss >= 0 else ' ' * 22
                    ) + f' \tTime: {time:.4f} ms'
                )
                output_file.write(f'{prediction} ')

        avg_time = total_time / time_count
        print(
            f'Loss: {avg_loss:.5f} (average)\n'
            f'Time: {avg_time:.4f} ms (average) | {max_time:.4f} ms (max)'
        )

        plot_predictions(
            PREDICTIONS_FIGURE_PATH,
            expected_x[-DISPLAY_SIZE:],
            expected_y[-DISPLAY_SIZE:],
            prediction_x[-DISPLAY_SIZE:],
            prediction_y[-DISPLAY_SIZE:],
        )
        plot_train_loss(
            TRAIN_LOSS_FIGURE_PATH,
            train_loss_x,
            train_loss_y,
        )

    except KeyboardInterrupt:
        print('\nAborted.')
