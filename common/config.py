import os

MODEL = 'arima'

INPUT_SIZE = 1
OUTPUT_SIZE = 1
BATCH_SIZE = 50
ARIMA_P = 5  # autoregressive model parameter
ARIMA_D = 1  # integrated model parameter
ARIMA_Q = 0  # moving average model parameter
LOSS_THRESHOLD = 2

DATA_DIR = 'data'
INPUT_PATH = os.path.join(DATA_DIR, 'input.txt')
OUTPUT_PATH = os.path.join(DATA_DIR, 'output.txt')
FIGURE_DIR = os.path.join('figures', MODEL)
PREDICTIONS_FIGURE_PATH = os.path.join(FIGURE_DIR, 'predictions.svg')
TRAIN_LOSS_FIGURE_PATH = os.path.join(FIGURE_DIR, 'train_loss.svg')
