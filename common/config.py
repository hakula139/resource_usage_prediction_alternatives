import os

MODEL = 'gru'

INPUT_SIZE = 1
HIDDEN_SIZE = 10
OUTPUT_SIZE = 1
BATCH_SIZE = 20
MAX_SIZE = 5000
ARIMA_P = 5  # autoregressive model parameter
ARIMA_D = 1  # integrated model parameter
ARIMA_Q = 0  # moving average model parameter
N_LAYERS = 2
DROPOUT = 0.0
LEARNING_RATE = 4e-3
LOSS_THRESHOLD = 100

DATA_DIR = 'data'
INPUT_PATH = os.path.join(DATA_DIR, 'input.txt')
OUTPUT_PATH = os.path.join(DATA_DIR, 'output.txt')
FIGURE_DIR = os.path.join('figures', MODEL)
PREDICTIONS_FIGURE_PATH = os.path.join(FIGURE_DIR, 'predictions.svg')
TRAIN_LOSS_FIGURE_PATH = os.path.join(FIGURE_DIR, 'train_loss.svg')
