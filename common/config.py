import os

MODEL = 'gru'

ARIMA_P = 20  # autoregressive model parameter
ARIMA_D = 1  # integrated model parameter
ARIMA_Q = 0  # moving average model parameter
ARIMA_WINDOW_SIZE = 300

HIDDEN_SIZE = 20
OUTPUT_SIZE = 1
MAX_SIZE = 1000
SEQ_LEN = 50
RNN_WINDOW_SIZE = 400
N_LAYERS = 1
DROPOUT = 0.5
LEARNING_RATE = 1e-2
LOSS_THRESHOLD = 0.1

DISPLAY_SIZE = 5000
DATA_DIR = 'data'
INPUT_PATH = os.path.join(DATA_DIR, 'input.txt')
OUTPUT_PATH = os.path.join(DATA_DIR, 'output.txt')
FIGURE_DIR = os.path.join('figures', MODEL)
PREDICTIONS_FIGURE_PATH = os.path.join(FIGURE_DIR, 'predictions.svg')
LOSS_FIGURE_PATH = os.path.join(FIGURE_DIR, 'loss.svg')
