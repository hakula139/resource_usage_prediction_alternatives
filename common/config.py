import os

MODEL = 'gru'

BATCH_SIZE = 1
HIDDEN_SIZE = 20
OUTPUT_SIZE = 1
SEQ_LEN = 50
MAX_SIZE = 5000
ARIMA_P = 20  # autoregressive model parameter
ARIMA_D = 1  # integrated model parameter
ARIMA_Q = 0  # moving average model parameter
ARIMA_WINDOW_SIZE = 300
N_LAYERS = 2
DROPOUT = 0.0
LEARNING_RATE = 6e-3
LOSS_THRESHOLD = 10
DISPLAY_SIZE = 5000

DATA_DIR = 'data'
INPUT_PATH = os.path.join(DATA_DIR, 'input.txt')
OUTPUT_PATH = os.path.join(DATA_DIR, 'output.txt')
FIGURE_DIR = os.path.join('figures', MODEL)
PREDICTIONS_FIGURE_PATH = os.path.join(FIGURE_DIR, 'predictions.svg')
TRAIN_LOSS_FIGURE_PATH = os.path.join(FIGURE_DIR, 'train_loss.svg')
