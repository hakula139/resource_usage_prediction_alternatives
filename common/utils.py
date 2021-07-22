from typing import List
from matplotlib import pyplot as plt


def read_data(input_path: str) -> List[int]:
    '''
    Read data from local file.

    Args:
        `input_path`: the path to input file

    Returns:
        A list of input data.
    '''

    with open(input_path, 'r') as input_file:
        lines = input_file.readlines()
        data = [
            int(value)
            for line in lines
            for value in line.split()
        ]
        if len(data) > 0 and data[-1] < 0:
            data.pop()
        return data


def plot_predictions(
    output_path: str,
    x1: List[int], y1: List[int],
    x2: List[int], y2: List[int],
) -> None:
    '''
    Plot predicted values with respective expected values.

    Args:
        `output_path`: the path to output file
        `x1`: x-axis of expected values
        `y1`: y-axis of expected values
        `x2`: x-axis of predicted values
        `y2`: y-axis of predicted values
    '''

    plt.figure(figsize=(len(x1) / 25, 9))
    plt.title('Prediction figure')
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.plot(x1, y1, 'r', label='Expected')
    plt.plot(x2, y2, 'b', label='Predictions')
    plt.legend()
    plt.savefig(output_path)


def plot_train_loss(
    output_path: str,
    x1: List[int], y1: List[float],
) -> None:
    '''
    Plot training losses.

    Args:
        `output_path`: the path to output file
        `x1`: x-axis of training losses
        `y1`: y-axis of training losses
    '''

    plt.figure(figsize=(len(x1) / 25, 9))
    plt.title('Loss figure')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(x1, y1, 'r', label='Training loss')
    plt.legend()
    plt.savefig(output_path)
