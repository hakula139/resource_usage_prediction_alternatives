from typing import List, Tuple
import math
from common.config import *
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


def normalize(data: List[int], max_value: int = None) -> Tuple[List[float], int]:
    '''
    Normalize a dataset to range [0, 1].

    Args:
        `data`: the dataset to normalize
        `max_value` (optional): a provided maximum value for normalization

    Returns:
        `normalized_data`: a list of normalized data
        `max_value`: the maximum value in dataset (for denormalization)
    '''

    if max_value is None:
        max_value = max(data) if len(data) > 0 else MAX_SIZE
    normalized_data = [
        value / max_value
        for value in data
    ]
    return normalized_data, max_value


def denormalize(value: float, max_value: int = MAX_SIZE) -> int:
    '''
    Convert a normalized value to its original value.

    Args:
        `value`: the value to denormalize
        `max_value`: the maximum value in dataset

    Returns:
        The denormalized value.
    '''

    return max(round(value * max_value), 0)


def rmse(x: List[float], y: List[float]) -> float:
    '''
    Measure the root mean square error (RMSE) between each element in the
    two vectors.

    Args:
        `x`: a list of floats
        `y`: another list of floats, with the same size as `x`

    Returns:
        The RMSE between the two vectors.
    '''

    return (
        math.sqrt(sum([
            (x[i] - y[i]) ** 2 for i in range(len(x))
        ]) / len(x))
        if len(x) == len(y)
        else -1.0
    )


def figure_size(data_size: int) -> Tuple[int, int]:
    '''
    Get figure size for pyplot.

    Args:
        `data_size`: the number of points to plot

    Returns:
        The width and height of the figure.
    '''

    return max(min(data_size / 20, 600), 20), 9


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

    plt.figure(figsize=figure_size(len(x1)))
    plt.title('Prediction figure')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.plot(x1, y1, 'r', label='Expected')
    plt.plot(x2, y2, 'b', label='Predictions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)


def plot_loss(
    output_path: str,
    x1: List[int], y1: List[float],
    x2: List[int], y2: List[float],
) -> None:
    '''
    Plot training losses and validation losses.

    Args:
        `output_path`: the path to output file
        `x1`: x-axis of training losses
        `y1`: y-axis of training losses
        `x2`: x-axis of validation losses
        `y2`: y-axis of validation losses
    '''

    plt.figure(figsize=figure_size(len(x1)))
    plt.title('Loss figure')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(x1, y1, 'r', label='Training loss (average)')
    plt.plot(x2, y2, 'b', label='Validation loss (average)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
