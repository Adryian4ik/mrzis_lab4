import numpy as np
from scipy.signal import convolve2d
from math import ceil, exp



def linear(x: float) -> float:
    return x


def derivative_of_linear(x: float) -> float:
    return 1 + 0 * x


def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))


def derivative_of_sigmoid(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x: float) -> float:
    return 2 / (1 + exp(-2 * x)) - 1


def derivative_of_tanh(x: float) -> float:
    return 1 - tanh(x) ** 2


class CPO:

    input_size: tuple[int, int]
    output_size: tuple[int, int]
    poling_size: tuple[int, int]

    conv_weights: np.array

    def __init__(self, in_size: tuple[int, int], pol_size: tuple[int, int], af, daf):
        self.input_size = in_size
        self.poling_size = pol_size
        self.output_size = (ceil(in_size[0] / pol_size[0]), ceil(in_size[1] / pol_size[1]))
        self.conv_weights = np.zeros((in_size[0] * in_size[1] + 1, pol_size[0] * pol_size[1]))
        self.af = np.vectorize(af)
        self.daf = np.vectorize(daf)

    def get_output(self, image: np.ndarray) -> np.array:

        x = np.append(np.copy(image), -1)
        w_sum = self.af(np.dot(x, self.conv_weights))
        res_size = (ceil(image.shape[0] / self.poling_size[0]), ceil(image.shape[1] / self.poling_size[1]))
        result = np.reshape(w_sum, res_size)
        for y in range(res_size[0]):
            for x in range(res_size[1]):
                result[y, x] = np.max(image[y * res_size[0]: (y + 1) * res_size[0],
                                      x * res_size[1]: (x + 1) * res_size[1]])

        return result


class CNN:
    layers: list[list[CPO]]
    def __init__(self):
        self.layers = [
            [CPO((28, 28), (2, 2), linear, derivative_of_linear) for _ in range(6)],  # 14 * 14
            [CPO((14, 14), (2, 2), linear, derivative_of_linear) for _ in range(15)],  # 7 * 7
            [CPO((7, 7), (3, 3), linear, derivative_of_linear) for _ in range(20)],  # 3 * 3
        ]
        pass
