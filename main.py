
from CNN import *


def main():

    matrix_1_to_16 = np.ones((1, 16), dtype=bool)

    matrix_16_to_32 = np.ones((16, 32), dtype=bool)

    layers: list = [
        MapConvert((1, 28, 28), 16, matrix_1_to_16),
        Convolve((16, 28, 28), (16, 5, 5), relu, derivative_of_relu),
        Pooling((16, 24, 24), (2, 2)),

        MapConvert((16, 12, 12), 32, matrix_16_to_32),
        Convolve((32, 12, 12), (32, 5, 5), relu, derivative_of_relu),
        Pooling((32, 8, 8), (2, 2)),

        MLPLayer((32, 4, 4), 10, sigmoid, derivative_of_sigmoid)
    ]

    cnn: CNN = CNN(layers)
    cnn.load_trainset('datasets/test.csv')
    cnn.load('weights/state_model.wbc')
    cnn.fit(2.5e-3)

    cnn.save('weights/state_model.wbc')

    cnn.test('datasets/test.csv')


if __name__ == '__main__':
    main()
