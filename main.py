import numpy as np

from CNN import *


def main():
    kernel = np.zeros((2, 3, 3))
    kernel[0] = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 0, 0], ])
    kernel[1] = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [1, 0, 0], ])

    # c = Convolve((2, 4, 4), (3, 3), linear, derivative_of_linear, kernel)
    c = Convolve((2, 4, 4), (3, 3), linear, derivative_of_linear, kernel)

    image = np.zeros((2, 4, 4))
    image[0] = [[0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0]]
    image[1] = [[0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0]]
    output = np.zeros((2, 4, 4))
    output[0] = [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [2, 3, 3, 2],
                 [0, 0, 0, 0]]

    a = CNN()
    # a.load('./test.wbc')
    a.load_trainset('./test.csv')
    a.fit()
    a.save('./test.wbc')

    # a.test('./mnist_test.csv')



if __name__ == '__main__':
    main()
