import random
import numpy as np
from math import ceil, exp
import pickle
from matrix import *



np.random.seed(1)



def linear(x: float) -> float:
    return x


def derivative_of_linear(x: float) -> float:
    return x * 0 + 1

def tanh(x: float) -> float:
    return 2 / (1 + exp(-2 * x)) - 1


def derivative_of_tanh(x: float) -> float:
    return 1 - tanh(x) ** 2

def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))


def derivative_of_sigmoid(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))


class Operation:
    size: tuple[int, int, int]

    def get_output(self, images: np.array) -> np.array:
        pass

    def get_error(self, in_images: np.array, errors: np.array) -> np.array:
        pass

    def learn(self, in_image: np.array, errors: np.array):
        pass

class Convolve(Operation):

    kernel: np.ndarray
    lr: float = 1

    def __init__(self, in_size: tuple[int, int, int], kernel_size: tuple[int, int], af, daf, kernel: np.array = None):
        kernel_size = (in_size[0], kernel_size[0], kernel_size[1])
        self.size = in_size
        self.kernel_size = kernel_size

        self.af = np.vectorize(af)
        self.daf = np.vectorize(daf)

        self.shift = ((kernel_size[1] - 1) // 2, (kernel_size[2] - 1) // 2)

        if kernel is not None:
            self.kernel = kernel
            self.kernel_size = kernel.shape
        else:
            self.kernel = np.random.rand(kernel_size[0], kernel_size[1], kernel_size[2])

        self.weights = np.zeros((kernel_size[0], (in_size[1] + 2 * self.shift[0]) * (in_size[2] + 2 * self.shift[1]) + 1, in_size[1] * in_size[2]))

        for k in range(kernel_size[0]):
            image_matrix_with_zeros = np.zeros((in_size[1] + 2 * self.shift[0], in_size[2] + 2 * self.shift[1]))

            image_matrix_with_zeros[:kernel_size[1], :kernel_size[2]] = self.kernel[k]

            kernel_column = np.reshape(image_matrix_with_zeros,
                                       ((in_size[1] + 2 * self.shift[0]) * (in_size[2] + 2 * self.shift[1]), 1))

            for m in range(in_size[1]):
                for n in range(in_size[2]):
                    bias = 0#random.random()
                    self.weights[k, :, m * (in_size[2]) + n] = np.append(np.roll(kernel_column, m * (in_size[2] + 2 * self.shift[1]) + n), bias)
        # По итогу мы сформировали матрицу весов размера (k, (n + 2) * (m + 2) + 1, n * m),
        # где к отвечает за количество карт, а оставшиеся числа являются матрицами весов


    def get_output(self, images: np.array) -> np.array:
        result = np.zeros(self.size)
        list_of_s = self.get_s(images)
        for k in range(self.size[0]):
            s = list_of_s[k]
            y = self.af(s)
            result[k] = y
        return result

    def get_s(self, images: np.array) -> np.array:
        result = np.zeros(self.size)
        for k in range(self.size[0]):
            image = np.zeros((self.size[1] + 2 * self.shift[0], self.size[2] + 2 * self.shift[1]))
            image[self.shift[0]: -self.shift[0], self.shift[1]: -self.shift[1]] = images[k]
            image = np.reshape(image, (1, image.shape[0] * image.shape[1]))
            kernel_matrix = self.weights[k]
            x = np.append(image, np.array([[-1]]), axis=1)
            s = np.dot(x, kernel_matrix)
            result[k] = np.reshape(s, self.size[1:])
        return result


    def get_error(self, in_images: np.array, errors: np.array) -> np.array:
        result = np.zeros(self.size)

        list_of_der_s = self.daf(self.get_s(in_images))
        out_images_current = self.get_output(in_images)
        # error = out_images_current - out_images
        error_with_der = errors * list_of_der_s
        for k in range(self.size[0]):
            w = (self.weights[k, :-1]).T
            error_multiplier = error_with_der[k]
            error_multiplier = np.reshape(error_multiplier, (1, error_multiplier.shape[0] * error_multiplier.shape[1]))
            temp_res = np.dot(error_multiplier, w)
            temp_res = np.reshape(temp_res, (self.size[1] + 2 * self.shift[0], self.size[2] + 2 * self.shift[1]))
            result[k] = temp_res[self.shift[0]: -self.shift[0], self.shift[1]: -self.shift[1]]
        return result


    def learn(self, in_image: np.array, errors: np.array):


        list_of_s = self.get_s(in_image)

        list_errors_with_der = errors * self.daf(list_of_s)

        for k in range(self.size[0]):

            image = np.zeros((self.size[1] + 2 * self.shift[0], self.size[2] + 2 * self.shift[1]))
            image[self.shift[0]: -self.shift[0], self.shift[1]: -self.shift[1]] = in_image[k]
            x = np.reshape(image, (1, image.shape[0] * image.shape[1]))
            x = np.append(x, np.array([[-1]]), axis=1)
            error_with_der = list_errors_with_der[k]
            error_with_der = np.reshape(list_errors_with_der[k], (error_with_der.shape[0] * error_with_der.shape[0], 1))
            pass

            res = np.dot(error_with_der, x).T

            delta = np.zeros(self.kernel_size[1:])


            for pos_in_res in range(res.shape[1]):
                kernel_with_zeros = res[:-1, pos_in_res]

                x, y = pos_in_res % self.size[2] + self.shift[1], pos_in_res // self.size[1] + self.shift[0]

                kernel_with_zeros_matrix = np.reshape(kernel_with_zeros, (self.size[1] + 2 * self.shift[0],
                                                                          self.size[2] + 2 * self.shift[1]))
                kernel_delta = kernel_with_zeros_matrix[y - self.shift[0]: y + self.shift[0] + 1, x - self.shift[1]: x + self.shift[1] + 1]
                delta += kernel_delta
                bias = res[-1, pos_in_res]
                self.weights[-1, pos_in_res] -= bias * self.lr


            delta /= self.size[1] * self.size[2]
            # print(self.lr * delta)

            self.kernel[k] -= self.lr * delta

            biases = self.weights[-1]

            image_matrix_with_zeros = np.zeros((self.size[1] + 2 * self.shift[0], self.size[2] + 2 * self.shift[1]))

            image_matrix_with_zeros[:self.kernel_size[1], :self.kernel_size[2]] = self.kernel[k]

            kernel_column = np.reshape(image_matrix_with_zeros,
                                       ((self.size[1] + 2 * self.shift[0]) * (self.size[2] + 2 * self.shift[1]), 1))

            for m in range(self.size[1]):
                for n in range(self.size[2]):
                    bias = 0  # random.random()
                    self.weights[k, :, m * (self.size[2]) + n] = np.append(
                        np.roll(kernel_column, m * (self.size[2] + 2 * self.shift[1]) + n), bias)

            self.weights[-1] = biases

            pass






class MapConvert(Operation):


    def __init__(self, in_size: tuple[int, int, int], out_map_count: int, bool_matrix: np.array):
        self.size = in_size
        self.out_size = (out_map_count, self.size[1], self.size[2])
        self.bool_matrix = bool_matrix
        self.af = np.vectorize(linear)
        self.daf = np.vectorize(derivative_of_linear)

    def get_output(self, images: np.array):
        result = np.zeros(self.out_size)
        for result_index in range(self.bool_matrix.shape[1]):
            for image_index in range(self.bool_matrix.shape[0]):
                if self.bool_matrix[image_index, result_index]:
                    result[result_index] += images[image_index]
            result[result_index] /= np.count_nonzero(self.bool_matrix[:, result_index])
        return result


    def get_error(self, in_images: np.array, errors: np.array) -> np.array:
        result = np.zeros(self.size)

        out_images_current = self.get_output(in_images)

        for k1 in range(out_images_current.shape[0]):
            # out_image_etalon = out_images[k1]
            # out_image_current = out_images_current[k1]

            bool_mask = self.bool_matrix[:, k1]

            count_of_source_matrix = np.count_nonzero(bool_mask)  # посчитаем сколько матриц было задействовано вначале, чтобы разделить ошибку между ними
            error = 1 / count_of_source_matrix * errors[k1]
            result[bool_mask] += error
        return result


class Pooling(Operation):

    def __init__(self, in_size: tuple[int, int, int], kernel_size: tuple[int, int]):
        self.size = in_size
        self.out_size = (in_size[0], ceil(in_size[1] / kernel_size[0]), ceil(in_size[2] / kernel_size[1]))
        self.pool_size = kernel_size

    def get_output(self, images: np.array) -> np.array:
        if images.shape == self.size[1:]:
            images = np.array([images])
        result = np.zeros(self.out_size)
        for k in range(self.size[0]):
            image = images[k]
            for y in range(self.out_size[1]):
                for x in range(self.out_size[2]):
                    result[k, y, x] = np.max(image[y * self.pool_size[0]: (y + 1) * self.pool_size[0],
                                                   x * self.pool_size[1]: (x + 1) * self.pool_size[1]])
        return result


    def get_error(self, in_images: np.array, errors: np.array) -> np.array:
        result = np.copy(in_images)
        out_images_current = self.get_output(in_images)
        for k in range(self.size[0]):
            # out_image_etalon = out_images[k]
            out_image_current = out_images_current[k]

            error = errors[k]#out_image_current - out_image_etalon

            for y in range(self.out_size[1]):
                for x in range(self.out_size[2]):
                    max_value = out_image_current[y, x]

                    mask = np.zeros_like(result, dtype=bool)
                    mask[k, y * self.pool_size[0]: (y + 1) * self.pool_size[0],
                            x * self.pool_size[1]: (x + 1) * self.pool_size[1]] = True
                    result[mask] = np.where(result[mask] >= max_value, error[y, x], 0)
        return result





class MLP_Layer(Operation):

    lr: float = 1

    def __init__(self, in_size: tuple[int, int, int], out_count_of_neurons: int, af, daf):
        self.out_size = (in_size[0], 1, out_count_of_neurons)
        self.size = in_size
        self.weights = np.random.rand(in_size[0] * in_size[1] * in_size[2] + 1, out_count_of_neurons)
        self.af = np.vectorize(af)
        self.daf = np.vectorize(daf)

        pass

    def get_s(self, images: np.array):


        if images.shape == self.size[1:]:
            images = np.array([images])

        x = np.reshape(images, (1, images.shape[0] * images.shape[1] * images.shape[2]))
        x = np.append(x, np.array([[-1]]), axis=1)
        return np.dot(x, self.weights)

    def get_output(self, images: np.array) -> np.array:
        s = self.get_s(images)
        y = self.af(s)
        return y

    def get_error(self, in_images: np.array, errors: np.array) -> np.array:

        if in_images.shape == self.out_size[1:]:
            in_images = np.array([in_images])

        error_with_der = errors * self.daf(self.get_s(in_images))

        res = np.dot(error_with_der, self.weights[:-1].T)
        return np.reshape(res, in_images.shape)

        pass



    def learn(self, images: np.array, errors: np.array):

        if images.shape == self.size[1:]:
            images = np.array([images])

        x = np.reshape(images, (1, images.shape[0] * images.shape[1] * images.shape[2]))
        x = np.append(x, np.array([[-1]]), axis=1)

        error_with_der_s = errors * self.daf(self.get_s(images))

        self.weights -= self.lr * np.dot(error_with_der_s.T, x).T
        # print(self.lr * np.dot(error_with_der_s.T, x).T)
        pass



class CNN:
    layers: list[MLP_Layer | Convolve | MapConvert | Pooling]
    def __init__(self):
        self.layers = [
            MapConvert((1, 28, 28), 6, matrix_for_converter1),
            Convolve((6, 28, 28), (9, 9), linear, derivative_of_linear),
            Pooling((6, 28, 28), (2, 2)),

            MapConvert((6, 14, 14), 15, matrix_for_converter2),
            Convolve((15, 14, 14), (5, 5), tanh, derivative_of_tanh),
            Pooling((15, 14, 14), (4, 4)),

            MapConvert((15, 4, 4), 30, matrix_for_converter3),
            Convolve((30, 4, 4), (3, 3), linear, derivative_of_linear),
            Pooling((30, 4, 4), (2, 2)),


            MLP_Layer((30, 2, 2), 30, linear, derivative_of_linear),
            MLP_Layer((1, 1, 30), 10, sigmoid, derivative_of_sigmoid)
        ]

        self.etalons = np.zeros((0, 10))
        self.trainset = np.zeros((0, 28, 28))


    def fit(self, error=1.0e-1):

        epoch = 0

        current_error = self.get_error()
        try:
            while current_error > error:
                self.learn()
                current_error = self.get_error()
                epoch += 1
                # break
                if epoch < 100 or epoch % 100 == 0:
                    print(f"#{epoch:6d} - {current_error:.3e}")
        except:
            pass
            # print(f"#{epoch:6d} - {current_error:.3e}")
        print(f"#{epoch:6d} - {current_error:.3e}")

        pass

    def learn(self):

        count_of_images = self.trainset.shape[0]
        for index, image in enumerate(self.trainset):
            output = np.copy(image)
            etalon = self.etalons[index]


            in_out_list = [output]


            for layer in self.layers:
                in_out_list.append(layer.get_output(in_out_list[-1]))


            error = [in_out_list[-1] - etalon]

            for index, layer in enumerate(self.layers[::-1]):
                error.append(layer.get_error(in_out_list[::-1][index+1], error[-1]))

            error = error[:-1]
            error = error[::-1]


            for index, layer in enumerate(self.layers):

                layer.learn(in_out_list[index], error[index])

            pass

        pass

    def get_error(self) -> np.array:
        error = 0
        count_of_images = self.trainset.shape[0]
        for index, image in enumerate(self.trainset):
            output = image
            etalon = self.etalons[index]
            for layer in self.layers:
                output = layer.get_output(output)
            for absolute_error in (output - etalon)[0]:
                error += absolute_error ** 2

        error /= 2 * count_of_images
        return error

    def load_trainset(self, path: str):
        temp = np.genfromtxt(path, skip_header=True, delimiter=',')

        self.etalons = np.zeros((temp.shape[0], 10))
        self.trainset = np.zeros((temp.shape[0], 28, 28))
        for j in range(temp[:, 0].shape[0]):
            self.etalons[j] = np.array([0 if i != temp[j, 0] else 1 for i in range(10)])
            self.trainset[j] = np.reshape(temp[j, 1:], (28, 28)) / 255

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.layers, f)

    def load(self, path: str):
        with open(path, 'rb') as f1:
            self.layers = pickle.load(f1)

    def test(self, path):
        temp = np.genfromtxt(path, skip_header=True, delimiter=',')

        etalons = np.zeros((temp.shape[0], 10))
        trainset = np.zeros((temp.shape[0], 28, 28))
        for j in range(temp[:, 0].shape[0]):
            etalons[j] = np.array([0 if i != temp[j, 0] else 1 for i in range(10)])
            trainset[j] = np.reshape(temp[j, 1:], (28, 28)) / 255


        for index, image in enumerate(trainset):
            output = np.copy(image)
            etalon = etalons[index]


            in_out_list = [output]


            for layer in self.layers:
                in_out_list.append(layer.get_output(in_out_list[-1]))

            result = in_out_list[-1]


            print(np.argmax(etalon), np.argmax(result))


