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

    dataset_name = 'test100.csv'
    error = 0.25
    cnn: CNN = CNN(layers)

    is_learn = False

    if is_learn:
        cnn.load_trainset(f'datasets/{dataset_name}')    # загружаем датасет
        cnn.load('weights/test100.csv 0.5.wbc')       # загружаем предобученные веса
        cnn.fit(error)                                   # обучаем до ошибки
        cnn.save(f'weights/{dataset_name} {error}.wbc')  # сохраняем веса
        cnn.test(f'datasets/{dataset_name}')             # проверяем работу
    else:
        cnn.load('weights/test100.csv 0.25.wbc')           # загружаем предобученные веса
        cnn.load_trainset(f'datasets/{dataset_name}')        # загружаем датасет
        print(cnn.get_error())                               # выводим ошибку на датасете
        cnn.test(f'datasets/mnist_test.csv')                 # выводим тестовые данные
        # cnn.test(f'datasets/{dataset_name}')                 # выводим тестовые данные
        # cnn.test_with_vizual(f'datasets/{dataset_name}')     # выводим цифру и процент угаданности
        cnn.kernel_viz()                                     # визуализируем ядра свертки


if __name__ == '__main__':
    main()
