import math


class Kernels:
    @staticmethod
    def generate_gaussian_kernel(size, sigma):
        if size % 2 == 0:
            raise ValueError("Tamaño impar requerido")

        kernel = [[0.0 for _ in range(size)] for _ in range(size)]
        center = size // 2
        sum_val = 0

        for i in range(size):
            for j in range(size):
                x = i - center
                y = j - center
                value = math.exp(-(x * x + y * y) / (2 * sigma * sigma))
                kernel[i][j] = value
                sum_val += value

        for i in range(size):
            for j in range(size):
                kernel[i][j] /= sum_val

        return kernel

    @staticmethod
    def generate_sobel_x_kernel(size):
        kernel = [[0.0 for _ in range(size)] for _ in range(size)]
        center = size // 2

        for i in range(size):
            for j in range(size):
                kernel[i][j] = (j - center) 

        return kernel

    @staticmethod
    def generate_laplacian_kernel(size):
        kernel = [[-1 for _ in range(size)] for _ in range(size)]
        center = size // 2

        kernel[center][center] = size * size - 1

        return kernel