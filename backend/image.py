import pycuda.driver as cuda
import numpy as np
from PIL import Image
from pycuda.compiler import SourceModule
import os

class Imagen:
    @staticmethod
    def obtener_info_cuda():
        cuda.init()
        info = {
            "cantidad_dispositivos": cuda.Device.count(),
            "dispositivos": []
        }

        for i in range(cuda.Device.count()):
            dev = cuda.Device(i)
            dispositivo_info = {
                "nombre": dev.name(),
                "compute_capability": "%d.%d" % dev.compute_capability(),
                "memoria_total": f"{dev.total_memory()} bytes",
                "multiprocesadores": dev.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
            }
            info["dispositivos"].append(dispositivo_info)

        return info

    def apply_convolution_parallel_rgb(self, image, kernel):
        cuda.init()
        device = cuda.Device(0)
        context = device.make_context()

        try:
            width, height = image.size
            size_kernel = len(kernel)
            margin = size_kernel // 2

            img_array = np.array(image).astype(np.float32)
            kernel = np.array(kernel, dtype=np.float32)

            if img_array.ndim != 3 or img_array.shape[2] != 3:
                raise ValueError("La imagen debe estar en formato RGB (3 canales).")

            flat_img_array = img_array.flatten()
            result_image = np.zeros_like(flat_img_array)

            img_array_gpu = cuda.mem_alloc(flat_img_array.nbytes)
            result_image_gpu = cuda.mem_alloc(result_image.nbytes)
            kernel_gpu = cuda.to_device(kernel.flatten())

            cuda.memcpy_htod(img_array_gpu, flat_img_array)

            block_size = (16, 16, 1)
            grid_size = (
                int(np.ceil(width / block_size[0])),
                int(np.ceil(height / block_size[1])),
                1
            )

            convolution_kernel = """
            __global__ void apply_convolution(float *img_array, float *kernel, float *result_image, int width, int height, int kernel_size, int margin) {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= width || y >= height)
                    return;

                if (x >= margin && x < width - margin && y >= margin && y < height - margin) {
                    float red_sum = 0.0f;
                    float green_sum = 0.0f;
                    float blue_sum = 0.0f;

                    for (int i = -margin; i <= margin; i++) {
                        for (int j = -margin; j <= margin; j++) {
                            int img_x = x + i;
                            int img_y = y + j;

                            int index = (img_y * width + img_x) * 3;
                            float pixel_r = img_array[index];
                            float pixel_g = img_array[index + 1];
                            float pixel_b = img_array[index + 2];

                            float kernel_value = kernel[(i + margin) * kernel_size + (j + margin)];

                            red_sum += pixel_r * kernel_value;
                            green_sum += pixel_g * kernel_value;
                            blue_sum += pixel_b * kernel_value;
                        }
                    }

                    int result_index = (y * width + x) * 3;
                    result_image[result_index]     = fminf(fmaxf(red_sum, 0.0f), 255.0f);
                    result_image[result_index + 1] = fminf(fmaxf(green_sum, 0.0f), 255.0f);
                    result_image[result_index + 2] = fminf(fmaxf(blue_sum, 0.0f), 255.0f);
                }
            }
            """

            module = SourceModule(convolution_kernel)
            convolution_func = module.get_function("apply_convolution")

            convolution_func(
                img_array_gpu, kernel_gpu, result_image_gpu,
                np.int32(width), np.int32(height), np.int32(size_kernel), np.int32(margin),
                block=block_size, grid=grid_size
            )

            cuda.memcpy_dtoh(result_image, result_image_gpu)
            result_image = result_image.reshape((height, width, 3)).astype(np.uint8)

            return Image.fromarray(result_image), grid_size, block_size
        
        except cuda.Error as e:
            raise RuntimeError(f"Error al usar la GPU: {e}")

        finally:
            context.pop()
