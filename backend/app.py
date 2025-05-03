from flask import Flask, request, render_template, url_for
from PIL import Image
import os
import time
from image import Imagen
from kernels import Kernels

app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'img')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

img_processor = Imagen()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def procesar_imagen():
    archivo = request.files.get('imagen')
    if not archivo or archivo.filename == '':
        return "No se seleccionó ninguna imagen", 400

    try:
        filtro = request.form.get('filtro', 'default')
        kernel_size = int(request.form.get('kernel_size', 3))

        if kernel_size != 3:
            return "Por ahora solo se admite kernel 3x3", 400

        img = Image.open(archivo).convert('RGB')
        image_size = img.size 

        original_filename = "imagen_original.png"
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        img.save(original_path)

        if filtro == "gaussian":
            kernel = Kernels.generate_gaussian_kernel(kernel_size, sigma=4.5)
        elif filtro == "sobel_x":
            kernel = Kernels.generate_sobel_x_kernel(kernel_size)
        elif filtro == "laplacian":
            kernel = Kernels.generate_laplacian_kernel(kernel_size)
        else:
            return "Filtro no reconocido", 400

        inicio = time.time()
        result_img, grid_size, block_size = img_processor.apply_convolution_parallel_rgb(img, kernel)
        fin = time.time()
        tiempo_ejecucion = round(fin - inicio, 4) * 1000  

        result_filename = f"{filtro}.png"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        result_img.save(result_path)

        return render_template(
            'index.html',
            imagen_original=url_for('static', filename=f'img/{original_filename}'),
            imagen_resultado=url_for('static', filename=f'img/{result_filename}'),
            filtro_aplicado=filtro,
            kernel_size=kernel_size,
            tiempo_ejecucion=tiempo_ejecucion,
            tamaño_imagen=f"{image_size[0]} x {image_size[1]}",
            grid_size=grid_size,
            block_size=block_size,
        )

    except Exception as e:
        return f"Ocurrió un error al procesar la imagen: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
