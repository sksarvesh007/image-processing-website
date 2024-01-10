import gunicorn
from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from io import BytesIO

app = Flask(__name__)

def blur_image(image):
    blurred = cv2.GaussianBlur(image, (25, 25), 0)
    return blurred

def sharpen_image(image):
    gaussian_blur = cv2.GaussianBlur(image, (25, 25), 0)
    sharpened = cv2.addWeighted(image, 1.5, gaussian_blur, -0.5, 0)
    return sharpened

def process_image(image, operation):
    if operation == 'blur':
        return blur_image(image)
    elif operation == 'sharpen':
        return sharpen_image(image)
    else:
        return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        operation = request.form.get('operation')
        processed_image = process_image(image, operation)
        _, buffer = cv2.imencode('.png', processed_image)
        buffer = BytesIO(buffer)

        return send_file(buffer, mimetype='image/png', as_attachment=True, download_name='processed_image.png')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
