from flask import Flask, render_template, request, redirect
from flask_restful import Resource, Api
import requests
import numpy as np
import base64

from server.endpoints.image_compute import ImageCompute

from east_text_detector.detector import EASTDetector
from crnn_text_recognizer.recognizer import CRNNRecognizer

east_model_path = './east_text_detector/frozen_east_text_detection.pb'
crnn_model_path = './crnn_text_recognizer/crnn_text_recognizer_best.pth'
demo_image_path = './assets'
white_list = ['.DS_Store']

text_detector = EASTDetector(east_model_path)
text_recognizer = CRNNRecognizer(crnn_model_path)

app = Flask(__name__)
api = Api(app)

api.add_resource(ImageCompute, '/compute', resource_class_kwargs={'detector': text_detector, 'recognizer': text_recognizer})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files.get('image')
        img_b64 = base64.b64encode(image.read()).decode('ascii')

        res = requests.post('http://localhost:5000/compute', data={
          'image': img_b64
        })

        computed_img_b64 = res.json()['image']

        return render_template('home.html', image=f'data:image/png;base64,{computed_img_b64}')
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
    # app.run()