from flask import Flask, render_template, request, redirect, Response
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
from server.camera import VideoCamera
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

# video_stream = VideoCamera()

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

api = Api(app)
api.add_resource(ImageCompute, '/compute', resource_class_kwargs={'detector': text_detector, 'recognizer': text_recognizer})

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/health', methods=['GET', 'POST'])
def health():
    return {'status': 'good'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files.get('image')
        img_b64 = base64.b64encode(image.read()).decode('ascii')

        res = requests.post('http://localhost:5000/compute', data={
          'image': img_b64
        })

        computed_img_b64 = res.json()['image']
        return render_template('home.html', static=True, image=f'data:image/png;base64,{computed_img_b64}')

    else:
        mode = request.args.get('mode')
        if mode == 'static':
            return render_template('home.html', static=True)
        elif mode == 'live':
            return render_template('home.html', live=True)
        else:
            return render_template('home.html', static=True)

@app.route('/video_feed')
def video_feed():
    return None
    # return Response(gen(video_stream), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
    # app.run()