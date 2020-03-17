import cv2
import base64
import requests
import numpy as np
from PIL import Image
from io import BytesIO

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()        

    def get_frame(self):
        ret, frame = self.video.read()

        img_b64 = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode()
        res = requests.post('http://localhost:5000/compute', data={
          'image': img_b64
        })

        computed_img_b64 = res.json()['image']
        image = Image.open(BytesIO(base64.b64decode(computed_img_b64))).convert('RGB')
        image_cv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        ret, jpeg = cv2.imencode('.jpg', image_cv)

        return jpeg.tobytes()