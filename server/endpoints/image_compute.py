from flask import request, render_template
from flask_cors import cross_origin
from flask_restful import Resource, Api, reqparse
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import cv2
from utils.cv2_helper import angular_correction
from utils.sentence_formatter import format_sentence

parser = reqparse.RequestParser()

'''
body = {
  image: base64 encoded-image
}

Returns a base64 encoded-image
'''

class ImageCompute(Resource):
    def __init__(self, detector, recognizer):
        self.detector = detector
        self.recognizer = recognizer

    @cross_origin()
    def post(self):
        parser.add_argument('image')
        parser.add_argument('show_sentence')

        args = parser.parse_args()
        img_b64 = args['image']
        show_sentence = args['show_sentence']

        image = Image.open(BytesIO(base64.b64decode(img_b64))).convert('RGB')
        image_cv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        boxes, confidences, indices, width_ratio, height_ratio = self.detector.detect(image_cv)
        index_map = {}

        for i in indices:
            key = i[0]
            vertices = cv2.boxPoints(boxes[i[0]])
            for j in range(4):
                vertices[j][0] *= width_ratio
                vertices[j][1] *= height_ratio

            top_left = (min([vertices[0][0], vertices[1][0]]), min([vertices[1][1], vertices[2][1]]))
            btm_right = (max([vertices[2][0], vertices[3][0]]), max([vertices[0][1], vertices[3][1]]))

            if top_left[0] < 0 or top_left[1] < 0:
                continue

            text_roi = angular_correction(key, image_cv, vertices)

            if text_roi.shape[0] > 0 and text_roi.shape[1] > 0:
                index_map[key] = {
                  'vertices': vertices,
                  'pred_text': self.recognizer.predict(text_roi),
                }

        if len(index_map.keys()):
            try:
                for i in indices:
                    key = i[0]
                    for j in range(4):
                        p1 = (index_map[key]['vertices'][j][0], index_map[key]['vertices'][j][1])
                        p2 = (index_map[key]['vertices'][(j + 1) % 4][0], index_map[key]['vertices'][(j + 1) % 4][1])
                        cv2.line(image_cv, p1, p2, (255, 170, 0), 2, cv2.LINE_AA)
                    
                    cv2.putText(image_cv, index_map[key]['pred_text'], (index_map[key]['vertices'][1][0], index_map[key]['vertices'][1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            except:
                pass

        img_str = base64.b64encode(cv2.imencode('.jpg', image_cv)[1]).decode()

        if show_sentence:
            sentences = format_sentence(index_map)
        else:
            sentences = []

        return {
          'image': img_str,
          'sentences': sentences
        }