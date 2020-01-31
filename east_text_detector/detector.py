import cv2
import math
from utils.logger import Logger

log = Logger()

'''
Handles the text detection task in the pipeline for scene text recognition
Currently using a pre-trained EAST model that can be downloaded at:
https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1.
Large part of this file was referenced from the following URL:
https://github.com/spmallick/learnopencv/blob/master/TextDetectionEAST/textDetection.py
'''

class EASTDetector:
    def __init__(self, model_path, conf_threshold=0.5, nms_threshold=0.4):
        log.info('Loading pre-trained EAST Detector model')
        self.net = cv2.dnn.readNet(model_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def detect(self, img, input_width=320, input_height=320):
        img_height, img_width = img.shape[:2]

        width_ratio = img_width / float(input_width)
        height_ratio = img_height / float(input_height)

        blob = cv2.dnn.blobFromImage(img, 1.0, (input_width, input_height), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        outputLayers = []
        outputLayers.append("feature_fusion/Conv_7/Sigmoid")
        outputLayers.append("feature_fusion/concat_3")

        self.net.setInput(blob)
        scores, geometry = self.net.forward(outputLayers)

        [boxes, confidences] = self.decode(scores, geometry, self.conf_threshold)
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, self.conf_threshold, self.nms_threshold)

        t, _ = self.net.getPerfProfile()
        # log.info(f'Inference Time: {(t * 1000.0 / cv2.getTickFrequency()):.2f}ms')

        return boxes, confidences, indices, width_ratio, height_ratio

    def decode(self, scores, geometry, scoreThresh):
        detections = []
        confidences = []

        assert len(scores.shape) == 4, "Incorrect dimensions of scores"
        assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
        assert scores.shape[0] == 1, "Invalid dimensions of scores"
        assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
        assert scores.shape[1] == 1, "Invalid dimensions of scores"
        assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
        assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
        assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
        height = scores.shape[2]
        width = scores.shape[3]

        for y in range(0, height):
            scoresData = scores[0][0][y]
            x0_data = geometry[0][0][y]
            x1_data = geometry[0][1][y]
            x2_data = geometry[0][2][y]
            x3_data = geometry[0][3][y]
            anglesData = geometry[0][4][y]
            for x in range(0, width):
                score = scoresData[x]

                if(score < scoreThresh):
                    continue

                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = anglesData[x]

                cosA = math.cos(angle)
                sinA = math.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

                p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
                p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
                center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
                detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
                confidences.append(float(score))

        return [detections, confidences]