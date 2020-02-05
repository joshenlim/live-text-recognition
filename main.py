import argparse
import cv2
import os
import imutils
import math
from datetime import datetime
from east_text_detector.detector import EASTDetector
from crnn_text_recognizer.recognizer import CRNNRecognizer
from utils.logger import Logger
from utils.cv2_helper import display_image

parser = argparse.ArgumentParser()
parser.add_argument('--live', action='store_true', help='Runs program through a live camera feed')
parser.add_argument('--viewWidth', type=int, default=640, help='Width of camera frame')

east_model_path = './east_text_detector/frozen_east_text_detection.pb'
crnn_model_path = './crnn_text_recognizer/crnn_text_recognizer_finetuned.pth'
demo_image_path = './assets'
white_list = ['.DS_Store']

log = Logger()

def compute_frame(frame):
    boxes, confidences, indices, width_ratio, height_ratio = text_detector.detect(frame)

    start = datetime.now()
    for i in indices:
        vertices = cv2.boxPoints(boxes[i[0]])
        for j in range(4):
            vertices[j][0] *= width_ratio
            vertices[j][1] *= height_ratio

        top_left = (min([vertices[0][0], vertices[1][0]]), min([vertices[1][1], vertices[2][1]]))
        btm_right = (max([vertices[2][0], vertices[3][0]]), max([vertices[0][1], vertices[3][1]]))

        if top_left[0] < 0 or top_left[1] < 0:
            continue

        text_roi = frame[int(top_left[1]):int(btm_right[1]), int(top_left[0]):int(btm_right[0])]
        predicted_text = text_recognizer.predict(text_roi)

        for j in range(4):
            p1 = (vertices[j][0], vertices[j][1])
            p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
            cv2.line(frame, p1, p2, (255, 170, 0), 2, cv2.LINE_AA)
        
        cv2.putText(frame, predicted_text, (vertices[1][0], vertices[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    # log.info(f'Recognition Time: {(datetime.now() - start).total_seconds() * 100:.2f} ms')      
    # cv2.putText(frame, "{:.3f}".format(confidences[i[0]]), (vertices[0][0], vertices[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    return frame
    
if __name__ == "__main__":
    args = parser.parse_args()
    text_detector = EASTDetector(east_model_path)
    text_recognizer = CRNNRecognizer(crnn_model_path)

    # Run the program through a live camera feed
    if args.live:
        log.info('Initializing Camera Feed')
        cap = cv2.VideoCapture(0)
        log.info('Press q to quit streaming')
        while(cap.isOpened()):
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=args.viewWidth)

            detected_frame = compute_frame(frame)

            cv2.imshow('frame', detected_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    # Demonstrate program with a sequential feed of single input images
    else:
        log.info('Running computation for demo images')
        for file_name in os.listdir(demo_image_path):
            if file_name in white_list:
                continue
            input_image_path = f'{demo_image_path}/{file_name}'
            frame = cv2.imread(input_image_path)
            frame = compute_frame(frame)
            log.info('Press q to view next image')
            display_image(frame, args.viewWidth)