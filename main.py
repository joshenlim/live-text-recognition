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
from utils.cv2_helper import angular_correction
from utils.sentence_formatter import format_sentence

parser = argparse.ArgumentParser()
parser.add_argument('--live', action='store_true', help='Runs program through a live camera feed')
parser.add_argument('--sentence', action='store_true', help='Display detected text in its original order')
parser.add_argument('--angleCorrection', action='store_true', help='Enable angular correction for detected texts (Should improve accuracy)')
parser.add_argument('--verbose', action='store_true', help='Display logs for debugging')
parser.add_argument('--viewWidth', type=int, default=640, help='Width of camera frame')

'''
Accuracy                             ICDAR13 / IIIT5k
crnn_text_recognizer                  86.00  / 77.23
crnn_text_recognizer_finetuned        86.46  / 78.30
crnn_text_recognizer_best             87.75  / 78.10
crnn_text_recognizer_best_finetuned   86.11  / 80.00
'''

east_model_path = './east_text_detector/frozen_east_text_detection.pb'
crnn_model_path = './crnn_text_recognizer/crnn_text_recognizer_best.pth'
demo_image_path = './assets'
white_list = ['.DS_Store', 'Collated']

log = Logger()
args = parser.parse_args()
text_detector = EASTDetector(east_model_path)
text_recognizer = CRNNRecognizer(crnn_model_path)

def compute_frame(frame, show_sentence=False, correct_angle=False, debug=False):
    start = datetime.now()
    boxes, confidences, indices, width_ratio, height_ratio = text_detector.detect(frame)
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

        if correct_angle:
            text_roi = angular_correction(key, frame, vertices)
        else:
            text_roi = frame[int(top_left[1]):int(btm_right[1]), int(top_left[0]):int(btm_right[0])]

        if text_roi.shape[0] > 0 and text_roi.shape[1] > 0:
            # cv2.imshow(f't_{key}', text_roi)
            index_map[key] = {
              'vertices': vertices,
              'pred_text': text_recognizer.predict(text_roi),
            }

    if len(index_map.keys()):
        try:
            for i in indices:
                key = i[0]
                for j in range(4):
                    p1 = (index_map[key]['vertices'][j][0], index_map[key]['vertices'][j][1])
                    p2 = (index_map[key]['vertices'][(j + 1) % 4][0], index_map[key]['vertices'][(j + 1) % 4][1])
                    cv2.line(frame, p1, p2, (255, 170, 0), 2, cv2.LINE_AA)
                
                cv2.putText(frame, index_map[key]['pred_text'], (index_map[key]['vertices'][1][0], index_map[key]['vertices'][1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        except:
            pass

    # log.info(f'Recognition Time: {(datetime.now() - start).total_seconds() * 100:.2f} ms')

    if show_sentence:
        format_sentence(index_map, debug, pretty_print=True)

    return frame
    
if __name__ == "__main__":
    # Run the program through a live camera feed
    if args.live:
        log.info('Initializing Camera Feed')
        cap = cv2.VideoCapture(0)
        log.info('Press q to quit streaming')
        while(cap.isOpened()):
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=args.viewWidth)

            detected_frame = compute_frame(frame, correct_angle=args.angleCorrection)

            cv2.imshow('frame', detected_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    # Demonstrate program with a sequential feed of single input images
    else:
        log.info('Running computation for demo images')
        for file_name in os.listdir(demo_image_path):

            # Comment out the lines 113-114 if you want to run for all demo images
            # if file_name != "demo_4.jpg":
            #     continue

            if file_name in white_list:
                continue

            input_image_path = f'{demo_image_path}/{file_name}'
            frame = cv2.imread(input_image_path)
            frame = compute_frame(frame, show_sentence=args.sentence, correct_angle=args.angleCorrection, debug=args.verbose)
            log.info('Press q to view next image')
            display_image(frame, args.viewWidth)