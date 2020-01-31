import cv2
import imutils

y_threshold = 20

def display_image(img, width):
    img = imutils.resize(img, width=width)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def sentence_formatter(words):
    print(words)