import cv2
import imutils

def display_image(img, width):
    img = imutils.resize(img, width=width)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)