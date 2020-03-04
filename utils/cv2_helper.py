import cv2
import math
import numpy as np
import imutils

def display_image(img, width):
    img = imutils.resize(img, width=width)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def find_midpoint(vertices):
    pass

def angular_correction(key, frame, vertices, padding=3):
    p1 = [vertices[0][0], -vertices[0][1]]
    p2 = [vertices[3][0], -vertices[3][1]]
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])             
    angle_rad = math.atan(abs(p2[1] - p1[1]) / abs(p2[0] - p1[0]))
    angle_deg = math.degrees(angle_rad) if m < 0 else -math.degrees(angle_rad)

    temp_frame = imutils.rotate(frame, angle_deg)
    height, width, channels = temp_frame.shape
    frame_center = (width/2, height/2)
    
    M = cv2.getRotationMatrix2D(frame_center, angle_deg, 1)

    ones = np.ones(shape=(len(vertices), 1))
    v_ones = np.hstack([vertices, ones])
    transformed_points = M.dot(v_ones.T).T

    top_left = (int(min([transformed_points[0][0], transformed_points[1][0]])) - padding, int(min([transformed_points[1][1], transformed_points[2][1]])) - padding)
    btm_right = (int(max([transformed_points[2][0], transformed_points[3][0]]) + padding), int(max([transformed_points[0][1], transformed_points[3][1]])) + padding)

    # Comment out the lines below to visualize the rotation correction for debugging
    # for j in range(4):
    #     p1 = (int(vertices[j][0]), int(vertices[j][1]))
    #     p2 = (int(vertices[(j + 1) % 4][0]), int(vertices[(j + 1) % 4][1]))
    #     cv2.line(temp_frame, p1, p2, (255, 170, 255), 2, cv2.LINE_AA)
    # for j in range(4):
    #     p1 = (int(transformed_points[j][0]), int(transformed_points[j][1]))
    #     p2 = (int(transformed_points[(j + 1) % 4][0]), int(transformed_points[(j + 1) % 4][1]))
    #     cv2.line(temp_frame, p1, p2, (255, 170, 0), 2, cv2.LINE_AA)
    # cv2.rectangle(temp_frame,
    #   top_left,
    #   btm_right,
    #   (0, 0, 255), 2
    # )
    # cv2.imshow(f'temp_{key}', temp_frame[int(top_left[1]):int(btm_right[1]), int(top_left[0]):int(btm_right[0])])

    return temp_frame[int(top_left[1]):int(btm_right[1]), int(top_left[0]):int(btm_right[0])]