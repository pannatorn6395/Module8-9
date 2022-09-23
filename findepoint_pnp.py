
import cv2
import numpy as np
from Detection.perspective import *
from Detection.sideDetection_HSV import *
from itertools import count
import DetectionFunctions as df
from DetectAllPoints import *
from transform import order_points, poly2view_angle
from solve_pnp import *
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
category_reference = {0: 'bishop', 1: 'king', 2: 'knight', 3: 'pawn', 4: 'queen', 5: 'rook'}
model = load_model("model.h5")
chess_piece_height = {"king": (0.081, 0.097), "queen": (0.07, 0.0762), "bishop": (0.058, 0.065), "knight": (0.054, 0.05715), "rook": (0.02845, 0.048), "pawn": (0.043, 0.045)}
chess_piece_diameter = {"king": (0.028, 0.0381), "queen": (0.028, 0.0362), "bishop": (0.026, 0.032), "knight": (0.026, 0.03255), "rook": (0.026, 0.03255), "pawn": (0.0191, 0.02825)}
based_obj_points = np.array([[-0.2, -0.23, 0], [0.2, -0.23, 0], [0.2, 0.23, 0], [-0.2, 0.23, 0]])
cameraMatrix = np.array([[1438.4337197221366, 0.0, 934.4226787746103], [0.0, 1437.7513778197347, 557.7771398018671], [0.0, 0.0, 1.0]], np.float32) # Module
dist = np.array([[0.07229278436610362, -0.5836205675336522, 0.0003932499370206642, 0.0002754754987376089, 1.7293977700105942]])
mask_contour_index_list = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
scan_box_height = min(chess_piece_height['king'])
export_size = 224
if __name__ == "__main__":
    direction = Borad_Direction()
    img = cv2.imread('/Users/pannatorn/Desktop/chessboard/raw_dataset/0_180.jpg')
    # img = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2,))
    clear_image, encoded_image, matrix = getMatrixFromImage(img)
    img,points,all_point=show_point_on_image(img, matrix)
    print(points)
    four_points=expand_FourPoint(img,(np.array(points,dtype='float32')),offset = 38)
    ret, rvec, tvec = cv2.solvePnP(objectPoints=based_obj_points,
                                        imagePoints=np.array(four_points, dtype=np.double),
                                        cameraMatrix=cameraMatrix,
                                        distCoeffs=dist,
                                        flags=0)
    while True:
        canvas = img.copy()
        cv2.aruco.drawAxis(image=canvas, cameraMatrix=cameraMatrix, distCoeffs=dist, rvec=rvec, tvec=tvec, length=0.2)
        cv2.imshow("Rotated", canvas)
        key = cv2.waitKey(0)
        if key == ord(' '):
            cv2.destroyWindow("Rotated")
            break
    tvec=np.array(tvec,dtype=np.float32).reshape((1,3))
    CNNinputs_padded, angle_list=get_tile(img,rvec,tvec)
    out = model.predict(np.array([CNNinputs_padded[0]]))

    # top_pred = np.argmax(out) 
    # pred = category_reference[top_pred]
    # print(pred)

    # vertical_images = []
    for y in range(8):
        image_list_vertical = []
        for x in range(8):
            canvas = resize_and_pad(CNNinputs_padded[8 * y + x].copy())
    #         # cv2.putText(canvas, str(round(angle_list[8 * y + x])), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 255))
    #         # label = board[y][x]
    #         # if label != 0:
    #         #     cv2.putText(canvas, labels[label - 1], (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #         #                 color=(0, 0, 255))
            out = model.predict(np.array([CNNinputs_padded[8 * y + x]]))
            top_pred = np.argmax(out[0]) 
            pred = category_reference[top_pred]
            print(pred)
            cv2.imshow("All CNN inputs", canvas)
            key = cv2.waitKey(0)
            if key == ord(' '):
                cv2.destroyWindow("All CNN inputs")
                break
    #         image_list_vertical.append(
    #             cv2.copyMakeBorder(canvas, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, (0, 255, 0)))
                
    #     vertical_images.append(np.vstack(image_list_vertical))
    # combined_images = np.hstack(vertical_images)
    # cv2.imshow("All CNN inputs", combined_images)
    # cv2.waitKey(0)
