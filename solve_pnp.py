from traceback import print_tb
import cv2, imutils
import math
import os, glob
import numpy as np
import cv2
from transform import order_points, poly2view_angle
chess_piece_height = {"king": (0.081, 0.097), "queen": (0.07, 0.0762), "bishop": (0.058, 0.065), "knight": (0.054, 0.05715), "rook": (0.02845, 0.048), "pawn": (0.043, 0.045)}
chess_piece_diameter = {"king": (0.028, 0.0381), "queen": (0.028, 0.0362), "bishop": (0.026, 0.032), "knight": (0.026, 0.03255), "rook": (0.026, 0.03255), "pawn": (0.0191, 0.02825)}
based_obj_points = np.array([[-0.2, -0.23, 0], [0.2, -0.23, 0], [0.2, 0.23, 0], [-0.2, 0.23, 0]])
cameraMatrix = np.array([[1438.4337197221366, 0.0, 934.4226787746103], [0.0, 1437.7513778197347, 557.7771398018671], [0.0, 0.0, 1.0]], np.float32) # Module
dist = np.array([[0.07229278436610362, -0.5836205675336522, 0.0003932499370206642, 0.0002754754987376089, 1.7293977700105942]])
mask_contour_index_list = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
scan_box_height = min(chess_piece_height['king'])
export_size = 224

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
def rotate(rvec, angle):
    rotM = np.zeros(shape=(3, 3))
    new_rvec = np.zeros(3)
    rotM, _ = cv2.Rodrigues(rvec, rotM, jacobian=0)  # Convert from rotation vector -> rotation matrix (rvec=rot_vector, rotM=rot_matrix)
    rotM = np.dot(rotM, np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]))  # Rotate zeta degree about Z-axis
    new_rvec, _ = cv2.Rodrigues(rotM, new_rvec, jacobian=0)  # Convert from rotation matrix -> rotation vector
    return new_rvec
def click_corner(event, x, y, flags, param):
    global img, four_points, canvas
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(canvas, (x, y), 5, (255, 0, 0), -1)
        print(x,y)
        four_points.append((x, y))
    if event == cv2.EVENT_MOUSEMOVE:
        canvas = img.copy()
        cv2.line(canvas, (x, 0), (x, 1080), (0,255,0), 1)
        cv2.line(canvas, (0, y), (1920, y), (0, 255, 0), 1)
        for (ax, ay) in four_points: cv2.circle(canvas, (ax, ay), 5, (255, 0, 0), -1)
def getBox2D(rvec, tvec, size = 0.05, height = scan_box_height):
    objpts = np.float32([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0], [0, 0, height], [size, 0, height], [size, size, height], [0, size, height]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, dist)
    min_x = int(min(imgpts, key=lambda x: x[0][0]).ravel()[0])
    max_x = int(max(imgpts, key=lambda x: x[0][0]).ravel()[0])
    min_y = int(min(imgpts, key=lambda x: x[0][1]).ravel()[1])
    max_y = int(max(imgpts, key=lambda x: x[0][1]).ravel()[1])
    return (min_x, min_y), (max_x, max_y)
def getPoly2D(rvec, tvec, size = 0.05):
    objpts = np.float32([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, dist)
    return imgpts
def getValidContour2D(rvec, tvec, size = 0.05, height = scan_box_height):
    objpts = np.float32([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0], [0, 0, height], [size, 0, height], [size, size, height], [0, size, height]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, dist)
    valid_contours = []
    for mask_contour_index in mask_contour_index_list:
        valid_contours.append([imgpts[mask_contour_index[i]] for i in range(len(mask_contour_index))])
    return valid_contours
def llr_tile(rvec, tvec):
    rotM = np.zeros(shape=(3, 3))
    rotM, _ = cv2.Rodrigues(rvec, rotM, jacobian=0)
    ### Draw chess piece space ###
    counter = 0
    tile_volume_bbox_list, angle_list, valid_contours_list = [], [], []
    for y in range(3, -5, -1):
        for x in range(-4, 4):
            board_coordinate = np.array([x * 0.05, y * 0.05, 0.0])
            (min_x, min_y), (max_x, max_y) = getBox2D(rvec, tvec + np.dot(board_coordinate, rotM.T), size=0.05, height=scan_box_height)
            tile_volume_bbox_list.append([(min_x, min_y), (max_x, max_y)])

            # find angle of each tile
            translated_tvec = tvec + np.dot(board_coordinate, rotM.T)
            poly_tile = getPoly2D(rvec, translated_tvec, size=0.05)
            valid_contours = getValidContour2D(rvec, translated_tvec, size=0.05, height=scan_box_height)
            valid_contours_list.append(valid_contours)
            angle_rad = poly2view_angle(poly_tile)

            angle_deg = angle_rad / 3.14 * 180
            angle_list.append(angle_deg)
            counter += 1
    tile_volume_bbox_list_new, angle_list_new = [], []
    for i in range(64):
        y, x = 7 - int(i / 8), i % 8
        tile_volume_bbox_list_new.append(tile_volume_bbox_list[8*y+x])
        angle_list_new.append(angle_list[8*y+x])
    return tile_volume_bbox_list, angle_list, valid_contours_list
def getCNNinput(img, bbox_list, valid_contours_list):
    CNNinputs = []
    for i in range(len(bbox_list)):
        [(min_x, min_y), (max_x, max_y)] = bbox_list[i]
        if min_x < 0: min_x = 0
        if min_y < 0: min_y = 0
        if max_x >= img.shape[1]: max_x = img.shape[1]-1
        if max_y >= img.shape[0]: max_y = img.shape[0]-1
        cropped = img[min_y:max_y, min_x:max_x].copy()
        valid_contours = valid_contours_list[i]
        mask = np.zeros(cropped.shape[:2], dtype="uint8")
        for valid_contour in valid_contours:
            local_valid_contour = []
            for point in valid_contour:
                x = int(point[0][0] - min_x)
                y = int(point[0][1] - min_y)
                local_valid_contour.append([x, y])
            local_valid_contour = np.array(local_valid_contour).reshape((-1, 1, 2)).astype(np.int32)
            cv2.drawContours(mask, [local_valid_contour], -1, 255, -1)
        CNNinputs.append(cv2.bitwise_and(cropped, cropped, mask=mask))
    return CNNinputs
def resize_and_pad(img, size=300, padding_color=(0,0,0)):
    old_size = img.shape[:2]
    ratio = float(size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = size - new_size[1]
    delta_h = size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
def get_tile(img, rvec, tvec):
    tile_volume_bbox_list, angle_list, valid_contours_list = llr_tile(rvec, tvec)
    CNNinputs = getCNNinput(img, tile_volume_bbox_list, valid_contours_list)
    CNNinputs_padded = []
    for i in range(64):
        CNNinput_padded = resize_and_pad(CNNinputs[i], size=export_size)
        CNNinputs_padded.append(CNNinput_padded)
    return CNNinputs_padded, angle_list
# img=cv2.imread('/Users/pannatorn/Desktop/chessboard/PNP/chessboard/0_150.jpg')
# winName = 'Result'
# clicked = np.zeros((4,2),dtype='float32')
# count = 0
# trigger = False
# cv2.namedWindow(winName)
# cv2.setMouseCallback(winName, click_corner)
# offset = 20
# four_points = [] #buffer for recieve input
# labelNames = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]
# labels = ['b', 'k', 'n', 'p', 'q', 'r'] 
# canvas = img.copy()
# while True:  # Loop until 4 corners assigned
#     cv2.imshow(winName, canvas)
#     k = cv2.waitKey(1) & 0xFF
#     if len(four_points) == 4 and k == ord(' '):
#         ret, rvec, tvec = cv2.solvePnP(objectPoints=based_obj_points,
#                                         imagePoints=np.array(four_points, dtype=np.double),
#                                         cameraMatrix=cameraMatrix,
#                                         distCoeffs=dist,
#                                         flags=0)
#         while True:
#             canvas = img.copy()
#             cv2.aruco.drawAxis(image=canvas, cameraMatrix=cameraMatrix, distCoeffs=dist, rvec=rvec, tvec=tvec, length=0.2)
#             cv2.imshow("Rotated", canvas)
#             key = cv2.waitKey(0)
#             if key == ord(' '):
#                 cv2.destroyWindow("Rotated")
#                 break
#         tvec=np.array(tvec,dtype=np.float32).reshape((1,3))
#         CNNinputs_padded, angle_list=get_tile(img,rvec,tvec)
#         vertical_images = []
#         for y in range(8):
#             image_list_vertical = []
#             for x in range(8):
#                 x=7-x
#                 canvas = resize_and_pad(CNNinputs_padded[8 * y + x].copy(), size=100)
#                 # cv2.putText(canvas, str(round(angle_list[8 * y + x])), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 255))
#                 # label = board[y][x]
#                 # if label != 0:
#                 #     cv2.putText(canvas, labels[label - 1], (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                 #                 color=(0, 0, 255))
#                 cv2.imshow("All CNN inputs", canvas)
#                 key = cv2.waitKey(0)
#                 if key == ord(' '):
#                     cv2.destroyWindow("All CNN inputs")
#                     break
#                 image_list_vertical.append(
#                     cv2.copyMakeBorder(canvas, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, (0, 255, 0)))
                    
#             vertical_images.append(np.vstack(image_list_vertical))
#         combined_images = np.hstack(vertical_images)
#         # cv2.imshow("All CNN inputs", combined_images)
#     if k == ord('q') :
#         exit()

