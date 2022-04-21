# Module built to get a metrix containing oll the points from the chess board
# getMatrixFromImage receive a image file in input (if null takes a picture from the camera)
# It returns a colored image representing the image passed or taken and the matrix representing
# the points of the chess board

# from PIL import Image
from itertools import count
import PIL.Image as Image
import cv2
import base64
import io
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import math
import os
import DetectionFunctions as df
import PIL
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import chess
import warnings

np.set_printoptions(suppress=True, linewidth=200)  # Better printing of arrays
plt.rcParams['image.cmap'] = 'jet'  # Default colormap is jet

# return the image and the matrix

def image_scale(pts, scale):
	"""scale to original image size"""
	def __loop(x, y): return [x[0] * y, x[1] * y]
	return list(map(functools.partial(__loop, y=1/scale), pts))

def image_resize(img, height=500):
	"""resize image to same normalized area (height**2)"""
	pixels = height * height; shape = list(np.shape(img))
	scale = math.sqrt(float(pixels)/float(shape[0]*shape[1]))
	shape[0] *= scale; shape[1] *= scale
	img = cv2.resize(img, (int(shape[1]), int(shape[0])))
	img_shape = np.shape(img)
	return img, img_shape, scale
# Function : expandPerspective_IMG
# Input
#   pts : 4 points in original image (numpy's object)  
#   offset : how many do u want to expand Perspective image (int)
# Output
#   Perspective image
def expandPerspective_IMG (img,pts,offset = 5):
    m,size = get_MatrixTransform (pts)
    inverse_m = np.linalg.inv(m)
    offset_pts = [  [-offset,-offset,1] ,       [size[0]+offset,-offset,1],
                    [-offset,size[1]+offset,1], [size[0]+offset,size[1]+offset,1]]
    new_pts = np.zeros((4,2),dtype='float32')
    for i in range (4):
        new_h = np.matmul(inverse_m,np.array(offset_pts[i]))
        new_pts[i][0] = round(new_h[0]/new_h[2])
        new_pts[i][1] = round(new_h[1]/new_h[2])
    m,size = get_MatrixTransform (new_pts) 
    return cv2.warpPerspective(img,m,size)

# Function : expandPerspective_points
# Input
#   pts : 4 points in original image (numpy's object) 
#   new_pts : 4 points in first perspective image (numpy's object) 
# Output
#   new_pts : new 4 points (numpy's object)
def expandPerspective_points (pts,new_pts):
    m,size = get_MatrixTransform (pts)
    inverse_m = np.linalg.inv(m)
    for i in range (4):
        new_h = np.matmul(inverse_m,np.array([new_pts[i][0],new_pts[i][1],1]))
        new_pts[i][0] = round(new_h[0]/new_h[2])
        new_pts[i][1] = round(new_h[1]/new_h[2])
    return new_pts

# Function : get_MatrixTransform 
# Input
#   pts : 4 points (numpy's object) 
# Output
#   m : homogenous 3x3 (numpy's object)
def get_MatrixTransform (pts):
    rect = order_points(pts)
    (tl,tr,br,bl) = rect
    widthA = np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2))
    widthB = np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))
    maxWidth = max(int(widthA),int(widthB))

    heightA = np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))
    heightB = np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))
    maxHeight = max(int(heightA),int(heightB))

    dst = np.array([[0,0],
                    [maxWidth-1,0],
                    [maxWidth-1,maxHeight-1],
                    [0,maxHeight-1]],dtype='float32')
    return cv2.getPerspectiveTransform(rect,dst),(maxWidth,maxHeight)

# Function : order_points 
# Input
#   pts : 4 points (numpy's object) 
# Output
#   new_pts : ordered 4 points (numpy's object)
def order_points(pts):
    rect = np.zeros((4,2),dtype='float32')
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts,axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
def image_transform(img, points, square_length=150):
	"""crop original image using perspective warp"""
	board_length = square_length * 8
	def __dis(a, b): return np.linalg.norm(na(a)-na(b))
	def __shi(seq, n=0): return seq[-(n % len(seq)):] + seq[:-(n % len(seq))]
	best_idx, best_val = 0, 10**6
	for idx, val in enumerate(points):
		val = __dis(val, [0, 0])
		if val < best_val:
			best_idx, best_val = idx, val
	pts1 = np.float32(__shi(points, 4 - best_idx))
	pts2 = np.float32([[0, 0], [board_length, 0], \
			[board_length, board_length], [0, board_length]])
	W=expandPerspective_IMG (img,np.array(points),offset = 20)
	#M = cv2.getPerspectiveTransform(pts1, pts2)
	#W = cv2.warpPerspective(img, M, (board_length, board_length))
	return W

def getMatrixFromImage(img):

    # img = Image.open(io.BytesIO(bytes(byteArray)))
    img = np.array(img)
    img_orig = Image.fromarray(img)
    img_width, img_height = img_orig.size

    # Resize
    # aspect_ratio = min(500.0/img_width, 500.0/img_height)
    # new_width, new_height = (
    #     (np.array(img_orig.size) * aspect_ratio)).astype(int)
    # img = img_orig.resize((new_width, new_height), resample=Image.BILINEAR)
    img = img_orig
    img_rgb = img
    img = img.convert('L')  # grayscale
    img = np.array(img)
    img_rgb = np.array(img_rgb)
    # cv2.imshow('test', img_rgb)
    # cv2.waitKey(0)
    M, ideal_grid, grid_next, grid_good, spts = df.findChessboard(img)

    # xy-unwarp -> the inner points of the inner chessboard
    # board_outline -> the corners (they are five because the first one is repeated)
    # boarder_points_?? -> the edges (?? edge of board: boarder_points_01 = edge from corner 0 to 1)

    # View
    if M is not None:
        # generate mapping for warping image
        M, _ = df.generateNewBestFit((ideal_grid+8)*32, grid_next, grid_good)
        img_warp = cv2.warpPerspective(
            img, M, (17*32, 17*32), flags=cv2.WARP_INVERSE_MAP)

        best_lines_x, best_lines_y = df.getBestLines(img_warp)
        xy_unwarp = df.getUnwarpedPoints(best_lines_x, best_lines_y, M)
        board_outline_unwarp = df.getBoardOutline(
            best_lines_x, best_lines_y, M)

        borders_points_01 = []
        borders_points_12 = []
        borders_points_23 = []
        borders_points_30 = []
        for i in range(0, len(xy_unwarp)):
            if i % 7 == 0:
                a, b = df.slope_intercept(
                    xy_unwarp[i, 0], xy_unwarp[i, 1], xy_unwarp[i+1, 0], xy_unwarp[i+1, 1])
                x_30, y_30 = df.line_intersection(([np.float32(0), b], [xy_unwarp[i, 0], xy_unwarp[i, 1]]), ([
                                                  board_outline_unwarp[3, 0], board_outline_unwarp[3, 1]], [board_outline_unwarp[0, 0], board_outline_unwarp[0, 1]]))
                x_12, y_12 = df.line_intersection(([-b/a, np.float32(0)], [xy_unwarp[i, 0], xy_unwarp[i, 1]]), ([
                                                  board_outline_unwarp[1, 0], board_outline_unwarp[1, 1]], [board_outline_unwarp[2, 0], board_outline_unwarp[2, 1]]))
                borders_points_30.append([x_30, y_30])
                borders_points_12.append([x_12, y_12])

            if i in range(0, 7):
                a, b = df.slope_intercept(
                    xy_unwarp[i, 0], xy_unwarp[i, 1], xy_unwarp[i+7, 0], xy_unwarp[i+7, 1])
                x_01, y_01 = df.line_intersection(([np.float32(0), b], [xy_unwarp[i, 0], xy_unwarp[i, 1]]), ([
                                                  board_outline_unwarp[0, 0], board_outline_unwarp[0, 1]], [board_outline_unwarp[1, 0], board_outline_unwarp[1, 1]]))
                x_23, y_23 = df.line_intersection(([-b/a, np.float32(0)], [xy_unwarp[i, 0], xy_unwarp[i, 1]]), ([
                                                  board_outline_unwarp[2, 0], board_outline_unwarp[2, 1]], [board_outline_unwarp[3, 0], board_outline_unwarp[3, 1]]))
                borders_points_01.append([x_01, y_01])
                borders_points_23.append([x_23, y_23])

        first_line = np.concatenate(([board_outline_unwarp[0]], borders_points_01, [
                                    board_outline_unwarp[1]]), axis=0)
        last_line = np.concatenate(([board_outline_unwarp[3]], borders_points_23, [
                                   board_outline_unwarp[2]]), axis=0)
        inner_lines = df.chunks(xy_unwarp, 7)
        for i in range(0, len(borders_points_12)):
            inner_lines[i] = np.concatenate(
                ([borders_points_30[i]], inner_lines[i], [borders_points_12[i]]), axis=0)

        matrix = np.vstack(([first_line], inner_lines, [last_line]))
        clear_image = img_rgb.copy()
        # uncomment to see points on the image
        df.color_points(img_rgb, matrix)
        img_rgb = Image.fromarray(img_rgb)

        img_rgb = img_rgb.resize(
            (img_width, img_height), resample=Image.BILINEAR)

        byte_array = io.BytesIO()
        img_rgb.save(byte_array, format='JPEG')
        encoded_image = base64.encodebytes(
            byte_array.getvalue()).decode('ascii')
        # cv2.imshow("ImageRGB", img_rgb)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        return clear_image, encoded_image, matrix

    else:
        # cv2.imshow("Image", img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        return None, None, None

def show_point_on_image(img, matrixOfPoints):
    # print(len(matrixOfPoints))
    points=[]
    all_point=[]
    for i in range(0, len(matrixOfPoints)):
        for j in range(0, len(matrixOfPoints[i])):
            # print(i, j, matrixOfPoints[i, j])
            # print(matrixOfPoints[i, j, 0], matrixOfPoints[i, j, 1])
            #img = cv2.circle(img, (math.floor(matrixOfPoints[i, j, 0]), math.floor(matrixOfPoints[i, j, 1])), radius=2, color=(
            #    255, 0, 0), thickness=-1)
            #cv2.putText(
            #   img, f"{i*9+j}", (math.floor(matrixOfPoints[i, j, 0]), math.floor(matrixOfPoints[i, j, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            all_point.append([math.floor(matrixOfPoints[i, j, 0]), math.floor(matrixOfPoints[i, j, 1])])
            #cv2.imshow('point on image', img)
            #cv2.waitKey(1000)
            if i*9+j == 0 or i*9+j == 8 or i*9+j == 72 or  i*9+j == 80:
            #    cv2.putText(
            #    img, f"{(math.floor(matrixOfPoints[i, j, 0]))},{math.floor(matrixOfPoints[i, j, 1])}", (math.floor(matrixOfPoints[i, j, 0]), math.floor(matrixOfPoints[i, j, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)    
                points.append((math.floor(matrixOfPoints[i, j, 0]), math.floor(matrixOfPoints[i, j, 1])))
    return img,points,all_point

def Crop_labels(new_img):
    labels= ["a","b","c","d","e","f","g","h"]
    try:
    # print(bytearray(img))
        clear_image, encoded_image, matrix = getMatrixFromImage(new_img)
        img,points,all_point=show_point_on_image(new_img, matrix)
        print(len(all_point))
        clusters = sorted(list(all_point), key=lambda k: k[0])
        sorted_points = []
        j = 0
        for i in range(1,10):
                points = sorted(clusters[j:i*9], key=lambda k: k[1])
                sorted_points += points
                j+=9
        for i in range(8):
            for j in range(8):
                try:
                    cropped__img = img[int((sorted_points[(9*i)+j+9][1])-32):int((sorted_points[(9*i)+j+10][1])+32), int((sorted_points[(9*i)+j][0])-32):int((sorted_points[(9*i)+j+10][0])+32)]
                    img_name = labels[i]+"{}.jpg".format(8-j)
                    cv2.imwrite(f"Output/{img_name}", cropped__img)
                    # cv2.imwrite(f"dataset/{name}/crop/{img_name}", cropped__img)
                        #print("{} written!".format(file+img_name))
                except:
                    print("{} error!".format(img_name))
                        #print("{}".format(int(all_point[(9*i)+j+9][1]),int(all_point[(9*i)+j+10][1])))
                        #print("{}".format(int(all_point[(9*i)+j][0]),int(all_point[(9*i)+j+10][0])))
    except:
        print("error!")
def convert_image_to_bgr_numpy_array(image_path, size=(224, 224)):
    image = PIL.Image.open(image_path).resize(size)
    img_data = np.array(image.getdata(), np.float32).reshape(*size, -1)
    # swap R and B channels
    img_data = np.flip(img_data, axis=2)
    return img_data
def prepare_image(image_path):
    im = convert_image_to_bgr_numpy_array(image_path)

    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68

    im = np.expand_dims(im, axis=0)
    return im
def classify_cells(model, img_filename_list):
    category_reference = {0: 'chess', 1: 'empty'}
    pred_list = []
    dict=[]
    fen=""
    # print( img_filename_list)
    try:
        for filename in img_filename_list:
            img = prepare_image("Output/"+str(filename))
            out= model.predict(img)
            top_pred = np.argmax(out)
            pred = category_reference[top_pred]
            # print(pred,filename)
            pred_list.append(pred)
            fen="0" if pred=="chess" else "1" if  pred=="empty" else None
    except:
        pass
    return fen
def classify_cells2(model, img_filename_list):
    category_reference = {0: 'bishop', 1: 'king', 2: 'knight', 3: 'pawn', 4: 'queen', 5: 'rook'}
    pred_list = []
    dict=[]
    fen=""
    # print( img_filename_list)
    try:
        for filename in img_filename_list:
            img = prepare_image("Output/"+str(filename))
            out = model.predict(img)
            top_pred = np.argmax(out)
            pred = category_reference[top_pred]
            # print(pred,filename)
            pred_list.append(pred)
            fen="b" if pred=="bishop"  else "k" if pred=="king" else "n" if pred=="knight" else "p" if pred=="pawn" else "q" if pred=="queen" else "r" if pred=="rook" else None
    except:
        pass
    return fen


if __name__ == "__main__":
    dir=os.listdir('/Users/pannatorn/Desktop/chessboard/raw_dataset')
    print(dir)
    count_error=0
    error=[]
    for name in dir :
        file = f'/Users/pannatorn/Desktop/chessboard/raw_dataset/{name}'
        try:
            print(name)
            img = cv2.imread(file)
    # print(bytearray(img))
            clear_image, encoded_image, matrix = getMatrixFromImage(img)
            img,points,all_point=show_point_on_image(img, matrix)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
            W=expandPerspective_IMG (img,(np.array(points,dtype='float32')),offset = 35)
            cv2.imwrite(f"/Users/pannatorn/Desktop/chessboard/result/{name}",W)
        except:
            print(f"error_{name}")
            error.append(name)
            count_error+=1
    dir2=os.listdir('/Users/pannatorn/Desktop/chessboard/result')
    # dir2=["18_510.jpg"]
    count_crop=29774
    count_eror2=0
    for name in dir2 :
        file = f'/Users/pannatorn/Desktop/chessboard/result/{name}'
        try:
            print(name)
            img = cv2.imread(file)
    # print(bytearray(img))
            clear_image, encoded_image, matrix = getMatrixFromImage(img)
            img,points,all_point=show_point_on_image(img, matrix)
            print(len(all_point))
            print("11")

            clusters = sorted(list(all_point), key=lambda k: k[0])
            sorted_points = []
            j = 0
            for i in range(1,10):
                points = sorted(clusters[j:i*9], key=lambda k: k[1])
                sorted_points += points
                j+=9
            for i in range(8):
                for j in range(8):
                    try:
                        cropped__img = img[int((sorted_points[(9*i)+j+9][1])-32):int((sorted_points[(9*i)+j+10][1])+32), int((sorted_points[(9*i)+j][0])-32):int((sorted_points[(9*i)+j+10][0])+32)]
                        img_name = f"{count_crop}.jpg".format(8-j)
                        cv2.imwrite(f'/Users/pannatorn/Desktop/chessboard/chess_crop/All_data/{img_name}', cropped__img)
                        #print("{} written!".format(file+img_name))
                        count_crop+=1
                    except:
                        count_eror2+=1
                        print("{} error!".format(img_name))
                        #print("{}".format(int(all_point[(9*i)+j+9][1]),int(all_point[(9*i)+j+10][1])))
                        #print("{}".format(int(all_point[(9*i)+j][0]),int(all_point[(9*i)+j+10][0])))
        except:
            print(f"error_{name}")

        #cv2.imshow('point on image', W)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    print(matrix.size)
    print(f"total_error = {count_error}")
    print(f"error_list  = {error}")
    print(f"error_2 = {count_eror2}")

