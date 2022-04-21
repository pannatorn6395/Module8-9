import cv2
import base64
import io
import os
import math
import numpy as np
from Detection.perspective import *
from Detection.sideDetection_HSV import *
from itertools import count
import matplotlib.pyplot as plt
import DetectionFunctions as df
from DetectAllPoints import *
import PIL
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import chess
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
warnings.filterwarnings('ignore')
from keras.models import load_model
from keras.applications.imagenet_utils import decode_predictions
if __name__ == "__main__":
    State_check_chess=1
    chess_based="RNBQKBNR/PPPPPPPP/11111111/11111111/11111111/11111111/pppppppp/rnbqkbnr"
    check_based="00000000/00000000/11111111/11111111/11111111/11111111/00000000/00000000"
    moving=[]
    chess_based_index=[]
    check_based_index=[]
    position_index=[]
    direction = Borad_Direction()
    detection = SidePiece_Detection(colors='green')
    detection.colors_name='green'
    detection.load_jsonfile()
    detection.change_HSVBound_withDist_gain()
    vid = cv2.VideoCapture(0+cv2.CAP_DSHOW)
    vid.set(3,1920)
    vid.set(4,1080)
    count=0
    while(1):
        if State_check_chess==0:
            while True:
                    ret, frame = vid.read()
                    cv2.imshow('A',frame)
                    key=cv2.waitKey(1)
                    if key ==ord(' '):
                        cv2.imwrite('test/opencv_frame.jpg',frame)
                        break
            # Destroy all the windows
            cv2.destroyAllWindows()
            State_check_chess=1        
        elif State_check_chess==1:
            img = cv2.imread('test/1.jpg')
            # img = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2,))
            clear_image, encoded_image, matrix = getMatrixFromImage(img)
            img,points,all_point=show_point_on_image(img, matrix)
            W,m=expandPerspective_IMG_Matrix (img,(np.array(points,dtype='float32')),offset = 35)
            rect = direction.convert_coord(m,points)
            new_img = direction.rotate_borad(img=W,points=rect,show=False)
            Crop_labels(new_img)
            output=""
            columns=["a","b","c","d","e","f","g","h"]
            colors=[]
            model = load_model("VGG19_cam2_2class.h5")
            model2 = load_model("VGG19_cam2_6class.h5")
            result_fen=''
            fen_move=''
            for number in range(0,8):
                for alphabet in columns:
                    fen = classify_cells(model,[(alphabet+str(8-number)+".jpg")])
                    if fen=='0':
                        img = cv2.imread(f'Output/{(alphabet+str(8-number))}.jpg')
                        crop_img = img[35:116,35:115]
                        res = detection.pieceSide_check(img = crop_img,show = False)
                        print(res)
                        if res == 1 :
                            fen='2'
                    result_fen+=fen                
                result_fen+='/'
            print(result_fen)
            final_result_fen=""
            position=result_fen[:-1].split("/")
            check_based=check_based.split("/")
            chess_based=chess_based.split("/")
            for number in range(0,8):
                for alphabet in range(len(columns)):
                    # img_name = columns[alphabet]+"{}.jpg".format(8-number)
                    # crop_img=cv2.imread(f"dataset/Output/{img_name}")
                    if str(position[number] [alphabet]) != str(check_based[number][alphabet]) :
                        if position[number] [alphabet] =='1':
                            moving.append(chess_based[number] [alphabet])
                            check_based_index.append(number)
                            check_based_index.append(alphabet)
            print(moving)
            if len(moving) !=0:
                for number in range(0,8):
                    for alphabet in range(len(columns)):
                        # img_name = columns[alphabet]+"{}.jpg".format(8-number)
                        # crop_img=cv2.imread(f"dataset/Output/{img_name}")
                        # print(chess_based)
                        if str(position[number] [alphabet]) != check_based[number][alphabet] :
                            if position[number] [alphabet] =='0'and moving!=[]:
                                item=chess_based[number]
                                item_str=""
                                for i in range(len(item)):
                                    if i == alphabet:
                                        item_str+=moving[0]
                                    else:
                                        item_str+=item[i]
                                chess_based[number]=item_str
                                # chess_based[check_based_index[0]][check_based_index[1]]=='1'
                                item_str=""
                                for i in range(len(chess_based[check_based_index[0]])):
                                    if i ==  check_based_index[1] :
                                        item_str+='1'
                                    else:
                                        item_str+=chess_based[check_based_index[0]][i]
                                chess_based[check_based_index[0]]=item_str
                            elif position[number] [alphabet] =='2'and moving!=[]:
                                item=chess_based[number]
                                item_str=""
                                for i in range(len(item)):
                                    if i == alphabet:
                                        item_str+=moving[0]
                                    else:
                                        item_str+=item[i]
                                chess_based[number]=item_str
                                # chess_based[check_based_index[0]][check_based_index[1]]=='1'
                                item_str=""
                                for i in range(len(chess_based[check_based_index[0]])):
                                    if i ==  check_based_index[1] :
                                        item_str+='1'
                                    else:
                                        item_str+=chess_based[check_based_index[0]][i]
                                chess_based[check_based_index[0]]=item_str
            check_based='/'.join(position)
            final_result_fen='/'.join(chess_based)
            chess_based='/'.join(chess_based)
            final_result_fen=final_result_fen.replace('11','2').replace('111','3').replace('11111','5').replace('111111','6').replace('1111111','7')
            final_result_fen=final_result_fen.replace('22','4')
            final_result_fen=final_result_fen.replace('44','8')
            final_result_fen=final_result_fen.replace('41','5')
            final_result_fen=final_result_fen.replace('21','3')
            final_result_fen=final_result_fen.replace('43','7')
            final_result_fen=final_result_fen.replace('42','6')
            moving=[]
            check_based_index=[]
            print(final_result_fen)
            board=chess.Board(final_result_fen)
            print(board)
            State_check_chess=0
            count+=1

