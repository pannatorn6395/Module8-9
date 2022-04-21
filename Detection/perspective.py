import cv2
import numpy as np


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
def order_points(pts,dtype = 'float32'):
    rect = np.zeros((4,2),dtype=dtype)
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts,axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def expandPerspective_IMG_Matrix (img,pts,offset = 5):
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
    return cv2.warpPerspective(img,m,size),m






