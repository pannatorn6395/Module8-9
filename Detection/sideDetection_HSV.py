import cv2
import numpy as np
import json
from Detection.perspective import order_points



class Borad_Direction ():

    json_path = './Detection/setup_BoradDirection.json'
    windowName = 'Result'
    rotation = [cv2.ROTATE_90_COUNTERCLOCKWISE,cv2.ROTATE_180,cv2.ROTATE_90_CLOCKWISE]
    def __init__ (self,offset = 30,dist_gain = [3,2,3],thershold_value=100) :

        self.offset = offset
        self.dist_gain = dist_gain
        self.inputMouse = None
        self.trigger = False
        
        self.thershold_value = thershold_value
        self.Bound_HSV = np.array([[0,0,0],[0,0,0]])
        self.hsv_pixels = []
        self.hsv_mean = None
        self.hsv_sd = None
        self.load_jsonfile(exp=True)

    def convert_coord (self,m,pts):
        new_pts = np.zeros((4,2),dtype=int)
        for i in range(4):
            expand_pt = np.array([pts[i][0],pts[i][1],1])
            new_h = np.matmul(m,expand_pt)
            new_x = round(new_h[0]/new_h[2])
            new_y = round(new_h[1]/new_h[2])
            new_pts[i][0] = new_x
            new_pts[i][1] = new_y
        return order_points(new_pts,dtype=int)
        
    
    def rotate_borad(self,img,points,show=False):
        tl,tr,br,bl = points
        is_blackSide = 0
        max_area = 0
        corners = [ [tl[1]-self.offset,tl[1],tl[0],tr[0]],
                    [tr[1],br[1],tr[0],tr[0]+self.offset],
                    [bl[1],bl[1]+self.offset,bl[0],br[0]],
                    [tl[1],bl[1],tl[0]-self.offset,tl[0]]
                ]
        for i in range(4):
            crop = img[corners[i][0]:corners[i][1],corners[i][2]:corners[i][3]]
            mask = self.get_Mask(crop)
            area = np.count_nonzero(mask)
            if area > max_area:
                is_blackSide = i
                max_area = area
            if show :
                print(i,area)
                cv2.imshow('mask',mask)
                cv2.waitKey(0)
        if is_blackSide != 0 :
            img =  cv2.rotate(img,self.rotation[is_blackSide-1])
        return img
    
    
    def get_Mask(self,img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.Bound_HSV[0], self.Bound_HSV[1])
        return mask
    

    def tune_HSVBound_withNewPixel(self):
        if self.hsv_pixels != [] :
            hsv_pixels = np.array(self.hsv_pixels)
            self.hsv_mean = np.mean(hsv_pixels, axis=0)
            self.hsv_sd = np.std(hsv_pixels, axis=0)
            for i in range(3):
                offset = self.hsv_sd[i]*self.dist_gain[i]
                self.Bound_HSV[0][i] = round(self.hsv_mean[i] - offset )
                self.Bound_HSV[1][i] = round(self.hsv_mean[i] + offset )  
            print("HSV Bound : {}".format(self.Bound_HSV))

    def tune_HSVBound(self):
        if self.hsv_sd is not None :
            for i in range(3):
                offset = self.hsv_sd[i]*self.dist_gain[i]
                self.Bound_HSV[0][i] = round(self.hsv_mean[i] - offset )
                self.Bound_HSV[1][i] = round(self.hsv_mean[i] + offset )  
            print("HSV Bound : {}".format(self.Bound_HSV))

    def sampleHSV_withClick (self,img):
        winName = 'Pick The Color'
        cv2.namedWindow(winName)
        cv2.setMouseCallback(winName, self.mouseEvent)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imshow(winName,img)
        count = 0
        while True:
            key = cv2.waitKey(10) 
            if key == ord('q'):
                break
            else :
                if self.inputMouse != None:
                    if self.inputMouse[1] <= hsv.shape[0] and self.inputMouse[0] <= hsv.shape[1] :
                        print("Add HSV {} | {}".format(hsv[self.inputMouse[1],self.inputMouse[0]],count))
                        self.hsv_pixels.append(hsv[self.inputMouse[1],self.inputMouse[0]].tolist())
                        count +=1
                    else :
                        print("Cann't add HSV at {}".format(self.inputMouse))
                    self.inputMouse = None      
        cv2.destroyWindow(winName)

    def load_jsonfile (self,exp=False):
        try :
            f = open(self.json_path)
            data = json.load(f)
            f.close()
            self.Bound_HSV = np.array(data['boundHSV'])
            if exp == False:
                self.hsv_pixels = data['pixels']
            self.hsv_mean = data['mean']
            self.hsv_sd = data['sd']
            self.dist_gain = data['gain']
            print("Load Json File : Done")
        except :
            print("Load Json File : Fail")

    def save_jsonfile (self):
        try :
            data = {'boundHSV':self.Bound_HSV.tolist(),
                    'mean':self.hsv_mean.tolist(),
                    'sd': self.hsv_sd.tolist(),
                    'gain':self.dist_gain,
                    'pixels':self.hsv_pixels
                    }
            with open(self.json_path, "w") as f:
                json.dump(data, f)
            print("Save Json File : Done")
        except Exception as txt:
            print(txt)
            print("Save Json File : Fail")
    
    def mouseEvent(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP :
            self.inputMouse = [x,y]
            print("Click {}".format(self.inputMouse))

    

    # def hueGain_callback(self, val):
    #     self.dist_gain[0] = val
    #     self.trigger = True
    
    # def satGain_callback(self, val):
    #     self.dist_gain[1] = val
    #     self.trigger = True
    
    # def valGain_callback(self, val):
    #     self.dist_gain[2] = val
    #     self.trigger = True
    
    # def trackBarSetup(self):
    #     cv2.createTrackbar('hue gain', self.windowName, 1, 10, self.hueGain_callback)
    #     cv2.setTrackbarPos('hue gain', self.windowName, self.dist_gain[0])

    #     cv2.createTrackbar('sat gain', self.windowName, 1, 10, self.satGain_callback)
    #     cv2.setTrackbarPos('sat gain', self.windowName, self.dist_gain[1])

    #     cv2.createTrackbar('val gain', self.windowName, 1, 10, self.valGain_callback)
    #     cv2.setTrackbarPos('val gain', self.windowName, self.dist_gain[0])

    # def run(self,path):
    #     cv2.namedWindow(self.windowName)
    #     self.trackBarSetup()
    #     img_names = os.listdir(path)
    #     for name in img_names:
    #         img = cv2.imread(path+name)
    #         img = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2,))
    #         mask = self.get_Mask(img)
    #         result = cv2.bitwise_and(img,img,mask = mask)
    #         while True:
    #             cv2.imshow(self.windowName,result)
    #             cv2.imshow("IMG",img)
    #             key = cv2.waitKey(10)
    #             if self.trigger :
    #                 self.tune_HSVBound()
    #                 mask = self.get_Mask(img)
    #                 result = cv2.bitwise_and(img,img,mask = mask)
    #                 self.trigger = False
    #             if key == ord('/') or key == ord('q'):
    #                 # next img
    #                 break
    #             elif key == ord('p'):
    #                 self.sampleHSV_withClick(img)
    #                 self.tune_HSVBound_withNewPixel()
    #             elif key == ord('s'):
    #                 self.save_jsonfile(self.json_path,self.json_name)
    #         if key == ord('q') :
    #             break
            
            

class SidePiece_Detection ():
    
    json_path = './Detection/setup_PieceSide.json'
    def __init__ (self,colors='gold') :

        self.colors_name = colors
        self.offsetRatio = None
        self.dist_gain = None
        self.inputMouse = None
        self.trigger = False

        self.Bound_HSV = np.array([[0,0,0],[0,0,0]])
        self.hsv_mean = None
        self.hsv_sd = None
        self.is_red = False
        self.thes_area = 500

        self.thershold_value = None
        self.find_jsonfile = False
        self.load_jsonfile()

    
    def pieceSide_check (self,img,show = False):
        if self.colors_name != 0 :
            mask = self.get_Mask(img,show = show)
            area = np.count_nonzero(mask)
            return True if self.thes_area < area else False
        else :
            mask = self.get_Mask(img,is_color=False,show = show)
            area = np.count_nonzero(mask < self.thershold_value)
            return True if self.thes_area < area else False               
        
    def change_HSVBound_withDist_gain(self):
        if self.hsv_sd is not None :
            for i in range(3):
                offset = self.hsv_sd[i]*self.dist_gain[i]
                self.Bound_HSV[0][i] = round(self.hsv_mean[i] - offset )
                self.Bound_HSV[1][i] = round(self.hsv_mean[i] + offset )  
            print("HSV Bound : {}".format(self.Bound_HSV))
        self.update_jsonfile()

    def get_Mask(self,img,is_color = True,show = False):
        if is_color :
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.Bound_HSV[0], self.Bound_HSV[1])
        else :
            mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if show :
            cv2.imshow("Mask",mask)
        return mask


    def sampleHSV_withClick (self,img):
        winName = 'Pick The Color'
        hsv_pixels = []
        cv2.namedWindow(winName)
        cv2.setMouseCallback(winName, self.mouseEvent)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imshow(winName,img)
        count = 0
        while True:
            key = cv2.waitKey(10) 
            if key == ord('q'):
                break
            elif key == ord('b'):
                if hsv_pixels != []:
                    print("del {} :  HSV {}".format(count,hsv_pixels[-1]))
                    count -= 1
                    hsv_pixels.pop(-1)
            else :
                if self.inputMouse != None:
                    if self.inputMouse[1] <= hsv.shape[0] and self.inputMouse[0] <= hsv.shape[1] :
                        pixel = hsv[self.inputMouse[1],self.inputMouse[0]]
                        print("Add HSV {} | {}".format(pixel,count))
                        hsv_pixels.append(pixel.tolist())
                        color_show = np.zeros((100,100,3),dtype=np.uint8)
                        color_show[:,:] = pixel
                        color_show = cv2.cvtColor(color_show,cv2.COLOR_HSV2BGR)
                        cv2.imshow('color',color_show)
                        count +=1
                    else :
                        print("Cann't add HSV at {}".format(self.inputMouse))
                    self.inputMouse = None
        print(self.dist_gain)
        if hsv_pixels is not [] :
            hsv_pixels = np.array(hsv_pixels)
            self.hsv_mean = np.mean(hsv_pixels, axis=0)
            self.hsv_sd = np.std(hsv_pixels, axis=0)
            print(self.hsv_sd)
            for i in range(3):
                offset = self.hsv_sd[i]*self.dist_gain[i]
                self.Bound_HSV[0][i] = round(self.hsv_mean[i] - offset )
                self.Bound_HSV[1][i] = round(self.hsv_mean[i] + offset )  
            print("HSV Bound : {}".format(self.Bound_HSV))     
        cv2.destroyWindow(winName)
    
    def mouseEvent(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP :
            self.inputMouse = [x,y]
            self.trigger = True
            print("Click {}".format(self.inputMouse))

    def load_jsonfile (self):
        try :
            f = open(self.json_path)
            data = json.load(f)
            f.close()

            self.dist_gain = data['dist_gain']
            self.thershold_value = data['thershold_value']
            self.offsetRatio = data['offsetRatio']

            if self.colors_name != 0 :
                if self.colors_name in data['colors'].keys() :
                    self.Bound_HSV = np.array(data['colors'][self.colors_name]['boundHSV'])
                    self.hsv_mean = data['colors'][self.colors_name]['mean']
                    self.hsv_sd = data['colors'][self.colors_name]['sd']

            self.find_jsonfile = True
            print("Load Json File : Done")
        except Exception as txt:
            print(txt)
            self.find_jsonfile = False
            print("Save Json File : Fail")
    
    def update_jsonfile (self,is_color=True):
        try :
            f = open(self.json_path)
            alldata = json.load(f)
            f.close()
            alldata['thershold_value'] = self.thershold_value 
            alldata['offsetRatio'] = self.offsetRatio
            alldata['dist_gain'] = self.dist_gain
            if is_color != 0 :
                data = {'boundHSV':self.Bound_HSV.tolist(),
                            'mean':self.hsv_mean.tolist(),
                            'sd': self.hsv_sd.tolist()}
                alldata['colors'][self.colors_name] = data 
            with open(self.json_path, "w") as f:
                json.dump(alldata, f)
            print("Save Json File : Done")
        except Exception as txt:
            print(txt)
            print("Save Json File : Fail")
    
    # def tuingBoundHSV_withClick (self,img):
    #     hsv_pixels = []
    #     conner = []
    #     count = 1
    #     winName = 'Crop ' + self.colors_name
    #     cv2.namedWindow(winName)
    #     cv2.setMouseCallback(winName, self.mouseEvent)
    #     cv2.imshow(winName,img)
    #     while True:
    #         key = cv2.waitKey(10) 
    #         if key == ord('q'):
    #             break
    #         elif key == ord('b'):
    #             if hsv_pixels != []:
    #                 print("del {} : Median HSV {}".format(count,hsv_pixels[-1]))
    #                 count -= 1
    #                 hsv_pixels.pop(-1)
    #         else :
    #             if self.trigger :
    #                 self.trigger = False
    #                 if self.inputMouse[1] <= img.shape[0] and self.inputMouse[0] <= img.shape[1] :
    #                     conner.append(self.inputMouse)
    #                     if len(conner) == 2 :
    #                         crop = img[conner[0][1]:conner[1][1],conner[0][0]:conner[1][0]]
    #                         median = self.getMedian_HSV(crop)
    #                         hsv_pixels.append(median)
    #                         color_show = np.zeros((100,100,3),dtype=np.uint8)
    #                         color_show[:,:] = median
    #                         color_show = cv2.cvtColor(color_show,cv2.COLOR_HSV2BGR)
    #                         cv2.imshow('Crop Res',crop)
    #                         cv2.imshow('color',color_show)
    #                         print("{} : Median HSV {}".format(count,median))
    #                         count += 1
    #                         conner = []
    #                 else :
    #                     print("Out of range")
    #                 self.inputMouse = None
                    
    #     cv2.destroyAllWindows()
    #     if hsv_pixels != [] :
    #         hsv_pixels = np.array(hsv_pixels)
    #         self.hsv_mean = np.mean(hsv_pixels, axis=0).tolist()  
    #         self.hsv_sd = np.std(hsv_pixels, axis=0).tolist()  
    #         for i in range(3):
    #             offset = self.hsv_sd[i]*self.dist_gain[i]
    #             self.Bound_HSV[0][i] = round(self.hsv_mean[i] - offset)
    #             self.Bound_HSV[1][i] = round(self.hsv_mean[i] + offset)
    #         print("{} | {}".format(self.colors_name,self.Bound_HSV))
    #         self.update_jsonfile() 

    # def save_jsonfile (self):
    #     try :
    #         alldata = {}
    #         alldata['thershold_value'] = self.thershold_value 
    #         alldata['offsetRatio'] = self.offsetRatio
    #         alldata['dist_gain'] = self.dist_gain
    #         if self.colors_name != 0 :
    #             for index in range(2):
    #                 data = {'boundHSV':self.Bound_HSV[index],
    #                         'mean':self.hsv_mean[index],
    #                         'sd': self.hsv_sd[index],
    #                         }
    #                 alldata[self.colors_name[index]] = data
    #         with open(self.json_path+self.json_name, "w") as f:
    #             json.dump(alldata, f)
    #         print("Save Json File : Done")
    #     except Exception as txt:
    #         print(txt)
    #         print("Save Json File : Fail")
    

    # def tuingBoundHSV_withIMG (self,imgs,is_whiteSide = True, check_red = False):
    #     num = len(imgs)
    #     median_hsv = np.zeros((num,3),dtype=np.uint)
    #     for index in range(num) :
    #         self.getMedian_HSV(imgs[index])
    #         median_hsv[index,:] = self.getMedian_HSV(imgs[index])
    #     if check_red :
    #         if np.min(median_hsv[:,0]) < 30 and np.max(median_hsv[:,0]) >= 170 :
    #             self.is_red = True
    #     mean = np.mean(median_hsv, axis=0)
    #     sd = np.std(median_hsv, axis=0) 
    #     for i in range(3):
    #         self.Bound_HSV[0][i] = round(mean[i] - sd[i]*self.dist_gain[i])
    #         self.Bound_HSV[1][i] = round(mean[i] + sd[i]*self.dist_gain[i])
 
    #     if self.is_red :
    #         self.Bound_HSV[0][0] = 30
    #         self.Bound_HSV[1][0] = 170