import pickle
import numpy as np
import cv2
import os
import sys
sys.path.append('../')
from utils import measure_distance

class CCameraMovementEstimator:
    def __init__(self,frame):

        self.minimum_distance = 5

        self.lk_pramas = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        ) 
        
        # 这段代码定义了一个类的属性lk_pramas，它是一个字典，包含了光流法（Lucas-Kanade算法）的一些参数设置：
        # winSize：搜索窗口的大小，设置为(15,15)。
        # maxLevel：金字塔的最大层级，设置为2。
        # criteria：终止条件，包括EPS（误差平方和）、COUNT（迭代次数）和阈值，设置为(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)。
        

        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1
        mask_features[:,900: 1500] = 1
    
        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 7,
            blockSize = 7,
            mask = mask_features
        )
    
    def add_adjust_position_to_tracks(self,tracks,camera_movement_per_frame,frame):
        for object,object_track in tracks.items():
            for frame_idx,frame_track in enumerate(object_track):
                for track_id,object_info in frame_track.items():
                    position = object_info['position']
                    camera_movement = camera_movement_per_frame[frame_idx]
                    position_adjust = (position[0]-camera_movement[0],
                                       position[1]-camera_movement[1])
                    tracks[object][frame_idx][track_id]['position_adjusted'] = position_adjust
    def get_camera_movement(self,frames,read_from_stub = False,stub_path = None):
        #Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        camera_movement = [[0,0]]*len(frames)
        
        old_gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)
        #单星号 * 用于解包元组或列表作为位置参数，而双星号 ** 用于解包字典作为关键字参数。
        #为了正确地将字典内容作为关键字参数传递，确保字典的键精确对应于函数期望接收的关键字参数名。

        for frame_idx in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_idx],cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_pramas)
            #光流法，完全不会先不管

            max_distance = 0
            camera_movement_x,camera_movement_y = 0,0

            for i,(new,old) in enumerate(zip(new_features,old_features)):
                new_feature_point = new.ravel()
                old_feature_point = old.ravel()

                distance = measure_distance(new_feature_point,old_feature_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x = new_feature_point[0] - old_feature_point[0]
                    camera_movement_y = new_feature_point[1] - old_feature_point[1]
                    
            if max_distance > self.minimum_distance:
                camera_movement[frame_idx] = [camera_movement_x,camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray,**self.features)
                #移动的情况下需要新的特征点    
            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement            

    def draw_camera_movement(self,frames,camera_movement):
        output_frames = []

        for frame_idx,frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha = 0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement,y_movement = camera_movement[frame_idx]
            frmae = cv2.putText(frame,f'Camera Movement X:{x_movement:.2f}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frmae = cv2.putText(frame,f'Camera Movement Y:{y_movement:.2f}',(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame)

        return output_frames
        