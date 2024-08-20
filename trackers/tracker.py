from ultralytics import YOLO
import supervision as sv
import pickle
import cv2
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../')
#sys.path.append('../')：这行代码向sys.path列表中添加了一个新的条目，
# 即当前脚本所在目录的父目录。这意味着当你尝试导入一个模块时，
# Python也会在当前脚本的上级目录中查找这个模块。
from utils import get_bbox_cent,get_bbox_width,get_foot_position

class CTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        '''
        跟踪器内部维护了一系列跟踪对象，每个跟踪对象代表一个潜在的跟踪目标。这些对象包含了以下信息：
        位置和速度：当前和预测的位置，以及速度（或加速度）信息，用于卡尔曼滤波器的预测。
        跟踪ID：唯一标识符，用于区分不同的跟踪目标。
        连续未匹配计数：记录目标连续未被检测到的帧数，用于判断目标是否应该被删除。
        其他元数据：可能还包括目标的类别、大小等信息。
        '''
        
    def add_positon_to_track(self,tracks):
        for object,object_track in tracks.items():
            for frame_idx,frame_track in enumerate(object_track):
                for track_id,object_info in frame_track.items():
                    bbox = object_info['bbox']
                    if object == 'Ballon':
                        position = get_bbox_cent(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_idx][track_id]['position'] = position
    def interpolate_ball_position(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1 : {'bbox' : x}}for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions


    def detect_frames(self,frames):
        batch_size = 20
        detections = []
        '''
        分组预测是一种常见的优化技术，特别是在处理大量数据或受限于计算资源的情况下。
        它可以帮助更有效地利用计算资源，避免内存和计算瓶颈，同时确保结果的一致性和准确性。
        '''
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections+=detections_batch
            
        return detections
    def get_object_tracks(self,frames,read_from_stub = False,stub_path = None):

        if read_from_stub and  stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            'Arbitre':[],
            'Ballon':[],
            'Joueur':[]
        }
        #每一帧：{<tracker_id>:{bbox:[]},<tracker_id>:{bbox:[]}，，，}

        for frame_idx, detection in enumerate(detections):
            cls_names = detection.names
            # {0: 'Arbitre', 1: 'Ballon', 2: 'Gardien', 3: 'Joueur'}
            cls_names_inverted = {v: k for k, v in cls_names.items()}
            # {'Arbitre': 0, 'Ballon': 1, 'Gardien': 2, 'Joueur': 3}

            #convert detections to supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #convert Gardien to Joueur
            for obj_id,cls_id in enumerate(detection_supervision.class_id):
                if cls_names[cls_id] == 'Gardien':
                    detection_supervision.class_id[obj_id] = cls_names_inverted['Joueur']
            
            #track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            # 0:bbox 1:conf 2:mask(None) 3:cls_id 4:tracker_id 5:cls_name
            '''
            预测：使用卡尔曼滤波器预测每个跟踪对象在当前帧的位置。
            匹配：将预测的位置与当前帧的检测结果进行匹配，通常使用匈牙利算法来找到最优的匹配。
            更新：对于匹配成功的检测结果，更新相应跟踪对象的位置和状态；对于未匹配的检测结果，可能创建新的跟踪对象；对于连续未匹配的跟踪对象，可能将其标记为结束或删除。
            输出：返回带有跟踪ID的检测结果，这些结果可以用于后续的可视化或数据分析。
            '''
            #print(detection_with_tracks)
            #每一帧加一个{}
            tracks['Arbitre'].append({})
            tracks['Joueur'].append({})
            tracks['Ballon'].append({})

            for frame_detection in detection_with_tracks:
                #detection_with_tracks好像不能通过属性调用，需要通过索引
                bbox = frame_detection[0].tolist()
                tracker_id = frame_detection[4]
                cls_id = frame_detection[3]

                if cls_id == cls_names_inverted['Arbitre']:
                    tracks['Arbitre'][frame_idx][tracker_id] = {'bbox':bbox}
                
                if cls_id == cls_names_inverted['Joueur']:
                    tracks['Joueur'][frame_idx][tracker_id] = {'bbox':bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inverted['Ballon']:
                    tracks['Ballon'][frame_idx][1] = {'bbox':bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_triangle(self,frame,bbox,color):
        y1 = int(bbox[1])
        x_cent , _ = get_bbox_cent(bbox)

        point_array = np.array([
            [x_cent,y1],
            [x_cent+10,y1-20],
            [x_cent-10,y1-20]
        ])

        cv2.drawContours(frame,[point_array],0,color,cv2.FILLED)
        cv2.drawContours(frame,[point_array],0,(0,0,0),2)

        return frame
    def draw_ellipse(self,frame,bbox,color,track_id = None):
        y2 = bbox[3]
        x_cent , _ = get_bbox_cent(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(
                frame,
                center = (int(x_cent),int(y2)),
                axes = (int(width),int(width*0.35)),
                angle = 0.0,
                startAngle = -45,
                endAngle=235,
                color = color,
                thickness=2,
                lineType=cv2.LINE_4
            )
        rect_width = 40
        rect_height = 20
        x1_rect = x_cent - rect_width//2
        x2_rect = x_cent + rect_width//2
        y1_rect = y2 - rect_height//2 + 15
        y2_rect = y2 + rect_height//2 + 15
        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect),int(y1_rect)),
                (int(x2_rect),int(y2_rect)),
                color,
                cv2.FILLED)
            
            x1_text = x1_rect + 5
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,0,0),
                2
            )
        return frame
    def draw_possession(self,frame,frame_idx,team_ball_control):
        #Draw semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay,(1350,850),(1900,970 ),(255,255,255),cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)
        
        #calc possession
        team_ball_control_till_frame = team_ball_control[:frame_idx+1]
        team1_frame_num = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team2_frame_num = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team1 = team1_frame_num/(team1_frame_num+team2_frame_num)
        team2 = team2_frame_num/(team1_frame_num+team2_frame_num)
        
        cv2.putText(frame,f'Team 1 Possession:{team1*100:.2f}%',(1400,900),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        cv2.putText(frame,f'Team 2 Possession:{team2*100:.2f}%',(1400,950),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

        return frame

    def draw_annotations(self,vedio_frames,tracks,team_ball_control):
        output_frames = []
        for frame_idx,frame in enumerate(vedio_frames):
            frame = frame.copy()
            player_dict = tracks['Joueur'][frame_idx]
            referee_dict = tracks['Arbitre'][frame_idx]
            ball_dict = tracks['Ballon'][frame_idx]
            #Draw players
            ##player track id and team color
            for tracker_id,player_info in player_dict.items():
                bbox = player_info['bbox']
                color = player_info.get('team_color',(0,0,255))
            ##possession
                frame = self.draw_ellipse(frame,bbox,color,tracker_id)
                if player_info.get('has_ball',False):
                    frame = self.draw_triangle(frame,bbox,(0,0,255)) #RGB BGR
                
            #Draw referees
            for tracker_id,referee_info in referee_dict.items():
                bbox = referee_info['bbox']
                frame = self.draw_ellipse(frame,bbox,(0,255,255))
            
            #Draw ball
            for tracker_id,ball_info in ball_dict.items():
                bbox = ball_info['bbox']
                frame = self.draw_triangle(frame,bbox,(0,255,0))

            #Draw possession
            frame = self.draw_possession(frame,frame_idx,team_ball_control)

            output_frames.append(frame)
        return output_frames