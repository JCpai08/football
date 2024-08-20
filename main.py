from utils import read_video,save_video
from trackers import CTracker
#import cv2
import numpy as np
from team_assigner import CTeamAssigner
from player_ball_assigner import CPlayerBallAssigner
from camera_movement_estimator import CCameraMovementEstimator
def main():
    # #save a crop image of a player
    # for track_id,player_info in tracks['Joueur'][0].items():
    #     bbox = player_info['bbox']
    #     frame = video_frame[0]
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
    #     cv2.imwrite(f'output/cropped_image.jpg',cropped_image)
    #     break
    #read vedio
    video_frame = read_video('input/clip2.mp4')
    
    #initialize tracker
    tracker = CTracker('models/best.pt')
    #每一帧(tracks['Joueur'][0])
    # {<tracker_id>:{bbox:[]},<tracker_id>:{bbox:[]}，，，}
    tracks = tracker.get_object_tracks(video_frame,
                              read_from_stub=True,
                              stub_path = 'stubs/clip2_track_stubs.pkl')
    #get object position
    tracker.add_positon_to_track(tracks)

    #estimate camera movement
    camera_movement_estimator = CCameraMovementEstimator(video_frame[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frame,
        read_from_stub=True,
        stub_path = 'stubs/clip2_camera_movement_stubs.pkl'
        )
    camera_movement_estimator.add_adjust_position_to_tracks(tracks,camera_movement_per_frame)
    
    #interpolate ball
    #感觉这个插值有点多余，ByteTrack已经做了这个事情？
    tracks['Ballon'] = tracker.interpolate_ball_position(tracks['Ballon'])

    #assign player team
    team_assigner = CTeamAssigner()
    team_assigner.assign_team_color(
        video_frame[0],tracks['Joueur'][0]
        )
    for frame_idx,player_track in enumerate(tracks['Joueur']):
        for player_id,player_info in player_track.items():
            bbox = player_info['bbox']
            frame = video_frame[frame_idx]
            team_assigner.assign_player_team(frame,bbox,player_id)
            team = team_assigner.assign_player_team(frame,bbox,player_id)
            
            tracks['Joueur'][frame_idx][player_id]['team'] = team
            tracks['Joueur'][frame_idx][player_id]['team_color'] = team_assigner.team_colors[team]
    
    #assign ball possession to player
    ball_assigner = CPlayerBallAssigner()
    team_ball_control = []
    for frame_idx,players_track in enumerate(tracks['Joueur']):
        ball_bbox = tracks['Ballon'][frame_idx][1]['bbox']
        assign_player_id = ball_assigner.assign_ball_to_plaer(ball_bbox,players_track)
        
        if assign_player_id != -1:
            tracks['Joueur'][frame_idx][assign_player_id]['has_ball'] = True
            team_ball_control.append(tracks['Joueur'][frame_idx][assign_player_id]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)

    #draw output
    ##draw track
    output_vedio_frames = tracker.draw_annotations(video_frame,tracks,team_ball_control)

    ##draw camera movement
    output_vedio_frames = camera_movement_estimator.draw_camera_movement(output_vedio_frames,camere_movement_per_frame)

    save_video('output/clip2_output_video_track.mp4',output_vedio_frames)
    #save_video('output/output_video.mp4',video_frame)

    print('done')
if __name__ == '__main__':
    main()