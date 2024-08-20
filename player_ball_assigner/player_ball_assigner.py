import sys
sys.path.append("../")

from utils.bbox_utils import get_bbox_cent,measure_distance

class CPlayerBallAssigner:
    def __init__(self):
        self.max_distance = 100

    def assign_ball_to_plaer(self,ball_bbox,players):
        
        ball_cent = get_bbox_cent(ball_bbox)
        minn = 9999
        assign_player_id = -1

        for player_id,player_info in players.items():
            player_bbox= player_info['bbox']

            left_distance = measure_distance(ball_cent,(player_bbox[0],player_bbox[-1]))
            right_distance = measure_distance(ball_cent,(player_bbox[2],player_bbox[-1]))
            distance = min(left_distance,right_distance)

            if distance < self.max_distance:
                if distance < minn:
                    assign_player_id = player_id
                    minn = distance
        #print(assign_player_id)
        return assign_player_id