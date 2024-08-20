from sklearn.cluster import KMeans

class CTeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}#<player_id>:<team_id>
    def get_cluerstering_model(self,image):
        image_2d = image.reshape(image.shape[0]*image.shape[1],3)#自动是(-1,3)
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
        kmeans.fit(image_2d)

        return kmeans
    
    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        top_half_image = image[:image.shape[0]//2,:]

        #Get clustering Model
        kmeans = self.get_cluerstering_model(top_half_image)

        #Get Player Cluster
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        conners_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = sum(conners_clusters)//3
        player_cluster = 1 -  non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self,frame,player_detections):
        player_color = []
        for _,player_detection in player_detections.items():
            bbox = player_detection['bbox']
            color = self.get_player_color(frame,bbox)
            player_color.append(color)

        #print(player_color)
        
        #assign team according to color
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
        kmeans.fit(player_color)
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        self.kmeans = kmeans
        
    def assign_player_team(self,frame,player_bbox,player_id):

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame,player_bbox)
        #pridect可以输入一个数组，返回一个数组
        player_team = self.kmeans.predict([player_color])[0] + 1
        
        self.player_team_dict[player_id] = player_team

        return player_team