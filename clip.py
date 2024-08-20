'''
ultralytics 依赖 opencv pytorch
pytorch 要装cuda
opencv opencv-contrib-python
'''
from moviepy.editor import VideoFileClip

# 加载视频
video = VideoFileClip("input/4.mp4")

# 截取视频片段，例如从第10秒到第20秒
clip = video.subclip(95, 125)
clip.write_videofile("input/clip2.mp4")

# from ultralytics import YOLO

# model = YOLO('models/best.pt')
# results = model.predict("input/clip1.mp4",save = True)
# print (results[0])
# print("===================================")
# for box in results[0].boxes:
#     print(box)