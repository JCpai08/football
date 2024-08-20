import cv2
def read_video(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return frames
def save_video(video_path, frames):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out=cv2.VideoWriter(video_path, fourcc, 30, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()