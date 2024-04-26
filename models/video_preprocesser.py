import cv2

def extract_key_frames(video_path, methods="slow", gap=15):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    key_frames = []
    indexes = []

    index = 0
    while index < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        key_frames.append(frame)
        indexes.append(index)
        if not ret:
            break
        index += gap
    
    return key_frames, indexes

def extract_video_shots(video_path, shot_length=2):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    change_points = []
    n_frames_per_shot = int(shot_length * fps)
    index = 0
    while index < frame_count:
        change_points.append([index, index + n_frames_per_shot])
        index += n_frames_per_shot+1
    return change_points





    