from models.summary_generator import summary_generator
from models.video_preprocesser import extract_video_shots, extract_key_frames
from models.scoring import frame_scoring, shot_scoring
from models.features_extractor import load_clip_model, extract_query_features, extract_image_features

input_video_path = ''

output_video_path = ''

input_query = ''
print('Extracting video shots...')
video_change_points = extract_video_shots(input_video_path)
key_frames, key_frames_indexes = extract_key_frames(input_video_path)


print('Loading model...')
model, preprocess = load_clip_model()

print('Extracting query features...')
query_feature = extract_query_features(model, input_query)

print('Extracting image features...')
image_features = extract_image_features(model, preprocess, key_frames)

print('Scoring...')
frame_scores = frame_scoring(query_feature, image_features)
shot_scores = shot_scoring(frame_scores, video_change_points)

print('Generating summary...')
summary_generator(input_video_path, shot_scores, video_change_points, output_video_path)