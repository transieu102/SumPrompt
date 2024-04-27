from models.summary_generator import summary_generator
from models.scoring import frame_scoring, shot_scoring
from models.features_extractor import load_clip_model, extract_query_features, extract_image_features
import h5py 
import cv2
import numpy as np
import hdf5storage
from evals.evalate_h5 import evaluate_summary
file_name_to_h5 = {'0tmA_C6XwfM.mp4': 'video_13', 'EE-bNr36nyA.mp4': 'video_38', 'JgHubY5Vw3Y.mp4': 'video_44', 'VuWGsYPqAX8.mp4': 'video_31', '37rzWOQsNIw.mp4': 'video_19', 'eQu1rNs0an0.mp4': 'video_43', 'JKpqYvAdIsw.mp4': 'video_32', '3eYKfiOEJNs.mp4': 'video_14', '-esJrBWj2d8.mp4': 'video_50', 'kLxoNp-UchI.mp4': 'video_48', 'WxtbjNsCQ8A.mp4': 'video_36', '4wU_LUjG5Ic.mp4': 'video_30', 'EYqVtI9YWJA.mp4': 'video_42', 'LRw_obCPUt0.mp4': 'video_20', 'XkqCExn6_Us.mp4': 'video_23', '91IHQYk1IQM.mp4': 'video_26', 'fWutDQy1nnY.mp4': 'video_29', 'NyBmCxDoHJU.mp4': 'video_47', 'xmEERLqJ2kU.mp4': 'video_33', '98MoyGZKHXc.mp4': 'video_2', 'GsAD1KT1xo8.mp4': 'video_24', 'oDXZc0tZe04.mp4': 'video_40', '_xMr-HKMfVA.mp4': 'video_35', 'akI8YFjEmUw.mp4': 'video_10', 'gzDbaEs1Rlg.mp4': 'video_4', 'PJrm840pAUI.mp4': 'video_25', 'xwqBXPGE9pQ.mp4': 'video_9', 'AwmHb44_ouw.mp4': 'video_1', 'Hl-__g2gn_A.mp4': 'video_17', 'qqR6AEXwxoQ.mp4': 'video_41', 'xxdtq8mxegs.mp4': 'video_15', 'b626MiF1ew4.mp4': 'video_22', 'HT5vyqe0Xaw.mp4': 'video_6', 'RBCABdttQmI.mp4': 'video_27', 'XzYM3PfTM4w.mp4': 'video_5', 'Bhxk-O1Y7Ho.mp4': 'video_12', 'i3wAGJaaktw.mp4': 'video_11', 'Yi4Ij2NM7U4.mp4': 'video_18', 'byxOvuiIJV0.mp4': 'video_34', 'iVt07TCkFM0.mp4': 'video_45', 'sTEELN-vY30.mp4': 'video_7', 'z_6gVvQb2d0.mp4': 'video_28', 'cjibtmSLxQ4.mp4': 'video_21', 'J0nA4VgnoCo.mp4': 'video_3', 'uGu_10sucQo.mp4': 'video_37', 'E11zDS9XGzg.mp4': 'video_46', 'jcoYJXDG9sw.mp4': 'video_49', 'vdmoEJ5YbrQ.mp4': 'video_8', 'WG0MBPpPC6I.mp4': 'video_16', 'Se3oxnaPsz0.mp4': 'video_39'}
h5_to_file_name = {'video_13': '0tmA_C6XwfM.mp4', 'video_38': 'EE-bNr36nyA.mp4', 'video_44': 'JgHubY5Vw3Y.mp4', 'video_31': 'VuWGsYPqAX8.mp4', 'video_19': '37rzWOQsNIw.mp4', 'video_43': 'eQu1rNs0an0.mp4', 'video_32': 'JKpqYvAdIsw.mp4', 'video_14': '3eYKfiOEJNs.mp4', 'video_50': '-esJrBWj2d8.mp4', 'video_48': 'kLxoNp-UchI.mp4', 'video_36': 'WxtbjNsCQ8A.mp4', 'video_30': '4wU_LUjG5Ic.mp4', 'video_42': 'EYqVtI9YWJA.mp4', 'video_20': 'LRw_obCPUt0.mp4', 'video_23': 'XkqCExn6_Us.mp4', 'video_26': '91IHQYk1IQM.mp4', 'video_29': 'fWutDQy1nnY.mp4', 'video_47': 'NyBmCxDoHJU.mp4', 'video_33': 'xmEERLqJ2kU.mp4', 'video_2': '98MoyGZKHXc.mp4', 'video_24': 'GsAD1KT1xo8.mp4', 'video_40': 'oDXZc0tZe04.mp4', 'video_35': '_xMr-HKMfVA.mp4', 'video_10': 'akI8YFjEmUw.mp4', 'video_4': 'gzDbaEs1Rlg.mp4', 'video_25': 'PJrm840pAUI.mp4', 'video_9': 'xwqBXPGE9pQ.mp4', 'video_1': 'AwmHb44_ouw.mp4', 'video_17': 'Hl-__g2gn_A.mp4', 'video_41': 'qqR6AEXwxoQ.mp4', 'video_15': 'xxdtq8mxegs.mp4', 'video_22': 'b626MiF1ew4.mp4', 'video_6': 'HT5vyqe0Xaw.mp4', 'video_27': 'RBCABdttQmI.mp4', 'video_5': 'XzYM3PfTM4w.mp4', 'video_12': 'Bhxk-O1Y7Ho.mp4', 'video_11': 'i3wAGJaaktw.mp4', 'video_18': 'Yi4Ij2NM7U4.mp4', 'video_34': 'byxOvuiIJV0.mp4', 'video_45': 'iVt07TCkFM0.mp4', 'video_7': 'sTEELN-vY30.mp4', 'video_28': 'z_6gVvQb2d0.mp4', 'video_21': 'cjibtmSLxQ4.mp4', 'video_3': 'J0nA4VgnoCo.mp4', 'video_37': 'uGu_10sucQo.mp4', 'video_46': 'E11zDS9XGzg.mp4', 'video_49': 'jcoYJXDG9sw.mp4', 'video_8': 'vdmoEJ5YbrQ.mp4', 'video_16': 'WG0MBPpPC6I.mp4', 'video_39': 'Se3oxnaPsz0.mp4'}

mat = hdf5storage.loadmat('/content/drive/MyDrive/VSUM/ydata-tvsum50.mat', variable_names=['tvsum50'])
mat = mat['tvsum50'].ravel()
data_mat = {}
for item in mat:
    video, category, title, length, nframes, user_anno, gt_score = item

    item_dict = {
    'video': video[0, 0],
    'category': category[0, 0],
    'title': title[0, 0],
    'length': length[0, 0],
    'nframes': nframes[0, 0],
    'user_anno': user_anno,
    'gt_score': gt_score
    }

    data_mat[file_name_to_h5[video[0, 0]+'.mp4']] = item_dict

h5_path = ''

f = h5py.File(h5_path, 'r')
data = {}
for name in f.keys():
    data[name] = {}
    data[name]['features'] = f[name + '/features'][()]
    data[name]['gtscore'] = f[name + '/gtscore'][()]
    data[name]['user_summary'] = f[name + '/user_summary'][()]
    data[name]['change_points'] = f[name + '/change_points'][()]
    data[name]['n_frame_per_seg'] = f[name + '/n_frame_per_seg'][()]
    data[name]['n_frames'] = f[name + '/n_frames'][()]
    data[name]['picks'] = f[name + '/picks'][()]
    data[name]['n_steps'] = f[name + '/n_steps'][()]
    data[name]['gtsummary'] = f[name + '/gtsummary'][()]
    data[name]['video_name'] = data_mat[name]['title']
    data[name]['user_anno'] = data_mat[name]['user_anno']
def knapsack(shot_scores, shot_change_points, keep = 0.15):
        durations = [end - start for start, end in shot_change_points]
        total_duration = shot_change_points[-1][1] + 1
        capacity = int(keep * total_duration)
        n = len(durations)
        K = [[0 for x in range(capacity + 1)] for x in range(n + 1)] 
        for i in range(n + 1): 
            for w in range(capacity + 1): 
                if i == 0 or w == 0:
                    K[i][w] = 0 
                elif durations[i-1] <= w: 
                    K[i][w] = max(shot_scores[i-1] + K[i-1][w-durations[i-1]], K[i-1][w]) 
                else: 
                    K[i][w] = K[i-1][w]

        selected = []
        w = capacity
        for i in range(n,0,-1):
            if K[i][w]!= K[i-1][w]:
                selected.insert(0,i-1)
                w -= durations[i-1]
        return selected 
def extract_key_frames(video_path, indexes):
    cap = cv2.VideoCapture(video_path)
    key_frames = []
    for index in indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        key_frames.append(frame)
        if not ret:
            break
    return key_frames
print('Loading model...')
model, preprocess = load_clip_model()
f1 = []
for video in data:
    input_query = data[video]['title']
    print('Extracting video shots...')
    video_change_points = data[name]['change_points']
    key_frames_indexes = data[name]['picks']
    VIDEO_FOLDER = ''
    key_frames = extract_key_frames(VIDEO_FOLDER + video + '.mp4', key_frames_indexes)


    print('Extracting query features...')
    query_feature = extract_query_features(model, input_query)

    print('Extracting image features...')
    image_features = extract_image_features(model, preprocess, key_frames)

    print('Scoring...')
    frame_scores = frame_scoring(query_feature, image_features)
    shot_scores = shot_scoring(frame_scores, video_change_points)

    print('Generating summary...')
    selected_shots = knapsack(shot_scores, video_change_points)
    machine_summary = np.zeros(data[name]['n_frames'])
    for shot in selected_shots:
        start, end = video_change_points[shot]
        machine_summary[start:end+1] = 1
    final_f_score, final_prec, final_rec = evaluate_summary(machine_summary, data[name]['gtsummary'])
    f1.append(final_f_score)
print('f1:', f1)
print(np.mean(f1))