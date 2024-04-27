import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def frame_scoring(frames_features, query_features, method="cosine"):
    scores = []
    for i in range(len(frames_features)):
        if method == "cosine":
            scores.append(cosine_similarity(frames_features[i], query_features)[0][0])
    return scores

def shot_scoring(frame_scores, frame_indexes, shot_change_points, method="mean"):
    scores = []
    for shot in shot_change_points:
        sum_scores = []
        for i in range(shot[0], shot[1]+1):
            if i in frame_indexes:
                sum_scores.append(frame_scores[frame_indexes.index(i)])
        scores.append(np.mean(sum_scores))
    return scores
