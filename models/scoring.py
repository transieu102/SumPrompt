import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def frame_scoring(frames_features, query_features, method="cosine"):
    scores = []
    for i in range(len(frames_features)):
        if method == "cosine":
            scores.append(cosine_similarity(frames_features[i], query_features))
    return scores

def shot_scoring(frame_scores, shot_change_points, method="mean"):
    scores = []
    for shot in shot_change_points:
        scores.append(np.mean(frame_scores[shot[0]:shot[1]+1]))
    return scores
