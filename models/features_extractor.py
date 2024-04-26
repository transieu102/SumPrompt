import torch 
import clip
from PIL import Image
import cv2
def load_clip_model(model="ViT-L/14"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model, device=device)
    return model, preprocess
def extract_image_features(model, preprocess, images, device = "cuda"):
    features = []
    for i in range(len(images)):
        features.append(model.encode_image(preprocess(Image.fromarray(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)))).unsqueeze(0).to(device)).cpu().detach().numpy()
    return features

def extract_query_features(model, query, device = "cuda"):
    return model.encode_text(clip.tokenize([query]).to(device)).cpu().detach().numpy()

