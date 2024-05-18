import numpy as np
import os
import cv2
import torch
from tqdm import tqdm
import time

from ViCLIP import get_viclip, frames2tensor, _frame_from_video, get_vid_feat
from run import get_img_feat

model_cfgs = {
    'viclip-l-internvid-div-10m': {
        'size': 'l',
        'pretrained': 'models/ViCLIP-L_InternVid-DIV-10M.pth',
    },
}

def embed_database(database, model, device=torch.device('cuda')):
    clip, _ = model['viclip'], model['tokenizer']
    clip = clip.to(device)
    #video_feats = []
    print("Generating Embeddings")
    for vid in tqdm(os.listdir(database)):
        if f'{vid}.npy' not in os.listdir("./embeddings_individual"):
            print("\n",vid)
            video = cv2.VideoCapture(f"{database}/{vid}")
            #frames = [frame for frame in _frame_from_video(video)]
            frames = [cv2.resize(frame, (frame.shape[1]//2,frame.shape[0]//2)) if (frame.shape[0] > 2000) or (frame.shape[1] > 2000) else frame for frame in _frame_from_video(video)]
            print(frames[0].shape)
            frames_tensor = frames2tensor(frames, device=device)
            #video_feats.append(get_vid_feat(frames_tensor, clip))
            np.save(f'./embeddings_individual/{vid}.npy', get_vid_feat(frames_tensor, clip).cpu()) #to save individual embeddings
        else: print(f"Skipping {vid}, embedding already exists")
        
    video_feats_tensor = torch.cat([torch.from_numpy(np.load(f'./embeddings_individual/{vid}.npy')).to(device) for vid in os.listdir(database)], 0)
    #video_feats_tensor = torch.cat(video_feats, 0)   
    np.save(f'./embeddings/{time.strftime("%Y%m%d-%H%M%S")}_{database.split("/")[-1]}_{len(os.listdir(database))}.npy', video_feats_tensor.cpu())

def embed_annotations(base, device=torch.device('cuda')):
    for annotation in os.listdir(f"{base}/annotations"):
        if annotation != "_embeddings":
            full_annotation_feat = []
            for a in os.listdir(f"{base}/annotations/{annotation}"):
                print("embedding:", a)
                if f'{a}.npy' not in os.listdir(f"{base}/annotations/_embeddings/"):
                    x = get_img_feat(f"{base}/annotations/{annotation}/{a}", base)
                    full_annotation_feat.append(x)
                    np.save(f"{base}/annotations/_embeddings/{a}.npy", x.cpu())
                else: print()
            np.save(f"{base}/annotations/{annotation}/{annotation}_full.npy", sum(full_annotation_feat).cpu())
            
                

if __name__ == '__main__':

    #database = "E:/adhoc_search/nvidia_data"
    #cfg = model_cfgs['viclip-l-internvid-div-10m']
    #model = get_viclip(cfg['size'], cfg['pretrained'])
    #embed_database(database, model)
    embed_annotations(os.path.dirname(__file__).replace('\\', '/'))

    

