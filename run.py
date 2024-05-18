import numpy as np
import os
import random
import torch
from tqdm import tqdm
import streamlit as st
from pathlib import Path
import cv2

from ViCLIP import get_viclip, get_text_feat, get_vid_feat, normalize

def get_img_feat(img, base, device=torch.device('cuda')):
    model = get_viclip('l', f"{base}/models/ViCLIP-L_InternVid-DIV-10M.pth")
    clip, _ = model['viclip'], model['tokenizer']
    clip = clip.to(device)
    target_size=(224, 224)

    img = cv2.imread(img)
    img = cv2.resize(img, target_size)
    img = np.expand_dims(normalize(img), axis=(0, 1))

    img = np.transpose(img, (0, 1, 4, 2, 3))
    img = torch.from_numpy(img).to(device, non_blocking=True).float()

    img_feat = get_vid_feat(img, clip).to(device)

    return img_feat

def retrieve_vids(query, model, base, data, device=torch.device('cuda')):
    paths = {"database": "9063_1715580612.npy", "nvidia_data": "20240517-172725_nvidia_data_887.npy"}
    annotations = {"+": "4090", "*": "jensen", "_": "dancer"}

    clip, tokenizer = model['viclip'], model['tokenizer']
    clip = clip.to(device)

    embeddings_path = f'{base}/embeddings/{paths[data]}'
    #video_feats_tensor = torch.cat([torch.from_numpy(np.load(f'./embeddings_individual/{vid}')).to(device) for vid in os.listdir("./embeddings_individual")], 0)
    video_feats_tensor = torch.from_numpy(np.load(embeddings_path)).to(device)
    print(video_feats_tensor.shape)

    #text_feat = get_text_feat(query, tokenizer, clip)
    annotation_list = []
    for a in query.split(" "):
        if a in annotations:
            print("adding annotation:", a)
            annotation_list.append(annotations[a])
    
    #query_feat = torch.cat([1*get_img_feat(f"./{annotations[q]}", base) if q in annotations else 10*get_text_feat(q, tokenizer, clip) for q in query.split(" ")], 0) 
    text_feat = get_text_feat(query, tokenizer, clip) 
    query_feat = text_feat.to(device)
    
    if len(annotation_list) > 0:
        for a in os.listdir(f"{base}/annotations/{annotation_list[0]}"):
            if a[-4:] != ".npy":
                query_feat += 10*torch.from_numpy(np.load(f"{base}/annotations/_embeddings/{a}.npy")).to(device)
    
    sims = (query_feat @ video_feats_tensor.T).softmax(dim=-1)

    return sims

def search(query, base, top, data):
    
    database = f"{base}/{data}"
    model = get_viclip('l', f"{base}/models/ViCLIP-L_InternVid-DIV-10M.pth")
    titles = os.listdir(database)

    sims = retrieve_vids(query, model, base, data=data)
    sims_list = [i.item() for i in sims[0]]
    #print(sims_list)
    sims_sorted = sorted(sims_list.copy())[::-1]

    top_k = [sims_list.index(i) for i in sims_sorted[0:top]]
    return [f"{database}/{titles[ind]}" for ind in top_k]


def main(base, data):
    st.set_page_config(page_title="Semantic Search Research Demo", page_icon="🤟", layout="wide")
    st.title("Semantic Search Research Demo 🤟")

    query = st.text_input("Search videos", value="")

    cols_per_row = 4
    rows = 10

    if len(query) > 0:
        print("Testing Query: ", query)
        top_k = search(query, base, data=data, top=cols_per_row*rows)

        index = 0
        for i in range(rows):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if index < len(top_k):
                    with cols[j]:st.video(top_k[index])
                    index+=1
    if st.button("Load More"):
        rows+=2

if __name__ == '__main__': 
    main(os.path.dirname(__file__).replace('\\', '/'), data="nvidia_data")

    

#streamlit run "E:/adhoc_search/run.py"

    

