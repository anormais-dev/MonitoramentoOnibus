# classify_improved.py
# Run: python classify_improved.py
import os, sys, time, shutil
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import torch
import faiss
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
from tqdm import tqdm
import csv

# ---------- CONFIG ----------
RENDERS_DIR = "assets/model3d/renders"
DETECTIONS_DIR = "detections"
OUT_CORRECT = os.path.join(DETECTIONS_DIR, "correct")
OUT_FALSE = os.path.join(DETECTIONS_DIR, "false")
LOG_CSV = os.path.join(DETECTIONS_DIR, "classify_log_improved.csv")

INDEX_PATH = "renders_index.faiss"
META_PATH = "renders_meta.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_BATCH = 16
THRESH_CLIP = 0.35        # tune
THRESH_COLOR = 0.40       # tune (cosine of hist)
HIT_REQUIRED = 2          # require 2 of 3 votes (clip, color, classifier) - here we use clip+color
YOLO_MODEL = "yolov8n.pt" # detector to get vehicle crop
IMG_FOR_CLIP = 224

os.makedirs(OUT_CORRECT, exist_ok=True)
os.makedirs(OUT_FALSE, exist_ok=True)

# ---------- helpers ----------
def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted([str(p) for p in Path(folder).glob("*") if p.suffix.lower() in exts and Path(p).parent==Path(folder)])

# CLIP embedder (normalized)
class CLIPEmbedder:
    def __init__(self, device=DEVICE):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    def embed_pil(self, pil_list):
        inputs = self.proc(images=pil_list, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        feats = feats.cpu().numpy().astype("float32")
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-10
        feats = feats / norms
        return feats

# color hist (HSV) normalized flattened
def color_hist_hsv_bgr(bgr_img, bins=(32,32)):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0,1], None, [bins[0], bins[1]], [0,180,0,256])
    cv2.normalize(h, h)
    return h.flatten()

def cos_sim(a, b):
    a = a.astype("float32"); b = b.astype("float32")
    num = np.dot(a, b)
    den = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return float(num/den)

# build/load index (same as previous script)
def build_index_from_renders(embedder):
    paths = list_images(RENDERS_DIR)
    if not paths:
        print("Renders vazios em", RENDERS_DIR); sys.exit(1)
    print("Building renders index...")
    embs = []
    metas = []
    batch = []
    batch_paths = []
    for i, p in enumerate(paths):
        pil = Image.open(p).convert("RGB")
        batch.append(pil); batch_paths.append(p)
        if len(batch) >= EMB_BATCH or i == len(paths)-1:
            e = embedder.embed_pil(batch)
            embs.append(e)
            metas += batch_paths
            batch, batch_paths = [], []
    all_embs = np.vstack(embs).astype("float32")
    d = all_embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(all_embs)
    faiss.write_index(index, INDEX_PATH)
    np.save(META_PATH, np.array(metas))
    print("Index and meta saved.")
    return index, metas

def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        return None, None
    idx = faiss.read_index(INDEX_PATH)
    meta = np.load(META_PATH, allow_pickle=True).tolist()
    return idx, meta

# generate TTA crops (center crop + scales + horizontal flip)
def generate_tta_crops(bgr_crop, sizes=(224, 256)):
    pil_list = []
    h, w = bgr_crop.shape[:2]
    for scale in [1.0, 1.1, 0.9]:
        nh = int(h*scale); nw = int(w*scale)
        resized = cv2.resize(bgr_crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # center crop to IMG_FOR_CLIP
        ch = nh; cw = nw
        # if smaller than target, pad
        if nh < IMG_FOR_CLIP or nw < IMG_FOR_CLIP:
            pad_h = max(0, IMG_FOR_CLIP - nh); pad_w = max(0, IMG_FOR_CLIP - nw)
            resized = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0,0,0])
            nh, nw = resized.shape[:2]
        y0 = max(0, (nh - IMG_FOR_CLIP)//2); x0 = max(0, (nw - IMG_FOR_CLIP)//2)
        crop = resized[y0:y0+IMG_FOR_CLIP, x0:x0+IMG_FOR_CLIP]
        pil_list.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        # horizontal flip
        pil_list.append(Image.fromarray(cv2.cvtColor(cv2.flip(crop,1), cv2.COLOR_BGR2RGB)))
    return pil_list

# ---------- main classify ----------
def classify_all():
    embedder = CLIPEmbedder()
    index, meta = load_index()
    if index is None:
        index, meta = build_index_from_renders(embedder)
    yolo = YOLO(YOLO_MODEL)

    imgs = list_images(DETECTIONS_DIR)
    imgs = [p for p in imgs if Path(p).parent==Path(DETECTIONS_DIR)]
    if not imgs:
        print("Nenhuma imagem em", DETECTIONS_DIR); return

    csv_header = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        if csv_header:
            writer.writerow(["timestamp","src","dest","clip_score","color_score","best_render"])
        for p in tqdm(imgs):
            fname = os.path.basename(p)
            bgr = cv2.imread(p)
            if bgr is None:
                print("Erro abrir", p); continue

            # 1) detect vehicle crop using YOLO; we prefer bus bbox; if not found, fallback to big center crop
            try:
                res = yolo(bgr, imgsz=640)[0]
                crop_bgr = None
                if hasattr(res, "boxes") and len(res.boxes)>0:
                    # prefer bus class if present, else take largest box
                    chosen = None
                    for box in res.boxes:
                        cls = int(box.cls.item())
                        name = yolo.model.names.get(cls, str(cls))
                        x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
                        area = (x2-x1)*(y2-y1)
                        if name == "bus":
                            chosen = (x1,y1,x2,y2); break
                        if chosen is None or area > (chosen[2]-chosen[0])*(chosen[3]-chosen[1]):
                            chosen = (x1,y1,x2,y2)
                    if chosen:
                        x1,y1,x2,y2 = chosen
                        # clamp
                        h,w = bgr.shape[:2]
                        x1,x2 = max(0,x1), min(w,x2)
                        y1,y2 = max(0,y1), min(h,y2)
                        if x2-x1>5 and y2-y1>5:
                            crop_bgr = bgr[y1:y2, x1:x2].copy()
                else:
                    crop_bgr = None
            except Exception as e:
                print("YOLO error:", e); crop_bgr = None

            if crop_bgr is None:
                # fallback center crop (50% area)
                h,w = bgr.shape[:2]
                cw, ch = w//2, h//2
                x0 = max(0, (w-cw)//2); y0 = max(0, (h-ch)//2)
                crop_bgr = bgr[y0:y0+ch, x0:x0+cw].copy()

            # 2) compute color hist similarity: compare to all renders histograms (precompute once)
            # for efficiency: lazy compute renders histograms cache
            if not hasattr(classify_all, "renders_hists"):
                classify_all.renders_hists = []
                for rpath in meta:
                    rimg = cv2.imread(rpath)
                    if rimg is None:
                        classify_all.renders_hists.append(np.zeros(1))
                        continue
                    classify_all.renders_hists.append(color_hist_hsv_bgr(rimg))
                classify_all.renders_hists = np.array(classify_all.renders_hists, dtype=object)

            crop_hist = color_hist_hsv_bgr(crop_bgr)
            # compute best color similarity vs all renders
            best_color = 0.0; best_idx_color = 0
            for i_h, rh in enumerate(classify_all.renders_hists):
                try:
                    s = cos_sim(crop_hist, rh)
                except:
                    s = 0.0
                if s > best_color:
                    best_color = s; best_idx_color = i_h

            # 3) CLIP embeddings with TTA
            tta_pils = generate_tta_crops(crop_bgr)
            # batch in chunks
            emb_list = []
            B = 8
            for i in range(0, len(tta_pils), B):
                chunk = tta_pils[i:i+B]
                e = embedder.embed_pil(chunk)
                emb_list.append(e)
            emb_all = np.vstack(emb_list)  # (Ntta, D)
            emb_avg = emb_all.mean(axis=0, keepdims=True).astype("float32")
            emb_avg = emb_avg / (np.linalg.norm(emb_avg, axis=1, keepdims=True)+1e-10)

            # search index
            sims, idxs = index.search(emb_avg, 1)
            clip_score = float(sims[0,0]); best_idx_clip = int(idxs[0,0])
            best_render = meta[best_idx_clip]

            # Voting: require CLIP >= THRESH_CLIP and color >= THRESH_COLOR (2/2)
            votes = 0
            if clip_score >= THRESH_CLIP:
                votes += 1
            if best_color >= THRESH_COLOR:
                votes += 1

            dest_folder = OUT_CORRECT if votes >= 2 else OUT_FALSE
            dest_path = os.path.join(dest_folder, fname)
            try:
                shutil.move(p, dest_path)
            except Exception:
                shutil.copy2(p, dest_path); os.remove(p)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            writer.writerow([ts, fname, os.path.basename(dest_folder), f"{clip_score:.6f}", f"{best_color:.6f}", os.path.basename(best_render)])
            print(f"{fname} -> {os.path.basename(dest_folder)} clip={clip_score:.3f} color={best_color:.3f} best_render={os.path.basename(best_render)}")

    print("done.")

if __name__ == "__main__":
    classify_all()
