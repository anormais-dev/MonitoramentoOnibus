# single_image_classify_fixed.py
# Usage:
#   python single_image_classify_fixed.py path/to/image.jpg
#   python single_image_classify_fixed.py path/to/image_without_ext   # will try .jpg .png .jpeg
#
# This script is an improved, diagnostic version of single_image_classifier.py
# It searches for dataset in multiple likely locations and prints diagnostics.

import os, sys
from pathlib import Path
from typing import Union, List, Tuple
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

# optional faiss
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# CONFIG
CANDIDATE_DATASET_DIRS = ["dataset", "dataset/training", "dataset/train", "dataset/training", "dataset/trainings"]
TRY_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
INDEX_PATH = "dataset_index.faiss"
META_PATH = "dataset_meta.npy"
EMB_BATCH = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5
USE_CACHE = True

def locate_dataset():
    """
    Return tuple (base_dir_resolved, dict_of_class_dirs) or (None, None)
    class_dirs = {'correct': Path(...), 'false': Path(...)}
    """
    for base in CANDIDATE_DATASET_DIRS:
        b = Path(base)
        if not b.exists():
            continue
        # look for direct correct/false under base
        corr = b / "correct"
        fals = b / "false"
        if corr.exists() and fals.exists() and any(corr.glob("*")) and any(fals.glob("*")):
            return str(b), {"correct": str(corr), "false": str(fals)}
        # also check training/correct etc
        t_corr = b / "training" / "correct"
        t_fals = b / "training" / "false"
        if t_corr.exists() and t_fals.exists() and any(t_corr.glob("*")) and any(t_fals.glob("*")):
            return str(b / "training"), {"correct": str(t_corr), "false": str(t_fals)}
        # fallback: if base contains subfolders 'correct' and 'false' deeper
        for sub in b.glob("**/correct"):
            parent_false = sub.parent / "false"
            if parent_false.exists() and any(sub.glob("*")) and any(parent_false.glob("*")):
                return str(sub.parent), {"correct": str(sub), "false": str(parent_false)}
    return None, None

def list_dataset_images(base_dir: str, class_dirs: dict) -> List[Tuple[str,str]]:
    pairs = []
    for label, d in class_dirs.items():
        p = Path(d)
        for f in sorted(p.glob("*")):
            if f.is_file() and f.suffix.lower() in TRY_EXTS:
                pairs.append((str(f), label))
    return pairs

class CLIPEmbedder:
    def __init__(self, device=DEVICE):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    def embed_pil(self, pil_list):
        inputs = self.proc(images=pil_list, return_tensors="pt")
        for k in list(inputs.keys()):
            inputs[k] = inputs[k].to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        feats = feats.cpu().numpy().astype("float32")
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-10)
        return feats

def build_index(pairs, embedder):
    paths = [p for p, l in pairs]
    labels = [l for p, l in pairs]
    print(f"[build_index] embedding {len(paths)} images (device={DEVICE}) ...")
    all_embs = []
    meta = []
    batch_imgs = []
    batch_meta = []
    for i, path in enumerate(paths):
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            print("failed open", path, e); continue
        batch_imgs.append(pil); batch_meta.append((path, labels[i]))
        if len(batch_imgs) >= EMB_BATCH or i == len(paths)-1:
            embs = embedder.embed_pil(batch_imgs)
            all_embs.append(embs)
            meta.extend(batch_meta)
            batch_imgs, batch_meta = [], []
    all_embs = np.vstack(all_embs).astype("float32")
    if _HAS_FAISS:
        d = all_embs.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(all_embs)
        faiss.write_index(index, INDEX_PATH)
        np.save(META_PATH, np.array(meta))
        print("[build_index] faiss index saved.")
        return index, meta
    else:
        print("[build_index] faiss not available: returning brute-force arrays.")
        return (all_embs, None), meta

def load_cached_index():
    if USE_CACHE and _HAS_FAISS and Path(INDEX_PATH).exists() and Path(META_PATH).exists():
        try:
            idx = faiss.read_index(INDEX_PATH)
            meta = np.load(META_PATH, allow_pickle=True).tolist()
            print("[load_cached_index] loaded cached index")
            return idx, meta
        except Exception as e:
            print("failed to load cached index", e)
            return None, None
    return None, None

def try_open_image_candidate(arg: str):
    p = Path(arg)
    if p.exists() and p.is_file():
        return str(p)
    # try with extensions
    for ext in TRY_EXTS:
        cand = p.with_suffix(ext)
        if cand.exists() and cand.is_file():
            return str(cand)
    # try common subfolders (detections/...)
    for ext in TRY_EXTS:
        cand = Path(str(p) + ext)
        if cand.exists() and cand.is_file():
            return str(cand)
    return None

def majority_vote(meta, ids):
    counts = {}
    for i in ids:
        label = meta[int(i)][1]
        counts[label] = counts.get(label, 0) + 1
    sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return sorted_items[0][0], counts

def classify_single(image_arg):
    # locate dataset
    base_dir, class_dirs = locate_dataset()
    if base_dir is None:
        print("[ERROR] Could not find dataset. Expected folders like dataset/correct and dataset/false or dataset/training/...")
        print("Looked for:", CANDIDATE_DATASET_DIRS)
        return None
    print("[INFO] dataset base:", base_dir)
    print("[INFO] class dirs:", class_dirs)

    pairs = list_dataset_images(base_dir, class_dirs)
    if not pairs:
        print("[ERROR] No images found inside class dirs. Check that there are image files with extensions jpg/png/jpeg.")
        return None
    print(f"[INFO] found {len(pairs)} reference images (example): {pairs[:3]}")

    # prepare embedder and index
    embedder = CLIPEmbedder(device=DEVICE)
    idx, meta = load_cached_index()
    if idx is None:
        idx, meta = build_index(pairs, embedder)
        if not _HAS_FAISS:
            emb_array = idx[0]
        else:
            emb_array = None
    else:
        emb_array = None

    # resolve input image path (try extensions)
    resolved = try_open_image_candidate(image_arg)
    if resolved is None:
        print("[ERROR] Could not open image argument:", image_arg)
        print("Tried extensions:", TRY_EXTS)
        return None
    print("[INFO] using input image:", resolved)

    pil = Image.open(resolved).convert("RGB")
    emb = embedder.embed_pil([pil])  # (1,D)

    if _HAS_FAISS and emb_array is None:
        k = min(TOP_K, idx.ntotal)
        sims, ids = idx.search(emb.astype("float32"), k)
        sims = sims[0].tolist(); ids = ids[0].tolist()
        chosen, counts = majority_vote(meta, ids)
        top = [(meta[int(i)][0], meta[int(i)][1], float(s)) for i, s in zip(ids, sims)]
    else:
        if emb_array is None:
            print("[ERROR] No embeddings available for brute force.")
            return None
        sims = np.dot(emb_array, emb[0])
        ids = np.argsort(-sims)[:TOP_K].tolist()
        chosen, counts = majority_vote(meta, ids)
        top = [(meta[int(i)][0], meta[int(i)][1], float(sims[int(i)])) for i in ids]

    # print result
    print("----- RESULT -----")
    print("Prediction:", chosen.upper())
    print("Votes:", counts)
    print("Top matches:")
    for pth, lbl, sc in top:
        print(f"  {lbl:6s}  {sc:.4f}  {os.path.basename(pth)}")
    print("------------------")

    # IMPORTANT: return structured result so callers can use it programmatically
    return {
        "predicted": chosen,
        "votes": counts,
        "matches": top
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python single_image_classify_fixed.py path/to/image.jpg")
        sys.exit(1)
    classify_single(sys.argv[1])
