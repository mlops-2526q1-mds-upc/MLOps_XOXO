import os
import cv2
import struct
import numpy as np
import re
import json
from pathlib import Path
from tqdm import tqdm


class RecordIO:
    """Pure Python RecordIO Reader."""
    def __init__(self, idx_path, rec_path):
        self.idx_path = idx_path
        self.rec_path = rec_path
        self.idx = {}
        with open(idx_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    rec_id, pos = parts
                    size = None
                elif len(parts) == 3:
                    rec_id, pos, size = parts
                else:
                    continue
                self.idx[int(rec_id)] = (int(pos), int(size) if size is not None else None)
        self.sorted_keys = sorted(self.idx.keys(), key=lambda k: self.idx[k][0])
        self.frec = open(rec_path, 'rb')

    def read_idx(self, idx):
        pos, size = self.idx[idx]
        self.frec.seek(pos)
        if size is not None:
            return self.frec.read(size)
        else:
            idx_pos = self.sorted_keys.index(idx)
            if idx_pos + 1 < len(self.sorted_keys):
                next_pos, _ = self.idx[self.sorted_keys[idx_pos + 1]]
                read_len = max(0, next_pos - pos)
                return self.frec.read(read_len)
            else:
                return self.frec.read()  # till EOF

    def close(self):
        self.frec.close()


def load_lst_mapping(lst_path):
    """Return mapping from record ID to original image path."""
    mapping = {}
    with open(lst_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            rec_id = int(parts[0])
            img_path = parts[1]
            mapping[rec_id] = img_path
    return mapping


def rec_to_images(rec_path, idx_path, lst_path, output_dir, limit=None, show_progress=True):
    reader = RecordIO(idx_path, rec_path)

    # Load lst lines in order
    with open(lst_path, 'r') as f:
        lst_lines = [line.strip().split() for line in f]

    keys = list(reader.idx.keys())
    if limit:
        keys = keys[:limit]

    if show_progress:
        from tqdm import tqdm
        keys_iter = tqdm(keys, desc="Extracting images")
    else:
        keys_iter = keys

    for i, rec_id in enumerate(keys_iter):
        raw = reader.read_idx(rec_id)
        if i >= len(lst_lines):
            break  # no corresponding line
        img_path = lst_lines[i][1]  # use path from lst line
        person_id = os.path.basename(os.path.dirname(img_path))
        person_dir = os.path.join(output_dir, person_id)
        os.makedirs(person_dir, exist_ok=True)

        # Find JPEG start
        start = re.search(b'\xff\xd8', raw)
        if not start:
            continue
        img_buf = raw[start.start():]
        img = cv2.imdecode(np.frombuffer(img_buf, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        save_path = os.path.join(person_dir, f"{rec_id}.jpg")
        cv2.imwrite(save_path, img)

    reader.close()
    print(f"Images extracted to {output_dir}")
    
def create_manifest(data_dir, output_path, splits={'train': 0.7, 'val': 0.2, 'test': 0.1}, shuffle=True):
    """
    Create train/val/test manifest.json from per-person folders.
    """
    data_dir = Path(data_dir)
    manifest = {'train': [], 'val': [], 'test': []}

    for person_dir in data_dir.iterdir():
        if not person_dir.is_dir():
            continue
        images = list(person_dir.glob("*.jpg"))
        if shuffle:
            np.random.shuffle(images)
        n = len(images)
        n_train = int(n * splits['train'])
        n_val = int(n * splits['val'])
        split_map = {
            'train': images[:n_train],
            'val': images[n_train:n_train+n_val],
            'test': images[n_train+n_val:]
        }
        for split, imgs in split_map.items():
            manifest[split].extend([str(img) for img in imgs])

    output_dir = os.path.dirname(output_path)
    if output_dir: 
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)