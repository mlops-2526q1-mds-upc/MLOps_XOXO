import re
import cv2
import numpy as np
from mlops_xoxo.utils.data_utils import RecordIO

REC_PATH = "data/raw/casioface/train.rec"
IDX_PATH = "data/raw/casioface/train.idx"

reader = RecordIO(IDX_PATH, REC_PATH)
rec_id = list(reader.idx.keys())[0]  # take first record
raw = reader.read_idx(rec_id)

# Find JPEG start
start = re.search(b'\xff\xd8', raw)
if start:
    img_buf = raw[start.start():]
    img = cv2.imdecode(np.frombuffer(img_buf, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        print("Image decoded successfully!")
        print("Shape:", img.shape)
        # Optionally save to check visually
        cv2.imwrite("test_img.jpg", img)
    else:
        print("Failed to decode image")
else:
    print("No JPEG marker found in record")

reader.close()