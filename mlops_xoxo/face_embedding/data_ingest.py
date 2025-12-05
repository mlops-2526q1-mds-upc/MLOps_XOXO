from utils.data_utils import rec_to_images


# Paths
REC_PATH = "data/external/face_embedding/casioface/train.rec"
IDX_PATH = "data/external/face_embedding/casioface/train.idx"
LST_PATH = "data/external/face_embedding/casioface/train.lst"
OUTPUT_DIR = "data/raw/face_embedding"
LIMIT = 10000  # Set an integer for debugging


def main():
    """Extract images"""
    rec_to_images(REC_PATH, IDX_PATH, LST_PATH, OUTPUT_DIR, limit=LIMIT, show_progress=True)


if __name__ == "__main__":
    main()
