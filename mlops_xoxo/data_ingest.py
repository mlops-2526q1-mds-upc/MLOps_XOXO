from mlops_xoxo.utils.data_utils import rec_to_images
# Paths
REC_PATH = "data/external/casioface/train.rec"
IDX_PATH = "data/external/casioface/train.idx"
LST_PATH = "data/external/casioface/train.lst"
OUTPUT_DIR = "data/raw"
LIMIT = 10000  # Set an integer for debugging

def main():
    # Step 1: Extract images
    rec_to_images(REC_PATH, IDX_PATH, LST_PATH, OUTPUT_DIR, limit=LIMIT, show_progress=True)

if __name__ == "__main__":
    main()