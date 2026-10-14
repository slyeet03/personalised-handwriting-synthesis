TRAIN_PATH = "../dataset/deepwriting_training.npz"
VAL_PATH = "../dataset/deepwriting_validation.npz"
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 150
CLIP = 10.0
VOCAB_SIZE = 70
CKPT_DIR = "../checkpoints"
SAMPLE_EVERY = 2
SAMPLE_TEXT = "hello world"
OUT_DIR = "../outputs"
MAX_LEN = 400

PERSONAL = '../personal_data/personal.npz'
FT_LR = 1e-4
FT_EPOCHS = 60
PATIENCE = 10
FT_BATCH = 16
