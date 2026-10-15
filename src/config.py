TRAIN_PATH = "../dataset/deepwriting_training.npz"
VAL_PATH = "../dataset/deepwriting_validation.npz"
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 150
CLIP = 10.0
VOCAB_SIZE = 70
CKPT_DIR = "../checkpoints"
SAMPLE_EVERY = 2
OUT_DIR = "../outputs"
MAX_LEN = 400

PERSONAL = '../personal_data/personal.npz'
FT_LR = 1e-4
FT_EPOCHS = 60
PATIENCE = 10
FT_BATCH = 16

MAX_CHARS = 20
# user's data's median
TARGET_SPC = 9.6

LINE_H = 0.05
TRIES=6

TEXT = '''This page was written by an artificial intelligence system trained on my own handwriting. The purpose of this experiment is to demonstrate how deep learning can learn and reproduce the unique characteristics of an individual's handwriting style. By analyzing samples of my handwritten text, the system learns patterns such as letter shapes, spacing, slant, stroke thickness, and the connections between characters. These features help the model generate new text that resembles my natural handwriting.'''
