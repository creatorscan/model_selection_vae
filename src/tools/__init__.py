import sys
sys.path.append('/mnt/matylda6/baskar/experiments/Tensorflow/sandbox/ModelSelectionVAE')
from src.utils.logger import *

DEFAULT_FEAT_TYPE="fbank_raw"

def is_audio(egs):
    if egs in ["mnist"]:
        return False
    return True
