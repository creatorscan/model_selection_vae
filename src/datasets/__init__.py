import tensorflow as tf
import sys
sys.path.append('/mnt/matylda6/baskar/experiments/Tensorflow/sandbox/ModelSelectionVAE/src/')
from src.utils import *
from src.utils.logger import *
from src.datasets.dataset_utils import *

TF_FLOAT = tf.float32
NP_FLOAT = TF_FLOAT.as_numpy_dtype
