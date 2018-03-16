# In[24]:


import tensorflow as tf
import numpy as np
import os, time
#import matplotlib.pyplot as plt
from scipy.stats import norm
from AE_train import vae_training
from AE_test import noise_corruption
from sklearn.decomposition import PCA
import sys
sys.path.append('/mnt/matylda6/baskar/experiments/Tensorflow/sandbox/ModelSelectionVAE/src')
#from runners.vae_runner import train, test, dump_repr, repl_repr_utt
from tools.kaldi.prep_kaldi_feat import flatten_channel
from datasets.simple_datasets_loaders import datasets_loader
from datasets.datasets_loaders import get_frame_ra_dataset_conf
from tools.vis import plot_heatmap
from tools.kaldi.plot_scp import plot_kaldi_feat
from kaldi_io_for_python import kaldi_io as kio
from AE_model import CVAE

if __name__ == '__main__':

    conf = {"n_chan" : int(1),
            "train_feat_rspec": "scp:/mnt/matylda6/baskar/experiments/kaldi/Timit/data-fbank/train/feats.scp",
            "test_feat_rspec": "scp:/mnt/matylda6/baskar/experiments/kaldi/Timit/data-fbank/train/feats.scp",
            #"train_feat_rspec": "scp:/mnt/matylda6/baskar/experiments/kaldi/Timit/data-fbank/noises/train_comb_exh/feats.scp",
            "dev_feat_rspec": "scp:/mnt/matylda6/baskar/experiments/kaldi/Timit/data-fbank/dev/feats.scp",
            #"test_feat_rspec": "/mnt/matylda6/baskar/experiments/kaldi/Timit/data-fbank/test/feats.scp",
            #"test_feat_rspec": "/mnt/matylda6/baskar/experiments/kaldi/Timit/data-fbank/noises_15db/test_exhibition/feats.scp",
            #"test_feat_rspec": "/mnt/matylda6/baskar/experiments/kaldi/Timit/data-fbank/test/feats.scp",
            "train_utt2label_paths": "{'uttid':['3697','/mnt/matylda6/baskar/experiments/Tensorflow/sandbox/ModelSelectionVAE/data/spec_scp/train/utt2uttid'],'spk':['463','/mnt/matylda6/baskar/experiments/Tensorflow/sandbox/ModelSelectionVAE/data/spec_scp/train/utt2spkid']}",
            "test_utt2label_paths": "{'uttid':['3697','/mnt/matylda6/baskar/experiments/Tensorflow/sandbox/ModelSelectionVAE/data/spec_scp/train/utt2uttid'],'spk':['463','/mnt/matylda6/baskar/experiments/Tensorflow/sandbox/ModelSelectionVAE/data/spec_scp/train/utt2spkid']}",
            #"train_utt2label_paths": "{'uttid':['7693','/mnt/matylda6/baskar/experiments/kaldi/Timit/data-fbank/noises/train_comb_exh/utt2uttid'],'spk':['463','/mnt/matylda6/baskar/experiments/kaldi/Timit/data-fbank/noises/train_comb_exh/utt2spkid']}",
            "dev_utt2label_paths" : "{'uttid':['401','/mnt/matylda6/baskar/experiments/Tensorflow/sandbox/ModelSelectionVAE/data/spec_scp/dev/utt2uttid'],'spk':['51','/mnt/matylda6/baskar/experiments/Tensorflow/sandbox/ModelSelectionVAE/data/spec_scp/dev/utt2spkid']}",
            #"test_utt2label_paths" : "{'uttid':['193','/mnt/matylda6/baskar/experiments/Tensorflow/sandbox/ModelSelectionVAE/data/spec_scp/test/utt2uttid'],'spk':['25','/mnt/matylda6/baskar/experiments/Tensorflow/sandbox/ModelSelectionVAE/data/spec_scp/test/utt2spkid']}",
            #"test_utt2label_paths" : "{'uttid':['193','/mnt/matylda6/baskar/experiments/kaldi/Timit/data-fbank/noises_15db/test_exhibition/utt2uttid'],'spk':['25','/mnt/matylda6/baskar/experiments/kaldi/Timit/data-fbank/noises_15db/test_exhibition/utt2spkid']}",
            #"test_utt2label_paths" : "{'uttid':['12613','/mnt/matylda6/baskar/experiments/kaldi/ami/data-fbank/ihm/eval_43/utt2uttid'],'spk':['64','/mnt/matylda6/baskar/experiments/kaldi/ami/data-fbank/ihm/eval_43/utt2spkid']}",
            "seg_len" : int(20),
            "seg_shift" : int(20),
            "seg_rand" : True,
            "remove_0th": False,
            "if_rand" : True,
            "mvn_path" : "/mnt/matylda6/baskar/experiments/Tensorflow/sandbox/ModelSelectionVAE/exp/fbank_mvn.pkl",
            "fmt" : "kaldi_ra",
            "max_to_load" : int(-1),
            "n_bins" : False,
            "train_utt2talabels_paths": None,
            "dev_utt2talabels_paths": None,
            "test_utt2talabels_paths": None,
            "use_chan": [0],
            "use_fbin": "slice(1, None, None)"}

    [train_set, dev_set, test_set] = datasets_loader(conf, train=True, dev=True, test=True)

    _model_conf = {"input_shape": train_set.feat_shape,
                            "input_dtype": tf.float32,
                            "target_shape": train_set.feat_shape,
                            "target_dtype": tf.float32,
                            "conv_enc": [(64,1,42,1,1,'valid'),(128,3,1,2,1,'same'),(256,3,1,2,1,'same')],#,128_3_1_2_1_same,256_3_1_2_1_same'",
                            "conv_enc_output_shape": None,
                            "hu_enc": [512],
                            "hu_dec": [],
                            "deconv_dec": [],
                            "n_latent": int(256),
                            "x_conti": True,
                            "x_mu_nl": None,
                            "x_logvar_nl": None,
                            "n_bins": None,
                            "if_bn": True,
                            "l2_weight": float(1.0),
                            "sym": True}
    exp_dir="exp/cvae_fbank_256bs_256latent_lr1e-3_grad_tanh_100epochs_nol2"
    vae_training(train_set, dev_set, test_set,
                 exp_dir, _model_conf,
                 batch_size=256, learn_rate=1e-3, num_epochs=100, save_model=True, debug=False)

    #is_train = False
    #tf.reset_default_graph()
    #exp_dir="exp/cvae_simple_128latent"
    #test(exp_dir, CVAE, test_set)

"""
    datadir='/mnt/matylda6/baskar/experiments/kaldi/Timit/data_tensorflow/cha'
    train_mfcc_dir = os.path.join(datadir, "train", "fbank")
    test_mfcc_dir = os.path.join(datadir, "test", "fbank")
    train_label_dir = os.path.join(datadir, "train", "label")
    test_label_dir = os.path.join(datadir, "test", "label")
    
    XX = [np.load(os.path.join(train_mfcc_dir, fn)) for fn in os.listdir(train_mfcc_dir)]
    inputs=[]
    for uttid in range(len(XX)):
        for t in XX[uttid].T:
            inputs.append(t)
    inputs=np.array(inputs)
    print(np.shape(inputs))
    pca = PCA()
    inputs = pca.fit_transform(inputs)
    #print(pca.explained_variance_ratio_)  
    #PCA(copy=True, iterated_power='auto',random_state=None, svd_solver='auto', tol=0.0, whiten=False)
    print(np.shape(inputs))

    YY = [np.load(os.path.join(test_mfcc_dir, fn)) for fn in os.listdir(test_mfcc_dir)]
    test_inputs=[]
    for uttid in range(len(YY)):
        for t in YY[uttid].T:
            test_inputs.append(t)
    test_inputs=np.array(test_inputs)
    test_inputs = pca.fit_transform(test_inputs)
    noise_inputs = noise_corruption(inputs, 0.6)
    # training on a subset to get a quick result
    print('Training ...')
    inpDim=np.shape(inputs)[1]
    vae_training(noise_inputs, inputs, test_inputs, test_inputs,
                 num_input=inpDim,
                 num_hidden_recog=512, 
                 num_hidden_gener=512,
                 num_z=128, 
                 batch_size=256, learn_rate=1e-3, num_epochs=500, save_model=True, debug=False)
"""
