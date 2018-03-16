import tensorflow as tf
import numpy as np
import os, time
#import matplotlib.pyplot as plt
from scipy.stats import norm
from AE_model import CVAE
from src.libs.costs import kld, log_gauss
from src.utils import check_and_makedirs
from collections import OrderedDict

def init_or_restore_model(sess, model_dir, model_global_step, saver):
    ckpt = tf.train.get_checkpoint_state(model_dir)                                                  
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model params from %s" % ckpt.model_checkpoint_path)                            
        saver.restore(sess, ckpt.model_checkpoint_path)                                         
    else:
        print("Creating model with fresh params")
        sess.run(tf.global_variables_initializer())                                                  
    return sess.run(model_global_step)

def _build_graph(model_conf, learn_rate=0.008):
    feat_dim=model_conf["input_shape"]
    X = tf.placeholder(tf.float32, shape=(None,) + feat_dim, name="X") # input
    Y = tf.placeholder(tf.float32, shape=(None,) + feat_dim, name="Y") # targets
    M = tf.placeholder(tf.float32, shape=(None,) + feat_dim, name="M") # masks
    is_train = tf.placeholder(tf.bool, name="is_train")
    _feed_dict = {"X": X,
                  "Y": Y,
                  "M": M,
                  "is_train": is_train}
    cvae=CVAE(model_conf, _feed_dict)
    #qz_x, z = variationalAutoencoder_enc(X, _feed_dict["is_train"], model_conf, reuse=False)
    qz_x, z = cvae._build_encoder(X)
    px_z, x = cvae._build_decoder(z)
    #px_z, x = variationalAutoencoder_dec(z, _feed_dict["is_train"], model_conf, mu_nl=None, logvar_nl=None, reuse=False)

    with tf.name_scope("costs"):
        with tf.name_scope("neg_kld"):
                neg_kld = tf.reduce_mean(tf.reduce_sum(-1 * kld(*qz_x), axis=1))
        with tf.name_scope("logpx_z"):
            x_mu, x_logvar = px_z
            logpx_z = tf.reduce_mean(tf.reduce_sum(M * log_gauss(x_mu, x_logvar,
                          Y),axis=(1,2,3)))
        with tf.name_scope("l2"):
            reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        lb = neg_kld + logpx_z
        loss = -lb + reg_loss
    _outputs = {"qz_x": qz_x, "z": z, "px_z": px_z, "x": x,
                    "neg_kld": neg_kld, "logpx_z": logpx_z,
                    "reg_loss": reg_loss, "lb": lb, "loss": loss}
                    
    #optimizer with decayed learning rate
    learn_rate = tf.get_variable("learn_rate", trainable=False,  initializer=float(learn_rate))
    params = tf.trainable_variables()
    global_step = tf.Variable(0, trainable=False)#, initializer=0.0)
    with tf.name_scope("grad"):
        grads = tf.gradients(loss, params)
        #grads, _ = tf.clip_by_global_norm(grads, 50.0)
    with tf.name_scope("train"):
        opt = tf.train.AdamOptimizer(learning_rate=learn_rate, beta1=0.9, beta2=0.999)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = opt.apply_gradients(zip(grads, params), 
                                global_step=global_step)
        decay_op = learn_rate.assign(learn_rate * 0.8)
             
    _ops = {"train_step": train_step, "decay_op": decay_op}

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    return _feed_dict, _outputs, _ops, global_step, saver
   

def vae_training(train_set, dev_set, test_set, exp_dir, _model_conf, batch_size=500, learn_rate=0.008,
             num_epochs=1, save_model=False, debug=False):

    SESS_CONF = tf.ConfigProto(allow_soft_placement=True,device_count= {'GPU': 1}, log_device_placement=True)
    #SESS_CONF.gpu_options.per_process_gpu_memory_fraction = 0.9

    #exp_dir="exp/cvae_simple_128latent"
    model_dir = "%s/models" % exp_dir
    check_and_makedirs(model_dir)
    ckpt_path = os.path.join(model_dir, "vae.ckpt")

    _feed_dict, _outputs, _ops, model_global_step, saver =_build_graph(_model_conf,    
                                     learn_rate=learn_rate)
    model = [_feed_dict, _outputs, _ops, model_global_step, saver]

    #assert len(train_x.shape) == 2
    #[num_samples, feat_dim] = train_x.shape
    #num_classes = train_x.shape[-1]
    #num_steps = int(np.ceil(num_samples / float(batch_size)))

    #valid_num_samples = valid_x.shape[0]
    #valid_num_steps = int(np.ceil(valid_num_samples / float(2048)))

    lr_decay = 0.8 
    # build graph and define objective function
    # create summaries
    sum_names = ["reg_loss", "loss", "lb", "neg_kld", "logpx_z"]
    sum_vars = [tf.reduce_mean(_outputs[name]) for name in sum_names]
    with tf.variable_scope("train"):
        train_summaries = tf.summary.merge(
                [tf.summary.scalar(*p) for p in zip(sum_names, sum_vars)])
    with tf.variable_scope("test"):
        test_vars = OrderedDict([(name, tf.get_variable(name, initializer=0.)) \
                for name in sum_names])
        test_summaries = tf.summary.merge(
                [tf.summary.scalar(k, test_vars[k]) for k in test_vars])

    def make_feed_dict(inputs, targets, is_train):
        feed_dict = {_feed_dict["X"]: inputs,
                     _feed_dict["Y"]: targets,
                     _feed_dict["M"]: np.ones_like(inputs),
                     _feed_dict["is_train"]: is_train}
        return feed_dict

    dev_iterator_fn  = lambda: dev_set.iterator(2048) if dev_set else None 
    test_iterator_fn  = lambda: test_set.iterator(1) if test_set else None 
    n_steps_per_epoch=2000
    #n_steps_per_epoch=num_steps
    n_print_steps=100
    bs=batch_size
    n_epochs=num_epochs
    global_step=-1
    #epoch=-1
    passes=0
    # train the graph
    with tf.Session(config=SESS_CONF) as sess:
        start_time = time.time()
        init_step = init_or_restore_model(sess, model_dir, model_global_step, saver)
        global_step =  int(init_step)
        epoch = int(global_step // n_steps_per_epoch)
        print("init or restore model takes %.2f s" % (time.time() - start_time), flush=True)
        print("current #steps=%s, #epochs=%s" % (global_step, epoch), flush=True)
        if epoch >= n_epochs or epoch > 50:
            print("training is already done. exit")
            return test(SESS_CONF, _outputs, sum_names, sum_vars, saver, exp_dir, _feed_dict, model_global_step, test_iterator_fn)
        train_writer = tf.summary.FileWriter("%s/log/train" % exp_dir, sess.graph)
        dev_writer = tf.summary.FileWriter("%s/log/dev" % exp_dir)
        print("start training /\/\/|")
        best_epoch = -1
        best_dev_lb = -np.inf
        train_start_time = time.time()
        print_start_time = time.time()
        
        while True:
            for inputs, _, _, targets in train_set.iterator(bs):
                global_step, _ = sess.run([model_global_step, _ops["train_step"]],
                                 make_feed_dict(inputs, inputs, True))
                if global_step % n_print_steps == 0 and global_step != init_step:
                    outputs, summary = sess.run([sum_vars, train_summaries],
                                        make_feed_dict(inputs, inputs, False))
                    train_writer.add_summary(summary, global_step)
                    print("[epoch %.f step %.f pass %.f]: " % (
                            epoch, global_step, passes) + "print time=%.2fs" % (
                            time.time() - print_start_time) +", total time=%.2fs, " % (
                            time.time() - train_start_time) + ", ".join(["%s %.4f" % p for p in zip(
                            sum_names, outputs)]))
                    print_start_time = time.time()
                    if np.isnan(outputs[0]):
                        print("...exit training and not saving this epoch")
                        return
                if global_step % n_steps_per_epoch == 0 and global_step != init_step:
                    if dev_iterator_fn:
                        val_start_time = time.time()
                        dev_vals = _valid(sess, _feed_dict, sum_names, sum_vars, dev_iterator_fn)
                        feed_dict = dict(zip(test_vars.values(), dev_vals.values()))
                        summary = sess.run(test_summaries, feed_dict)
                        dev_writer.add_summary(summary, global_step)
                        print("[epoch %.f]: dev  \t" % epoch + "valid time=%.2fs" % (
                             time.time() - val_start_time) + ", total time=%.2fs, " % (
                             time.time() - train_start_time) + \
                             ", ".join(["%s %.4f" % p for p in dev_vals.items()]))
                        if dev_vals["lb"] > best_dev_lb:
                            best_epoch, best_dev_lb = epoch, dev_vals["lb"]
                            saver.save(sess, ckpt_path)
                    epoch += 1
                    if epoch >= n_epochs:
                        print("..finish training, time elapsed=%.2fs" % (
                            time.time() - train_start_time))
                        open("%s/.done" % exp_dir, "a")
                        return
            passes += 1    

def _valid(sess, _feed_dict, sum_names, sum_vars, iterator_fn):
    vals = OrderedDict([(name, 0) for name in sum_names])
    n_batches = 0
    for inputs, _, _, _ in iterator_fn():
        n_batches += 1
        outputs = sess.run(
                sum_vars, 
                feed_dict={
                    _feed_dict["X"]: inputs,
                    _feed_dict["Y"]: inputs,
                    _feed_dict["M"]: np.ones_like(inputs),
                    _feed_dict["is_train"]: 0})
        for name, val in zip(sum_names, outputs):
            vals[name] += val
    for name in vals:
        vals[name] /= n_batches
    return vals

def test(SESS_CONF, _outputs, sum_names, sum_vars, saver, exp_dir, _feed_dict, model_global_step, test_iterator_fn):
    model_dir = "%s/models" % exp_dir
    ckpt_path = os.path.join(model_dir, "vae.ckpt")
    with tf.variable_scope("sep_test"):
        sep_test_vars = OrderedDict([(name, tf.get_variable(name, initializer=0.)) \
                for name in sum_names])
        sep_test_summaries = tf.summary.merge(
                [tf.summary.scalar(k, sep_test_vars[k]) for k in sep_test_vars])
    with tf.Session(config=SESS_CONF) as sess:
        start_time = time.time()
        saver.restore(sess, ckpt_path)
        print("restore model takes %.2f s" % (time.time() - start_time))
        test_writer = tf.summary.FileWriter("%s/log/test" % exp_dir)

        test_vals = _valid(sess, _feed_dict, sum_names, sum_vars, test_iterator_fn)
        feed_dict = dict(zip(sep_test_vars.values(), test_vals.values()))
        summary, global_step = sess.run([sep_test_summaries, model_global_step], feed_dict)
        test_writer.add_summary(summary, global_step)
        print("test\t" + ", ".join(["%s %.4f" % p for p in test_vals.items()]))
        save_wloss=[]; save_wlb=[]; save_wkld=[]; save_wpx=[]
        for inputs, _, _, _ in test_iterator_fn():
            infeed_dict={ _feed_dict["X"]: inputs, _feed_dict["Y"]: inputs, _feed_dict["M"]: np.ones_like(inputs),_feed_dict["is_train"]: 0}
            wloss, wlb, wneg_kld, wlogpx_z =  sess.run([_outputs["loss"], _outputs["lb"], _outputs["neg_kld"], _outputs["logpx_z"]], feed_dict=infeed_dict)
            save_wloss.append(wloss)
            save_wlb.append(wlb)
            save_wkld.append(wneg_kld)
            save_wpx.append(wlogpx_z)
        np.save('vae_pdf_train_loss.npy', save_wloss)
        np.save('vae_pdf_train_lb.npy', save_wlb)
        np.save('vae_pdf_train_kld.npy', save_wkld)
        np.save('vae_pdf_train_px.npy', save_wpx)
