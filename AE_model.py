"""Convolutional Variational Autoencoder Class"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.ops import nn
from tensorflow.contrib.layers import fully_connected, conv2d, conv2d_transpose
from tensorflow.contrib.layers import batch_norm, l2_regularizer

from src.libs import get_conv_output_shape
from src.libs.layers import dense_latent, deconv_latent, dense_nonlatent, deconv_nonlatent
#from models import *
from src.models.base_vae import BaseVAE

DATA_FORMAT="NCHW"

class CVAE(object):
    """
    Convolutional + Fully-Connected VAE
    """
    def __init__(self, model_conf, feed_dict):
        self._model_conf = model_conf
        self._feed_dict = feed_dict
        if not self._model_conf["sym"]:
            return
        if not self._model_conf["conv_enc"]:
            raise ValueError("need at least one Convolutional layer")
        if self._model_conf["hu_dec"]:
            raise ValueError("do not specify hu_dec if using symmetric model")
        if self._model_conf["deconv_dec"]:
            raise ValueError("do not specify deconv_dec if using symmetric model")

        self._model_conf["hu_dec"] = self._model_conf["hu_enc"][::-1]
        # add a dense layer match the last conv output dim
        conv_shape = get_conv_output_shape(
                self._model_conf["input_shape"], 
                self._model_conf["conv_enc"])
        self._model_conf["hu_dec"].append(np.prod(conv_shape))
        self._model_conf["conv_enc_output_shape"] = conv_shape

        # compute deconv layer shapes
        for i in range(len(self._model_conf["conv_enc"]) - 1)[::-1]:
            self._model_conf["deconv_dec"].append(
                    self._model_conf["conv_enc"][i][:1] + \
                    self._model_conf["conv_enc"][i + 1][1:])
        
        print("MODEL CONFIG:\n%s" % str(self._model_conf))
        
    def _build_encoder(self, inputs, reuse=False):
        weights_regularizer = l2_regularizer(self._model_conf["l2_weight"])
        normalizer_fn = batch_norm if self._model_conf["if_bn"] else None
        normalizer_params = None
        if self._model_conf["if_bn"]:
            normalizer_params = {"is_training": self._feed_dict["is_train"], 
                                 "reuse": reuse}
        outputs = inputs
        with tf.variable_scope("enc"):
            for i, (c, h, w, s_h, s_w, pad) in \
                    enumerate(self._model_conf["conv_enc"]):
                outputs = conv2d(
                        inputs=outputs,
                        num_outputs=c,
                        kernel_size=(h, w),
                        stride=(s_h, s_w),
                        padding=pad,
                        data_format=DATA_FORMAT,
                        activation_fn=nn.tanh,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_regularizer=weights_regularizer,
                        reuse=reuse,
                        scope="enc_conv%s" % (i + 1))

            output_dim = np.prod(outputs.get_shape().as_list()[1:])
            outputs = tf.reshape(outputs, [-1, output_dim])

            for i, hu in enumerate(self._model_conf["hu_enc"]):
                outputs = fully_connected(
                        inputs=outputs,
                        num_outputs=hu,
                        activation_fn=nn.tanh,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_regularizer=weights_regularizer,
                        reuse=reuse,
                        scope="enc_fc%s" % (i + 1))

            z_mu, z_logvar, z = dense_latent(outputs,
                                             self._model_conf["n_latent"],
                                             scope="latent_z")
        return [z_mu, z_logvar], z

    def _build_decoder(self, inputs, reuse=False):
        weights_regularizer = l2_regularizer(self._model_conf["l2_weight"])
        normalizer_fn = batch_norm if self._model_conf["if_bn"] else None
        normalizer_params = None
        if self._model_conf["if_bn"]:
            normalizer_params = {"is_training": self._feed_dict["is_train"], 
                                 "reuse": reuse}

        outputs = inputs
        with tf.variable_scope("dec"):
            for i, hu in enumerate(self._model_conf["hu_dec"]):
                outputs = fully_connected(
                        inputs=outputs,
                        num_outputs=int(hu),
                        activation_fn=nn.tanh,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_regularizer=weights_regularizer,
                        reuse=reuse,
                        scope="dec_fc%s" % (i + 1))
            
            # if no deconvolutional layers, use dense_latent for target
            print("shape of conv enc", self._model_conf["conv_enc_output_shape"])
            target_shape = list(self._model_conf["target_shape"])
            mu_nl = self._model_conf["x_mu_nl"]
            logvar_nl = self._model_conf["x_logvar_nl"]
            outputs = tf.reshape(
                    outputs, 
                    (-1,) + self._model_conf["conv_enc_output_shape"])
            for i, (c, h, w, s_h, s_w, pad) in \
                    enumerate(self._model_conf["deconv_dec"]):
                outputs = conv2d_transpose(
                        inputs=outputs,
                        num_outputs=c,
                        kernel_size=(h, w),
                        stride=(s_h, s_w),
                        padding=pad,
                        data_format=DATA_FORMAT,
                        activation_fn=nn.tanh,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_regularizer=weights_regularizer,
                        reuse=reuse,
                        scope="dec_deconv%s" % (i + 1))

            h, w, s_h, s_w, pad = self._model_conf["conv_enc"][0][1:]
            post_trim = (slice(None, target_shape[1]), slice(None, target_shape[2]))
            if self._model_conf["x_conti"]:
                x_mu, x_logvar, x = deconv_latent(
                        outputs,
                        num_outputs=target_shape[0],
                        kernel_size=(h, w),
                        stride=(s_h, s_w),
                        padding=pad,
                        data_format=DATA_FORMAT,
                        mu_nl=mu_nl,
                        logvar_nl=logvar_nl,
                        post_trim=post_trim,
                        scope="recon_x")
                px_z = [x_mu, x_logvar]
            else:
                raise NotImplementedError()

        return px_z, x


class CAE(object):
    """
    Convolutional + Fully-Connected AE
    """
    def __init__(self, model_conf, feed_dict):
        self._model_conf = model_conf
        self._feed_dict = feed_dict
        if not self._model_conf["sym"]:
            return
        if not self._model_conf["conv_enc"]:
            raise ValueError("need at least one Convolutional layer")
        if self._model_conf["hu_dec"]:
            raise ValueError("do not specify hu_dec if using symmetric model")
        if self._model_conf["deconv_dec"]:
            raise ValueError("do not specify deconv_dec if using symmetric model")

        self._model_conf["hu_dec"] = self._model_conf["hu_enc"][::-1]
        # add a dense layer match the last conv output dim
        conv_shape = get_conv_output_shape(
                self._model_conf["input_shape"], 
                self._model_conf["conv_enc"])
        self._model_conf["hu_dec"].append(np.prod(conv_shape))
        self._model_conf["conv_enc_output_shape"] = conv_shape

        # compute deconv layer shapes
        for i in range(len(self._model_conf["conv_enc"]) - 1)[::-1]:
            self._model_conf["deconv_dec"].append(
                    self._model_conf["conv_enc"][i][:1] + \
                    self._model_conf["conv_enc"][i + 1][1:])
        
        print("MODEL CONFIG:\n%s" % str(self._model_conf))
        
    def _build_encoder(self, inputs, reuse=False):
        weights_regularizer = l2_regularizer(self._model_conf["l2_weight"])
        normalizer_fn = batch_norm if self._model_conf["if_bn"] else None
        normalizer_params = None
        if self._model_conf["if_bn"]:
            normalizer_params = {"is_training": self._feed_dict["is_train"], 
                                 "reuse": reuse}
        outputs = inputs
        with tf.variable_scope("enc"):
            for i, (c, h, w, s_h, s_w, pad) in \
                    enumerate(self._model_conf["conv_enc"]):
                outputs = conv2d(
                        inputs=outputs,
                        num_outputs=c,
                        kernel_size=(h, w),
                        stride=(s_h, s_w),
                        padding=pad,
                        data_format=DATA_FORMAT,
                        activation_fn=nn.tanh,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_regularizer=weights_regularizer,
                        reuse=reuse,
                        scope="enc_conv%s" % (i + 1))

            output_dim = np.prod(outputs.get_shape().as_list()[1:])
            outputs = tf.reshape(outputs, [-1, output_dim])

            for i, hu in enumerate(self._model_conf["hu_enc"]):
                outputs = fully_connected(
                        inputs=outputs,
                        num_outputs=hu,
                        activation_fn=nn.tanh,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_regularizer=weights_regularizer,
                        reuse=reuse,
                        scope="enc_fc%s" % (i + 1))

            z = fully_connected(
                        inputs=outputs,
                        num_outputs=hu,
                        activation_fn=None,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_regularizer=weights_regularizer,
                        reuse=reuse,
                        scope="enc_fc_latent")

            enc_out = dense_nonlatent(outputs,
                                             self._model_conf["n_latent"],
                                             scope="latent_z")
        return enc_out

    def _build_decoder(self, inputs, reuse=False):
        weights_regularizer = l2_regularizer(self._model_conf["l2_weight"])
        normalizer_fn = batch_norm if self._model_conf["if_bn"] else None
        normalizer_params = None
        if self._model_conf["if_bn"]:
            normalizer_params = {"is_training": self._feed_dict["is_train"], 
                                 "reuse": reuse}

        outputs = inputs
        with tf.variable_scope("dec"):
            for i, hu in enumerate(self._model_conf["hu_dec"]):
                outputs = fully_connected(
                        inputs=outputs,
                        num_outputs=int(hu),
                        activation_fn=nn.tanh,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_regularizer=weights_regularizer,
                        reuse=reuse,
                        scope="dec_fc%s" % (i + 1))
            
            # if no deconvolutional layers, use dense_latent for target
            print("shape of conv enc", self._model_conf["conv_enc_output_shape"])
            target_shape = list(self._model_conf["target_shape"])
            mu_nl = self._model_conf["x_mu_nl"]
            logvar_nl = self._model_conf["x_logvar_nl"]
            outputs = tf.reshape(
                    outputs, 
                    (-1,) + self._model_conf["conv_enc_output_shape"])
            for i, (c, h, w, s_h, s_w, pad) in \
                    enumerate(self._model_conf["deconv_dec"]):
                outputs = conv2d_transpose(
                        inputs=outputs,
                        num_outputs=c,
                        kernel_size=(h, w),
                        stride=(s_h, s_w),
                        padding=pad,
                        data_format=DATA_FORMAT,
                        activation_fn=nn.tanh,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_regularizer=weights_regularizer,
                        reuse=reuse,
                        scope="dec_deconv%s" % (i + 1))

            h, w, s_h, s_w, pad = self._model_conf["conv_enc"][0][1:]
            post_trim = (slice(None, target_shape[1]), slice(None, target_shape[2]))
            if self._model_conf["x_conti"]:
                dec_out = deconv_nonlatent(
                        outputs,
                        num_outputs=target_shape[0],
                        kernel_size=(h, w),
                        stride=(s_h, s_w),
                        padding=pad,
                        data_format=DATA_FORMAT,
                        mu_nl=mu_nl,
                        logvar_nl=logvar_nl,
                        post_trim=post_trim,
                        scope="recon_x")
            else:
                raise NotImplementedError()

        return dec_out
