# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os
import glob


##########################################################
def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

#he_normal_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False,seed=12345)
he_normal_init =tf.contrib.layers.xavier_initializer(uniform=False,seed=1234)


def Conv2D(inputs, kernel_shape, strides, padding, scope_name='Conv2d',W_initializer=he_normal_init, bias=True,trainable=True):
    '''
    A method that does convolution + relu on inputs
    '''
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        kernels=tf.get_variable('W',shape=kernel_shape,dtype=tf.float32,initializer=W_initializer,trainable=trainable)
        
        if bias is True:
            biases=tf.get_variable('b',shape=[kernel_shape[-1]],dtype=tf.float32,initializer=tf.constant_initializer(),trainable=trainable)
        else:
            biases = 0
        conv=tf.nn.bias_add(tf.nn.conv2d(inputs,kernels,strides=strides,padding=padding),biases)   

    return conv

def Conv3D(input, kernel_shape, strides, padding, scope_name='Conv3d', W_initializer=he_normal_init, trainable=True,bias=True):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        W = tf.get_variable("W", kernel_shape, dtype=tf.float32,initializer=W_initializer,trainable=trainable)
        if bias is True:
            b = tf.get_variable("b", (kernel_shape[-1]),dtype=tf.float32,initializer=tf.constant_initializer(value=0.0),trainable=trainable)
        else:
            b = 0
        
    return tf.nn.conv3d(input, W, strides, padding) + b