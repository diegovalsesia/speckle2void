#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.insert(0, './libraries')

import tensorflow as tf
import numpy as np
import time
import glob
import scipy
import argparse

#import os
import scipy.io as sio
import tensorflow as tf
from keras.engine.training_utils import iter_sequence_infinite
from DataGenerator import DataGenerator
from DataWrapper import DataWrapper
import keras.backend as K

from utils import Conv2D, Conv3D, safe_mkdir

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import json
import shutil
import argparse
import random

from termcolor import cprint
from tqdm import trange
from tqdm import tqdm


class Speckle2V(object):
    """The Speckle2V class to train and test a blind-spot network with variable spot shape
        Parameters
        ----------
        dir_train   : str
                      directory with training data.
        dir_test    : str
                      directory with test data.
        file_checkpoint    : str
                      checkpoint for loading a specific model. If None, the latest checkpoint is loaded
        batch_size  : int
                      size of the mini-batch.
        model_name : str
                      initial directory name where to save the checkpoints.
        lr             : float
                      learning rate.
        steps_per_epoch : int
                      steps for each epoch 
        k_penalty_tv : float
                      coefficient to weigh the total variation term in the loss
        norm         : float
                      normalization
        clip         : float
                      intensity value to clip the SAR images
        shift_list   : list of int
                      list of the possible shifts to apply to the receptive fields at the end of the network.
        prob         : list of float
                      list of the probabilities for choosing the possible shifts.
        L_noise      : float
                      parameter L of the noise distribution gamma(L,L) used to model the speckle
                      
        
        """
    
    def __init__(self, 
                 dir_train,
                 dir_test ,
                 file_checkpoint,
                 batch_size,
                 patch_size,
                 model_name, 
                 lr, 
                 steps_per_epoch,
                 k_penalty_tv = None, 
                 shift_list = [3,1], 
                 prob = [0.95,0.05],
                 clip = 500000,
                 norm = 100000,
                 L_noise=1):
        
        self.dir_train = dir_train
        self.dir_test = dir_test
        self.file_checkpoint = file_checkpoint
        self.batch_size = batch_size
        self.k_penalty_tv = k_penalty_tv if k_penalty_tv is not None else 0
        self.norm = norm #normalizer
        self.clip = clip #clipping very high value that don't make sense, bad electromagnetic reflection
        self.learning_rate = lr
        self.L = float(L_noise)
        self.shift_list = shift_list
        self.prob = prob
        self.steps_per_epoch = steps_per_epoch
        self.img_rows = patch_size
        self.img_cols = patch_size
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.checkpoint_dir = '{0}_tv_{3}_shiftlist{7}_prob{8}_clip{4}_norm{5}_Lnoise_{6}_lr_{1}_b_{2}_'.format(model_name,
                                                                                                          lr,
                                                                                                          batch_size,
                                                                                                          k_penalty_tv,
                                                                                                          clip,
                                                                                                          norm,
                                                                                                          L_noise,
                                                                                                '-'.join([str(x) for x in shift_list]),
                                                                                                '-'.join([str(x) for x in prob]))
        
        
       
        
        
        
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.placeholder()
        self.sess = tf.Session()
        
        self.loaded_weights = False
        
       
    def placeholder(self):
        
        
        self.X_noisy=tf.placeholder('float32',shape=[None,None,None,self.channels],name='X_noisy')
        self.mask_holder=tf.placeholder('float32',shape=[None,None,None,self.channels],name='mask')
        self.shift=tf.placeholder('int32',shape=[],name='shift')
        
        self.L_holder = tf.placeholder('float32', shape=[] ,name='L')
        self.is_train = tf.placeholder(tf.bool, shape=[])
        
        
    def get_data(self):
        with tf.name_scope('data'):
            
            # Training Data Preparation

            # DataGenerator to load data and extract patches for training
            datagen = DataGenerator()
            imgs = datagen.load_imgs_from_directory(directory = self.dir_train,filter='decorr*.mat',max_files=None)
            
            # Extracting overlapping training patches 256x256 
            images = datagen.generate_patches_from_list(imgs, shape=(256,256),num_patches_per_img=2000,augment=False)
            
            np.random.shuffle(images)
            
            X_train_noisy = images
            
            
            #####TRAINING
            #Compute mask for training images to exclude them in the loss computation
            indexes = np.where(X_train_noisy > self.clip)
            self.mask_train = np.ones_like(X_train_noisy,dtype=np.bool)
            self.mask_train[indexes] = False
            
            print('Clipping...')
            #Replace high backscatters with the median
            X_train_noisy = np.clip(X_train_noisy, 0, None)
            self.X_train_noisy_clipped = X_train_noisy
            medians = np.median(X_train_noisy,axis=[1,2],keepdims=True)
            self.X_train_noisy_clipped = np.where(self.X_train_noisy_clipped > self.clip,medians, X_train_noisy)
            
            
            print('Normalizing...')
            #Normalization
            self.X_train_noisy_clipped = (self.X_train_noisy_clipped.astype(np.float32))
            self.X_train_noisy_clipped /= self.norm
            
            self.training_data_wrapper=DataWrapper(self.X_train_noisy_clipped,self.mask_train,self.batch_size,shape=(self.img_rows,self.img_cols))
        
            self.training_data_iter=iter_sequence_infinite(self.training_data_wrapper)
            
            
            
            ######TEST IMAGES
            images_test = datagen.load_imgs_from_directory(directory = self.dir_test,filter='decorr_complex_tsx_SLC_0.mat')
            images_test = np.array(images_test)
            #cropping some test images
            images_test = np.array([images_test[0,0,i:i+1000,j:j+1000,:] for i,j in zip([5000,5500,4000,3000,0,1000,5500],[4000,5000,3500,7000,5000,5000,500])])
            self.images_test = images_test
            #Compute mask for test images to be able to place the point targets back into the denoised estimate
            indexes = np.where(self.images_test > self.clip)
            self.mask_test = np.ones_like(self.images_test,dtype=np.bool)
            self.mask_test[indexes] = False
    
            
            #Clipping high backscattering FOR TEST
            self.images_test_clipped = np.clip(self.images_test, 0, None)
            medians = np.median(self.images_test,axis=[1,2],keepdims=True)
            self.images_test_clipped = np.where(self.images_test_clipped > self.clip,medians, self.images_test)
            #Normalization 
            self.images_test_clipped = (self.images_test_clipped.astype(np.float32)) / self.norm
     
            
           

            
    def inference(self,h,scope_name):
        
        def dynamic_shift(inp, pad_size):
            x1 =tf.pad(inp, [[0,0], [pad_size,0], [0,0], [0,0]], mode='CONSTANT')
            x1 = x1[:,:-pad_size,:]
            return x1
            
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            
            F=64
            
            intermediates = []
            
            
            for j in range(4):
                x1=tf.image.rot90(h,k=j,name=None)
                
            
                if j in [0,2]:
                    with tf.variable_scope('nety', reuse=tf.AUTO_REUSE) as scope:
                        sp = [[0,0], [1,0], [0,0], [0,0]]
                        x1 = Conv2D(tf.pad(x1, sp, mode='CONSTANT'), [3,3,self.channels,F], [1,1,1,1], 'SAME', scope_name='conv_0')
                        x1 = tf.nn.leaky_relu(x1)
                        #Remove last row
                        x1 = x1[:,:-1,:]
                        
                        # 15 layers,Conv+BN+relu
                        for i in range(15):
                            x1 = Conv2D(tf.pad(x1, sp, mode='CONSTANT'), [3,3,F,F], [1,1,1,1], 'SAME', scope_name='conv_{0}'.format(i+1))
                            x1 = tf.layers.batch_normalization(x1, axis=-1,training=self.is_train,name='bn_{0}'.format(i+1))
                            x1 = tf.nn.leaky_relu(x1)
                            x1 = x1[:,:-1,:]
                         
                        # last layer, Conv
                        x1 = Conv2D(tf.pad(x1, sp, mode='CONSTANT'), [3,3,F,F], [1,1,1,1], 'SAME', scope_name='conv_last')
                        x1 = x1[:,:-1,:] 
                        
                        #Computing the shift to apply to the receptive fields
                        shift = tf.cond(tf.equal(self.shift, 1), 
                                   lambda: dynamic_shift(x1,1), 
                                   lambda: dynamic_shift(x1,2))
                        #Applying the computed shift only during training otherwise the canonical shift by 1 is applied
                        x1 = tf.cond(tf.equal(self.is_train, True), 
                                   lambda: shift, 
                                   lambda: dynamic_shift(x1,1))
                        
                        
                        #Rotating back
                        x1 = tf.image.rot90(x1,k=4-j,name=None)
                        intermediates.append(x1)
                else:
                    with tf.variable_scope('netx', reuse=tf.AUTO_REUSE) as scope:
                        sp = [[0,0], [1,0], [0,0], [0,0]]
                        x1 = Conv2D(tf.pad(x1, sp, mode='CONSTANT'), [3,3,self.channels,F], [1,1,1,1], 'SAME', scope_name='conv_0')
                        x1 = tf.nn.leaky_relu(x1)
                        #Remove last row
                        x1 = x1[:,:-1,:]
                        
                        # 15 layers, Conv+BN+relu
                        for i in range(15):
                            x1 = Conv2D(tf.pad(x1, sp, mode='CONSTANT'), [3,3,F,F], [1,1,1,1], 'SAME', scope_name='conv_{0}'.format(i+1))
                            x1 = tf.layers.batch_normalization(x1, axis=-1,training=self.is_train,name='bn_{0}'.format(i+1))
                            x1 = tf.nn.leaky_relu(x1)
                            x1 = x1[:,:-1,:]
                         
                        # last layer, Conv
                        x1 = Conv2D(tf.pad(x1, sp, mode='CONSTANT'), [3,3,F,F], [1,1,1,1], 'SAME', scope_name='conv_last')
                        x1 = x1[:,:-1,:] 
                        
                        #Applying the canonical shift for the horizontally extending receptive fields
                        x1 = dynamic_shift(x1,1)
                        
                        #Rotating back
                        x1 = tf.image.rot90(x1,k=4-j,name=None)
                        intermediates.append(x1)
                    
                
            images_to_combine=tf.stack(intermediates,axis=1)
            
            x1 = Conv3D(images_to_combine, [4,1,1,F,F], [1,1,1,1,1], 'VALID', scope_name='conv_comb_0')
            x1 = tf.nn.leaky_relu(x1)
            x1 = tf.squeeze(x1,axis=1)
            x1 = Conv2D(x1                , [1,1,F,F], [1,1,1,1], 'SAME', scope_name='conv_comb_1')
            x1 = tf.nn.leaky_relu(x1)
            x1 = Conv2D(x1                , [1,1,F,2], [1,1,1,1], 'SAME', scope_name='conv_comb_2')
            x1 = tf.nn.relu(x1)
        
    
        return x1 
    
    
    
  
    def build_inference(self):
        
        self.out_alpha_beta = self.inference(self.X_noisy,'denoising_network')
        self.alpha = ((self.out_alpha_beta[:,:,:,0:1])) + 1 #alpha>1
        self.beta = (self.out_alpha_beta[:,:,:,1:2])
        
        ########Compute prior mean of P(x|omega_y)#######
        self.X_prior = (self.beta) / (self.alpha - 1 + 1e-19) 
        
        ########Compute posterior mean of P(x|y,omega_y)#######
        #posterior with beta and alpha as they are coming out of the cnn 
        self.X_posterior = (self.beta + (self.L_holder * self.X_noisy))                            / (self.L_holder + self.alpha - 1 + 1e-19)
        
        self.X_posterior_clip = tf.clip_by_value((self.X_posterior * self.norm), 0, 50000)
        
    def loss(self):
        '''
        define loss: negative log of probability of noisy pixel yi given the receptive field of yi, excluding yi itself.
        '''
        sh = tf.shape(self.alpha)
        L_replicated = tf.broadcast_to(self.L_holder, [sh[0],sh[1],sh[2],1], name='L_replicated')
        alpha_L = tf.concat([self.alpha,L_replicated],axis=-1) 
        
        log_beta = tf.log((self.beta) + 1e-19)
        alpha_log_beta_complete = (- self.alpha * log_beta)
        alpha_log_beta_noisy_complete = (self.L_holder + self.alpha) * tf.log(self.beta + (self.L_holder * self.X_noisy) + 1e-19)
        beta_func_complete = tf.expand_dims(tf.math.lbeta(alpha_L), axis=-1)
        
        log_p_y =   (self.L_holder * tf.log(self.L_holder + 1e-19))                     + ((self.L_holder-1) * tf.log(self.X_noisy+1e-19))                     - (alpha_log_beta_complete)                     - (alpha_log_beta_noisy_complete)                     - beta_func_complete
        
        self.log_p_y_1 = log_p_y + 0.0
        #Apply mask to exclude the pixels with the median from the loss computation
        log_p_y = log_p_y * self.mask_holder
        
        ##tot variation
        tot_var = tf.image.total_variation(self.X_posterior)
        #tot_var = tf.image.total_variation(self.X_prior)
        
        self.total_variation = self.k_penalty_tv * tf.reduce_mean(tot_var)
        
        #From log likelihood to loss
        self.loss = - (tf.reduce_sum(log_p_y)/tf.reduce_sum(self.mask_holder)) 
        #Adding total variation regularizer
        self.loss = self.loss + self.total_variation
        
        
        #to be plotted
        self.alpha_log_beta = tf.reduce_mean(alpha_log_beta_complete)
        self.alpha_log_beta_noisy = tf.reduce_mean(alpha_log_beta_noisy_complete)
        self.difference = - self.alpha_log_beta + self.alpha_log_beta_noisy
        self.beta_func = tf.reduce_mean(beta_func_complete)
            
    def optimize(self):
        '''
        define optimization algorithm
        '''
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #print("Batch norm variables {}".format([v.name for v in update_ops]))
        with tf.control_dependencies(update_ops):
            with tf.name_scope('optimizer') as scope:
                self.opt=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.gstep)
    
        
    
    def PSNR(self, y_true, y_pred):
        """
        PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        The equation is:
        PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
        """
        return -10.0 * tf.log(tf.reduce_mean(tf.square(y_pred - y_true))) / tf.log(10.0) 
     
        
    def ENL(self, y_pred):
        """
        ENL
        """
        mean, variance = tf.nn.moments(y_pred, [1,2])
        return tf.reduce_mean(tf.square(mean)/(variance + 1e-08))
     
    
    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('performance') as scope:
            ################# summaries ###################
            tf.summary.scalar('loss', self.loss, collections=['loss'])
            tf.summary.scalar('beta_func', self.beta_func, collections=['portions_loss'])
            tf.summary.scalar('alpha_log_beta', self.alpha_log_beta, collections=['portions_loss'])
            tf.summary.scalar('alpha_log_beta_noisy', self.alpha_log_beta_noisy, collections=['portions_loss'])
            tf.summary.scalar('difference', self.difference, collections=['portions_loss'])
            tf.summary.scalar('total_var', self.total_variation, collections=['portions_loss'])
            
            #no reference metric
            tf.summary.scalar('ENL', self.enl, collections=['metrics'])
            
        self.summary_loss=tf.summary.merge_all(key='loss')
        self.summary_portions_loss=tf.summary.merge_all(key='portions_loss')
        self.summary_metrics=tf.summary.merge_all(key='metrics')
        
        
        #Images on tensorboard
        with tf.name_scope('images') as scope:
            tf.summary.image('images_denoised', self.X_posterior_clip, 3,collections=['images'])
            tf.summary.image('images_noisy', tf.clip_by_value((self.X_noisy * self.norm), 0, 50000)  , 3,collections=['images'])
            tf.summary.image('images_test', self.X_posterior_clip, 2,collections=['images_test'])
            tf.summary.image('images_noisy_test', tf.clip_by_value((self.X_noisy * self.norm), 0, 50000), 2,collections=['images_test'])
        
        
        ## Merge all summaries related to images collection
        self.tf_images_summaries = tf.summary.merge_all(key='images') 
        self.tf_images_test_summary = tf.summary.merge_all(key='images_test')
        
        ##Plot hist of prior and posterior images
        tf.summary.histogram('posterior_x_hist', tf.reshape(self.X_posterior,[-1]),collections=['parameters'])
        tf.summary.histogram('noisy_x_hist', tf.reshape(self.X_noisy,[-1]),collections=['parameters'])
        
        ## Merge all parameter histogram summaries together
        self.tf_param_summaries = tf.summary.merge_all(key='parameters')
        
        
    def train_one_epoch(self,saver,train_writer,test_writer,epoch,step):
        start_time = time.time()
        n_batches=0
        total_loss=0
        
        
        
        
        for i in range(0, self.steps_per_epoch):
            
            # ---------------------
            #  Train network
            # ---------------------
            # Select a random batch of images
                                         
                               
            noisy,mask = next(self.training_data_iter)
            #Randomly choose one of the two shifts
            shift=np.random.choice(self.shift_list,p=self.prob)
            
            #Run session to compute summaries
            if (step+1)%2000 == 0:
                
                
                _,loss,summary_loss,                summary_portions_loss,                summary_hist= self.sess.run([self.opt, 
                                             self.loss,
                                             self.summary_loss,
                                             self.summary_portions_loss,
                                             self.tf_param_summaries],feed_dict={self.X_noisy:noisy,
                                                                                 self.is_train:True,
                                                                                 self.L_holder:self.L,
                                                                                 self.mask_holder:mask,
                                                                                 self.shift:shift}
                                            )
                train_writer.add_summary(summary_loss, global_step=step)
                train_writer.add_summary(summary_portions_loss, global_step=step)
                train_writer.add_summary(summary_hist, global_step=step)
                
                train_writer.flush()
                
                cprint("step:{0} - epoch:{2} [loss: {1}]".format(step, loss,epoch))
            else:
                
                _,loss = self.sess.run([self.opt, self.loss],feed_dict = {self.X_noisy:noisy,
                                                                          self.is_train:True,
                                                                          self.L_holder:self.L,
                                                                          self.mask_holder:mask,
                                                                          self.shift:shift}) 
            
            
            
            if (step+1)%5000==0:
                
                
                
                #View images on tensorboard
                
                images_summary_test=self.sess.run(self.tf_images_test_summary,feed_dict={
                                                      self.X_noisy:self.images_test_clipped[[0,1]],
                                                      self.is_train:False,
                                                       self.L_holder:self.L,
                                                      self.shift:1})   
                
                
                summary_metrics = self.sess.run(self.summary_metrics,feed_dict={self.X_noisy:self.images_test_clipped[0:1,715:800, 43:113,:],
                                                                                self.is_train:False,
                                                                                self.L_holder:self.L,
                                                                               self.shift:1})
                
                
                #train_writer.add_summary(images_summaries, global_step=step)
                train_writer.add_summary(images_summary_test, global_step=step)
                train_writer.add_summary(summary_metrics, global_step=step)
                
            
            if (step+1)%5000 == 0:
                saver.save(self.sess, 'checkpoints/'+self.checkpoint_dir+'/'+'model.ckpt', step)
            
            if (step+1)%10000 == 0:
                self.test()
                                
            total_loss += loss
            n_batches += 1
            step += 1
            
        #Shuffle the 30000 images 
        self.training_data_wrapper.on_epoch_end()
        
        
        return step
        
        
    def train(self,n_epochs):
        
                              
        
        safe_mkdir('checkpoints')
        safe_mkdir('checkpoints/'+self.checkpoint_dir)
        #To plot two different curves on the same graph we need two different writers that write the
        #same group of summaries.
        train_writer = tf.summary.FileWriter('./graphs/'+self.checkpoint_dir + '/train', tf.get_default_graph())
        test_writer = tf.summary.FileWriter('./graphs/'+self.checkpoint_dir + '/test',tf.get_default_graph())
        #self.sess.run(tf.global_variables_initializer())
        #
        #
        #
        #saver = tf.train.Saver(max_to_keep=None)
        #ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/'+self.checkpoint_dir+'/checkpoint'))
        #if ckpt and ckpt.model_checkpoint_path:
        #    saver.restore(self.sess, ckpt.model_checkpoint_path)
            
        
        saver = self.load_weights()
                  
                                
        step = self.gstep.eval(session=self.sess)
        
        cprint("[!] Restarting at iteration {}".format(step), color="yellow")
                              
        for epoch in range(n_epochs):
            step = self.train_one_epoch(saver, train_writer,test_writer, epoch, step)
        
        return step
    
    
    def eval(self):
        '''
        Compute no-reference metric: enl
        '''
        with tf.name_scope('ENL'):
            self.enl=self.ENL(self.X_posterior)   
        


        
    def test(self,file_checkpoint=None):
        #return                      
        if not self.loaded_weights:
            self.sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=40)
            if file_checkpoint:
                if os.path.isfile('{0}.index'.format(file_checkpoint)):
                    print('Taking the specified checkpoint...')
                    saver.restore(self.sess,file_checkpoint )
                else:
                    print('Checkpoint {0} not found.'.format(file_checkpoint))
            else:
                print('Taking the last checkpoint...')
                #Restore the session from checkpoint
                self.sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/'+self.checkpoint_dir+'/checkpoint'))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        out_posterior = np.zeros_like(self.images_test_clipped[0:,:,:,:],dtype='float32')
        out_prior = np.zeros_like(self.images_test_clipped[0:,:,:,:],dtype='float32')
        out_alpha = np.zeros_like(self.images_test_clipped[0:,:,:,:],dtype='float32')
        out_beta = np.zeros_like(self.images_test_clipped[0:,:,:,:],dtype='float32')
        
        for i in range(np.shape(self.images_test_clipped[0:,:,:,:])[0]):
            out_posterior1, out_prior1, out_alpha1,out_beta1,out_L = self.sess.run([self.X_posterior,self.X_prior,self.alpha,self.beta,self.L_holder],feed_dict={self.X_noisy:self.images_test_clipped[i:i+1,:,:,:],
                                                                                                              self.L_holder:self.L,
                                                                                                              self.is_train:False,
                                                                                                              self.shift:1})
            out_posterior[i:i+1,:,:,:] = out_posterior1
            out_prior[i:i+1,:,:,:] = out_prior1
            out_alpha[i:i+1,:,:,:] = out_alpha1
            out_beta[i:i+1,:,:,:] = out_beta1
        
        #denormalize
        out_posterior *= self.norm
        out_prior *= self.norm
       
        #copy point targets back > clip
        mask_outliers = np.logical_xor(self.mask_test,True)
        self.mask_outliers=mask_outliers
        out_posterior[mask_outliers] = self.images_test[mask_outliers]
        out_prior[mask_outliers] = self.images_test[mask_outliers]
        
        dir_test = 'test'
        safe_mkdir(dir_test)
        dir_final = os.path.join(dir_test,self.checkpoint_dir)
        safe_mkdir(dir_final)
        
        step=self.gstep.eval(session=self.sess)
        sio.savemat(os.path.join(dir_final,'{0}_{1}.mat').format(self.checkpoint_dir,step), {'posterior':out_posterior[:,:,:,0],
                                                                       'prior':out_prior[:,:,:,0],
                                                                        'alpha': out_alpha[:,:,:,0],
                                                                        'beta': out_beta[:,:,:,0],                      
                                                                       'noisy':self.images_test[:,:,:,0],
                                                                       'L':out_L})

    
    def predict(self,img):
        '''
        Parameters
        --------------
        
        imgs: list(array(float))
        '''
        
        indexes = np.where(img > self.clip)
        mask = np.ones_like(img,dtype=np.bool)
        mask[indexes] = False
    
        
        #Clipping high backscattering
        medians = np.median(img,axis=[1,2],keepdims=True)
        img_clipped = np.where(img > self.clip, medians, img)
        #Normalization 
        img_clipped = (img_clipped.astype(np.float32)) / self.norm
     
        
        clean_img = self.sess.run(self.X_posterior,feed_dict={self.X_noisy:img_clipped,
                                                 self.L_holder:self.L,
                                                 self.shift:1,
                                                 self.is_train:False}
                         )
        clean_img*= self.norm
        #copy point targets back > clip
        mask_outliers = np.logical_xor(mask,True)
        clean_img[mask_outliers] = img[mask_outliers]
        
        return clean_img
    
    def load_weights(self):
        saver = tf.train.Saver(max_to_keep=None)
        #LOADING froms checkpoint
        if not self.loaded_weights:
            self.sess.run(tf.global_variables_initializer())
            
            if self.file_checkpoint:
                if os.path.isfile('{0}.index'.format(self.file_checkpoint)):
                    print('Taking the specified checkpoint...')
                    saver.restore(self.sess,self.file_checkpoint )
                    self.loaded_weights = True
                else:
                    print('Checkpoint {0} not found.'.format(self.file_checkpoint))
            else:
                print('Taking the last checkpoint...')
                #Restore the session from checkpoint
                ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/'+self.checkpoint_dir+'/checkpoint'))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
                    self.loaded_weights = True
        else:
            print('Model weights already loaded')
            
        return saver
    
    def build(self):
        '''
        Build the computation graph
        '''
        
        self.get_data()
        self.build_inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

