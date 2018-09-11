import tensorflow as tf
import tensorflow.contrib as contrib
from network import vggnet
from config import *
from PIL import Image
import numpy as np
import scipy.misc as misc
import os

class DFI:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [1, 200, 200, 3])
        self.w = tf.placeholder(tf.float32, [1, None])
        self.z = tf.get_variable("z", [1, 200, 200, 3], initializer=tf.random_normal_initializer(stddev=0.02))
        self.alpha = ALPHA# BETA / tf.reduce_mean(tf.square(self.w))
        self.phi_x = vggnet(self.x)
        self.phi_z_rec = vggnet(self.z)
        self.phi_z = self.phi_x + self.alpha * self.w
        self.R_tv = tf.reduce_sum(tf.pow(tf.square(self.z[:, :-1, :-1, :] - self.z[:, :-1, 1:, :]) +\
                    tf.square(self.z[:, :-1, :-1, :] - self.z[:, 1:, :-1, :]), 2.0/2))
        self.loss = 0.5 * tf.reduce_sum(tf.square(self.phi_z - self.phi_z_rec)) + LAMBDA * self.R_tv
        self.Opt = contrib.opt.ScipyOptimizerInterface(self.loss, method="L-BFGS-B", options={'maxiter': 300, 'disp': 0})#Optimizer of L-BFGS
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        def mapping(img):
            return 255.0 * (img - np.min(img)) / (np.max(img) - np.min(img))
        #input content image
        input_img = np.array(Image.open("./input_img//2.jpg"))
        w = self.attribute_vector("./source//", "./target//")#Calculate the attribute vector w.
        H = input_img.shape[0]
        W = input_img.shape[1]
        input_img = misc.imresize(input_img[H // 2 - 100:H // 2 + 100, W // 2 - 100:W // 2 + 100, :3], [200, 200])[np.newaxis, :, :, :]#Crop the center image from the raw image
        Image.fromarray(np.uint8(input_img[0, :, :, :])).save("./input_img//croped_img.jpg")#Save the croped image
        self.sess.run(tf.assign(self.z, input_img))#Initialize z with input_image
        for step in range(10):
            self.Opt.minimize(self.sess, feed_dict={self.x: input_img, self.w: w})
            if step % 1 == 0:
                [img, loss] = self.sess.run([self.z, self.loss],feed_dict={self.x: input_img, self.w: w})
                print("step: %d, loss: %f"%(step, loss))
                Image.fromarray(np.uint8(mapping(img[0, :, :, :]))).save("./results//result.jpg")


    def attribute_vector(self, source_path, target_path):
        source_filenames = os.listdir(source_path)
        phi_s = 0
        for filename in source_filenames:
            img = np.array(Image.open(source_path+filename))
            h = img.shape[0]
            w = img.shape[1]
            img = misc.imresize(img[h//2-100:h//2+100, w//2-100:w//2+100, :3], [200, 200])[np.newaxis, :, :, :]
            phi_s += self.sess.run(self.phi_x, feed_dict={self.x: img})
        phi_s_bar = phi_s / source_filenames.__len__()
        target_filenames = os.listdir(target_path)
        phi_t = 0
        for filename in target_filenames:
            img = np.array(Image.open(target_path + filename))
            h = img.shape[0]
            w = img.shape[1]
            img = misc.imresize(img[h // 2 - 100:h // 2 + 100, w // 2 - 100:w // 2 + 100, :3], [200, 200])[np.newaxis, :, :, :]
            phi_t += self.sess.run(self.phi_x, feed_dict={self.x: img})
        phi_t_bar = phi_t / target_filenames.__len__()
        w = phi_t_bar - phi_s_bar
        return w


if __name__ == "__main__":
    dfi = DFI()
    dfi.train()