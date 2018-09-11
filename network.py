import tensorflow as tf
import numpy as np

def conv(inputs, w, b):
    return tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME") + b

def relu(inputs):
    return tf.nn.relu(inputs)

def max_pooling(inputs):
    return tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

def vggnet(inputs):
    inputs = tf.reverse(inputs, [-1]) - np.array([103.939, 116.779, 123.68])
    para = np.load("./vgg_para//vgg19.npy", encoding="latin1").item()
    inputs = relu(conv(inputs, para["conv1_1"][0], para["conv1_1"][1]))
    inputs = relu(conv(inputs, para["conv1_2"][0], para["conv1_2"][1]))
    inputs = max_pooling(inputs)
    inputs = relu(conv(inputs, para["conv2_1"][0], para["conv2_1"][1]))
    inputs = relu(conv(inputs, para["conv2_2"][0], para["conv2_2"][1]))
    inputs = max_pooling(inputs)
    inputs = relu(conv(inputs, para["conv3_1"][0], para["conv3_1"][1]))
    phi = tf.layers.flatten(inputs)
    inputs = relu(conv(inputs, para["conv3_2"][0], para["conv3_2"][1]))
    inputs = relu(conv(inputs, para["conv3_3"][0], para["conv3_3"][1]))
    inputs = relu(conv(inputs, para["conv3_4"][0], para["conv3_4"][1]))
    inputs = max_pooling(inputs)
    inputs = relu(conv(inputs, para["conv4_1"][0], para["conv4_1"][1]))
    phi = tf.concat([phi, tf.layers.flatten(inputs)], 1)
    inputs = relu(conv(inputs, para["conv4_2"][0], para["conv4_2"][1]))
    inputs = relu(conv(inputs, para["conv4_3"][0], para["conv4_3"][1]))
    inputs = relu(conv(inputs, para["conv4_4"][0], para["conv4_4"][1]))
    inputs = max_pooling(inputs)
    inputs = relu(conv(inputs, para["conv5_1"][0], para["conv5_1"][1]))
    phi = tf.concat([phi, tf.layers.flatten(inputs)], 1)
    return phi



if __name__ == "__main__":
    inputs = tf.placeholder(tf.float32, [None, 128, 128, 3])
    vggnet(inputs)