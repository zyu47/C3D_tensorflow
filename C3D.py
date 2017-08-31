import tensorflow as tf
import numpy as np
import cv2
import os
from os.path import join


class C3D:
    def __init__(self, logs_path, num_classes=101):
        self.logs_path = logs_path
        self.num_classes = num_classes
        self.learning_rate = tf.placeholder(tf.float32, [])
        self.regularization = 5e-4


        self.x = tf.placeholder(tf.float32, [None, 16, 112, 112, 3])
        self.y_ = tf.placeholder(tf.float32, [None, self.num_classes])
        self.keep_prob = tf.placeholder(tf.float32)


        # Conv1a
        self.W1a = tf.get_variable("W1a", shape=[3, 3, 3, 3, 64], initializer=tf.contrib.layers.xavier_initializer())  # tf.Variable(tf.truncated_normal([3,3,3,64], stddev=0.1))
        b1a = tf.Variable(tf.constant(0.01, shape=[64]))
        self.conv1a = tf.nn.conv3d(self.x, self.W1a, strides=[1, 1, 1, 1, 1], padding='SAME')
        conv1a_relu = tf.nn.relu(self.conv1a + b1a)

        # Max pooling layer 1
        max_pool_1 = tf.nn.max_pool3d(conv1a_relu, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='SAME')

        #conv2a
        W2a = tf.get_variable("W2a", shape=[3, 3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())  # tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.1))
        b2a = tf.Variable(tf.constant(0.01, shape=[128]))
        conv2a = tf.nn.conv3d(max_pool_1, W2a, strides=[1, 1, 1, 1, 1], padding='SAME')
        conv2a_relu = tf.nn.relu(conv2a + b2a)

        # Max pooling layer 2
        max_pool_2 = tf.nn.max_pool3d(conv2a_relu, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

        #conv3a
        W3a = tf.get_variable("W3a", shape=[3, 3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())  # tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.1))
        b3a = tf.Variable(tf.constant(0.01, shape=[256]))
        conv3a = tf.nn.conv3d(max_pool_2, W3a, strides=[1, 1, 1, 1, 1], padding='SAME')
        conv3a_relu = tf.nn.relu(conv3a + b3a)

        #conv3b
        W3b = tf.get_variable("W3b", shape=[3, 3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())  # tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.1))
        b3b = tf.Variable(tf.constant(0.01, shape=[256]))
        conv3b = tf.nn.conv3d(conv3a_relu, W3b, strides=[1, 1, 1, 1, 1], padding='SAME')
        conv3b_relu = tf.nn.relu(conv3b + b3b)


        # Max pooling layer 3
        max_pool_3 = tf.nn.max_pool3d(conv3b_relu, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

        #conv4a
        W4a = tf.get_variable("W4a", shape=[3, 3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer())  # tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.1))
        b4a = tf.Variable(tf.constant(0.01, shape=[512]))
        conv4a = tf.nn.conv3d(max_pool_3, W4a, strides=[1, 1, 1, 1, 1], padding='SAME')
        conv4a_relu = tf.nn.relu(conv4a + b4a)

        # conv4b
        W4b = tf.get_variable("W4b", shape=[3, 3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())  # tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.1))
        b4b = tf.Variable(tf.constant(0.01, shape=[512]))
        conv4b = tf.nn.conv3d(conv4a_relu, W4b, strides=[1, 1, 1, 1, 1], padding='SAME')
        conv4b_relu = tf.nn.relu(conv4b + b4b)

        # Max pooling layer 4
        max_pool_4 = tf.nn.max_pool3d(conv4b_relu, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

        # conv5a
        W5a = tf.get_variable("W5a", shape=[3, 3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())  # tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.1))
        b5a = tf.Variable(tf.constant(0.01, shape=[512]))
        conv5a = tf.nn.conv3d(max_pool_4, W5a, strides=[1, 1, 1, 1, 1], padding='SAME')
        conv5a_relu = tf.nn.relu(conv5a + b5a)

        # conv5b
        W5b = tf.get_variable("W5b", shape=[3, 3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())  # tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.1))
        b5b = tf.Variable(tf.constant(0.01, shape=[512]))
        conv5b = tf.nn.conv3d(conv5a_relu, W5b, strides=[1, 1, 1, 1, 1], padding='SAME')
        conv5b_relu = tf.nn.relu(conv5b + b5b)

        # Max pooling layer 5
        max_pool_5 = tf.nn.max_pool3d(conv5b_relu, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


        #Restore convolutional layer features
        self.saver_conv = tf.train.Saver(tf.global_variables())


        # Fully connected layer 1
        W_fc1 = tf.get_variable("W_fc1", shape=[512*4*4, 2048], initializer=tf.contrib.layers.xavier_initializer())  # tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.01, shape=[2048]))
        max_pool5_flat = tf.reshape(max_pool_5, [-1, 4 * 4 * 512])

        h_fc1 = tf.nn.relu(tf.nn.xw_plus_b(max_pool5_flat, W_fc1, b_fc1))
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)


        # Fully connected layer 2
        W_fc2 = tf.get_variable("W_fc2", shape=[2048, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())  # tf.Variable(tf.truncated_normal([4096, 200], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(0.01, shape=[self.num_classes]))

        h_fc2 = tf.nn.xw_plus_b(h_fc1_drop, W_fc2, b_fc2)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=h_fc2))

        reg_loss = self.regularization * (tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_fc1)  +
                                               tf.nn.l2_loss(W5b) + tf.nn.l2_loss(W5a) + tf.nn.l2_loss(W4b) +
                                               tf.nn.l2_loss(W4a)+ tf.nn.l2_loss(W3b)+ tf.nn.l2_loss(W3a) +
                                               tf.nn.l2_loss(W2a) + tf.nn.l2_loss(self.W1a))


        self.loss = cross_entropy + reg_loss

        #self.train_step = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9).minimize(self.loss)
        self.train_step = tf.train.AdagradOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

        correct_prediction = tf.cast(tf.equal(tf.argmax(h_fc2, 1), tf.argmax(self.y_, 1)), tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        try:
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config= config)
        except:
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config= config)

        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('loss', self.loss)
        self.summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.logs_path + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.logs_path + '/test', self.sess.graph)

        # model saver
        self.saver = tf.train.Saver()


        # initialisation
        self.sess.run(tf.global_variables_initializer())







