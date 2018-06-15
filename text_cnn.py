import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout and predictions
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        #self.predictions=tf.placeholder(tf.float32, [None, 1], name="predictions")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        """
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        """
        print "input"
        print vocab_size
        print self.embedded_chars_expanded.get_shape()
        filter_shape = [3, embedding_size, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(
            self.embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        print conv.get_shape().as_list()
        relu = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        print "shape of relu output is:"
        print  relu.get_shape().as_list()
        print embedding_size
        print num_filters
        filter_shape2 = [3, 1, 128, 16]
        W2 = tf.Variable(tf.truncated_normal(filter_shape2, stddev=0.1), name="W2")
        b2 = tf.Variable(tf.constant(0.1, shape=[16]), name="b2")
        conv2 = tf.nn.conv2d(
            relu,
            W2,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv2")
        print "conv layer 2 shape"
        print conv2.get_shape().as_list()
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")


        pool = tf.nn.max_pool(
            relu2,
            ksize=[1, sequence_length - 4, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        print "pool shape"
        print pool.get_shape().as_list()
        list_temp=pool.get_shape().as_list()
        """
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        """


        with tf.name_scope("dropout"):
            num_filters_total = 16

            self.h_pool_flat = tf.reshape(pool, [-1, num_filters_total])
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        print "h_drop dimensions"
        print self.h_drop.get_shape().as_list()


        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W3 = tf.get_variable(
                "W2",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b2")
            l2_loss += tf.nn.l2_loss(W3)
            l2_loss += tf.nn.l2_loss(b3)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W3, b3, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            #print("PREDICTION IS ".format(self.predictions))
           
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
