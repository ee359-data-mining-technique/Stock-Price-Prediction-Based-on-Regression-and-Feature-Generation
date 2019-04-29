import tensorflow as tf
import numpy as np

class CNN_Model:
    
    def __init__(self):
        self.time_steps = 10
        self.element_size = 20
        # self.layer_size = [16, 32, 16]
        self.learning_rate = 0.005

    def build_model(self):
        with tf.variable_scope("Inputs"):
            self._inputs = tf.placeholder(
                tf.float32, 
                shape=[None, self.time_steps, self.element_size, 1]
            )
            self.y_orig = tf.placeholder(
                tf.float32,
                shape=[None, 1]
            )

        with tf.variable_scope("Conv1", reuse = tf.AUTO_REUSE):
            current = self.batch_activ_conv(self._inputs, 1, 16, 3)
            current = self.maxpool2d(current, k=2)
        with tf.variable_scope("Conv2", reuse = tf.AUTO_REUSE):
            current = self.batch_activ_conv(current, 16, 32, 3)
            current = self.maxpool2d(current, k=2)
            current = tf.reshape(current, [-1, 320])
        with tf.variable_scope("FC1", reuse = tf.AUTO_REUSE):
            current = self.batch_activ_fc(current, 320, 32)
        
        with tf.variable_scope("outputs"):
            Wo = tf.Variable(
                tf.truncated_normal([32, 1], mean = 0, stddev = 0.01)
            )
            self.y_pred = tf.matmul(current, Wo)

        self.losses = tf.losses.mean_squared_error(
            labels = self.y_orig,
            predictions = self.y_pred
        )

        self.train_step = tf.train.AdamOptimizer(
            learning_rate = self.learning_rate
        ).minimize(self.losses)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print("=========Finish Building Model=========")

    def close_session(self):
        self.sess.close()

    def train_model(self, train_X, train_Y):
        if len(train_X.shape) == 3:
            train_X = np.expand_dims(train_X, axis=3)
        self.sess.run(
            self.train_step,
            feed_dict = {
                self._inputs: train_X,
                self.y_orig: train_Y
            }
        )
        y_pred, mse_loss = self.sess.run(
            [self.y_pred, self.losses],
            feed_dict = {
                self._inputs: train_X,
                self.y_orig: train_Y
            }
        )
        return mse_loss

    def evaluate_model(self, test_X, test_Y):
        if len(test_X.shape) == 3:
            test_X = np.expand_dims(test_X, axis=3)
        mse_loss = self.sess.run(
            self.losses,
            feed_dict = {
                self._inputs: test_X,
                self.y_orig: test_Y
            }
        )
        return mse_loss

    '''
    Helper Functions
    '''
    def dense_layer(self, name, input_tensor, input_shape, output_shape):
        with tf.variable_scope(name):
            Wl = tf.Variable(
                tf.truncated_normal(
                    shape = [input_shape, output_shape],
                    mean = 0,
                    stddev = 0.01
                ),
                name = "weights"
            )
            bl = tf.Variable(
                tf.truncated_normal(
                    shape  = [output_shape],
                    mean = 0,
                    stddev = 0.01
                ),
                name = "bias"
            )
            return tf.nn.relu(tf.matmul(input_tensor, Wl) + bl)

    '''
    Helper Builder Functions: to build model more conveniently
    '''
    def weight_variable_msra(self, shape, name):
        return tf.get_variable(name = name, shape = shape, initializer = tf.contrib.layers.variance_scaling_initializer(), trainable=False)

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(name = name, shape = shape, initializer = tf.contrib.layers.xavier_initializer(), trainable=False)

    def bias_variable(self, shape, name = 'bias'):
        initial = tf.constant(0.0, shape = shape)
        return tf.get_variable(name = name, initializer = initial, trainable=False)

    def gate_variable(self, length, name = 'gate'):
        initial = tf.constant([1.0] * length)
        v = tf.get_variable(name = name, initializer = initial)
        self.AllGateVariables[v.name] = v
        v = tf.abs(v)
        v = v - tf.constant([0.01]*length)
        v = tf.nn.relu(v)
        self.AllGateVariableValues.append(v)
        return v

    def conv2d(self, input, in_features, out_features, kernel_size, with_bias=False):
        W = self.weight_variable_msra([ kernel_size, kernel_size, in_features, out_features ], name = 'kernel')
        conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
        if with_bias:
            return conv + self.bias_variable([ out_features ])
        return conv

    def batch_activ_conv(self, current, in_features, out_features, kernel_size, is_training=1, keep_prob=1.0):
        with tf.variable_scope("composite_function", reuse = tf.AUTO_REUSE):
            current = self.conv2d(current, in_features, out_features, kernel_size)
            current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None, trainable=False)
            # convValues.append(current)
            current = tf.nn.relu(current)
            #current = tf.nn.dropout(current, keep_prob)
        return current

    def batch_activ_fc(self, current, in_features, out_features, is_training=1):
        Wfc = self.weight_variable_xavier([ in_features, out_features ], name = 'W')
        bfc = self.bias_variable([ out_features ])
        current = tf.matmul(current, Wfc) + bfc
        current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None, trainable=False)
        current = tf.nn.relu(current)
        return current

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                            padding='VALID')