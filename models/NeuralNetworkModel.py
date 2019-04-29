import tensorflow as tf
import numpy as np

class VanillaNetworkModel:
    
    def __init__(self):
        self.time_steps = 10
        self.element_size = 20
        self.layer_size = [16, 32, 16]
        self.learning_rate = 0.005

    def build_model(self):
        with tf.variable_scope("Inputs"):
            self._inputs = tf.placeholder(
                tf.float32, 
                shape=[None, self.time_steps*self.element_size]
            )
            self.y_orig = tf.placeholder(
                tf.float32,
                shape=[None, 1]
            )

        linear_layer_1 = self.dense_layer(
            "linear_layer_1",
            self._inputs,
            self.time_steps*self.element_size,
            self.layer_size[0]
        )
        linear_layer_2 = self.dense_layer(
            "linear_layer_2",
            linear_layer_1,
            self.layer_size[0],
            self.layer_size[1]
        )
        linear_layer_3 = self.dense_layer(
            "linear_layer_3",
            linear_layer_2,
            self.layer_size[1],
            self.layer_size[2]
        )
        
        with tf.variable_scope("outputs"):
            Wo = tf.Variable(
                tf.truncated_normal([self.layer_size[2], 1], mean = 0, stddev = 0.01)
            )
            self.y_pred = tf.matmul(linear_layer_3, Wo)

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
        if len(train_X.shape) > 2:
            train_X = train_X.reshape(train_X.shape[0], -1)
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
        if len(test_X.shape) > 2:
            test_X = test_X.reshape(test_X.shape[0], -1)
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