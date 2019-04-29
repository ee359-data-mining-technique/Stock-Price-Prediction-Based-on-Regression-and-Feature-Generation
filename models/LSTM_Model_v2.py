import tensorflow as tf

class LSTM_Model:
    
    def __init__(self):
        self.time_steps = 10
        self.element_size = 20
        self.LSTM_Layer_Sizes = [128, 256]
        self.learning_rate = 0.001

    def build_model(self):
        with tf.variable_scope("Inputs"):
            self._inputs = tf.placeholder(
                tf.float32, 
                shape=[None, self.time_steps, self.element_size]
            )
            self.y_orig = tf.placeholder(
                tf.float32,
                shape=[None, 1]
            )

        with tf.variable_scope("LSTM_Layers"):
            # create 2 LSTMCells
            rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in self.LSTM_Layer_Sizes]

            # create a RNN cell composed sequentially of a number of RNNCells
            multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

            # 'outputs' is a tensor of shape [batch_size, max_time, 256]
            # 'state' is a N-tuple where N is the number of LSTMCells containing a
            # tf.contrib.rnn.LSTMStateTuple for each cell
            outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                            inputs=self._inputs,
                                            dtype=tf.float32)
            lstm_outputs = outputs[:, -1, :]

        with tf.variable_scope("Dense_Layer_1"):
            Wl = tf.Variable(
                tf.truncated_normal(
                    shape = [self.LSTM_Layer_Sizes[-1], 100],
                    mean = 0,
                    stddev = 0.01
                ),
                name = "weights"
            )
            bl = tf.Variable(
                tf.truncated_normal(
                    shape  = [100],
                    mean = 0,
                    stddev = 0.01
                ),
                name = "bias"
            )
            linear_outputs = tf.matmul(lstm_outputs, Wl) + bl
        
        with tf.variable_scope("outputs"):
            Wo = tf.Variable(
                tf.truncated_normal([100, 1], mean = 0, stddev = 0.01)
            )
            y_pred = tf.matmul(linear_outputs, Wo)

        self.losses = tf.losses.mean_squared_error(
            labels = self.y_orig,
            predictions = y_pred
        )

        self.train_step = tf.train.AdamOptimizer(
            learning_rate = self.learning_rate
        ).minimize(self.losses)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        print("=========Finish Building Model=========")

    def close_session(self):
        self.sess.close()

    def train_model(self, train_X, train_Y):
        self.sess.run(
            self.train_step,
            feed_dict = {
                self._inputs: train_X,
                self.y_orig: train_Y
            }
        )
        mse_loss = self.sess.run(
            self.losses,
            feed_dict = {
                self._inputs: train_X,
                self.y_orig: train_Y
            }
        )
        return mse_loss

    def evaluate_model(self, test_X, test_Y):
        mse_loss = self.sess.run(
            self.losses,
            feed_dict = {
                self._inputs: test_X,
                self.y_orig: test_Y
            }
        )
        return mse_loss