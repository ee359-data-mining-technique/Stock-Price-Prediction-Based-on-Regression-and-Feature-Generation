import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from tensorflow import keras
import tensorflow as tf

class Model():
    """A class for an building and inferencing an lstm model"""
    
    def __init__(self, name):
        self.model = keras.models.Sequential()
        self.name = name

    def save_model(self):
        self.model.save(self.name + ".h5")

    def load_model(self):
        self.model = keras.models.load_model("11042019-213012-e1" + ".h5")
        self.model.summary()

    def build_model(self):
        self.model.add(keras.layers.LSTM(
            units = 100, 
            input_shape = (10, 13), 
            return_sequences = True,
            kernel_regularizer=keras.regularizers.l2(0.001)
        ))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.LSTM(
            units = 100,
            return_sequences = True,
            kernel_regularizer=keras.regularizers.l2(0.001)
        ))
        self.model.add(keras.layers.LSTM(
            units = 100,
            return_sequences = False,
            kernel_regularizer=keras.regularizers.l2(0.001)
        ))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(
            100, 
            activation = "relu",
            kernel_regularizer=keras.regularizers.l2(0.001)
        ))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(1, activation = "linear"))

        self.model.compile(
            loss="mse", 
            optimizer="adam",
            metrics=['mse']
        )



    def train(self, x, y, epochs, batch_size, save_dir):
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
            keras.callbacks.ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)

    def predict_data(self, data):
        predicted = self.model.predict(data)
        return predicted