import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class MnistModel():  # tf.Module
  def __init__(self):

    self.loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    self.test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    self.opt = tf.optimizers.Adam()

  def build(self):

    self.model = Sequential()

    self.model.add(Conv2D(28, (3, 3), input_shape=(28, 28, 1), padding='same'))  # train_imgs.shape[1:]
    self.model.add(Activation('relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    self.model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    self.model.add(Dense(64))

    self.model.add(Dense(9))
    # self.model.add(Activation('softmax'))

    return self.model

  def apply_softmax(self, logits):
    return tf.nn.softmax(logits)

  def train(self, train_data):

    for step, (img_batch, lbl_batch) in enumerate(train_data):

      with tf.GradientTape() as tape:
        logits = self.model(img_batch)
        softmx = self.apply_softmax(logits)
        xent = self.loss_fn(lbl_batch, softmx)

      varis = self.model.trainable_variables
      grads = tape.gradient(xent, varis)

      self.opt.apply_gradients(zip(grads, varis))

      self.train_acc_metric(lbl_batch, logits)

      if tf.equal(step % 500, 0):  # not step % 500:
        acc = self.train_acc_metric.result()
        tf.print("Loss: {} Accuracy: {}".format(xent, acc))
        self.train_acc_metric.reset_states()

  def test(self, test_data, temp=None):
    for img_batch, lbl_batch in test_data:
      logits = self.model(img_batch)
      if not temp is None:
        logits = tf.math.divide(logits, temp)
      pred = self.apply_softmax(logits)
      self.test_acc_metric(lbl_batch, pred)
    tf.print("Test acc: {}".format(self.test_acc_metric.result()))

  def save_model(self, path):
    self.model.save(path)

  def load_model(self, path):
    self.model = tf.keras.models.load_model(path)
    return self.model

  def get_max_pred_value(self, data, temp=None):
    logits = self.model(data)
    if not temp is None:
      logits = tf.math.divide(logits, temp)
    pred = self.apply_softmax(logits)
    return np.max(pred)
