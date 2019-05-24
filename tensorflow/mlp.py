""" Neural Network.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

Links:
[MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
Sligtly modified by Minh Duc Tran
"""
from __future__ import print_function

import tensorflow as tf
import time
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import sys
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

print(mnist.train.images)

# Parameters
learning_rate = 0.1
batch_size = int(sys.argv[1])

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)


# Define the neural network
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    logits = neural_net(features)

    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                                 predictions=pred_classes)

    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': mnist.train.images},
                                                    y=mnist.train.labels,
                                                    batch_size=batch_size,
                                                    num_epochs=1,
                                                    shuffle=False)

runs=11
times = []

test_input, test_labels = mnist.train.next_batch(10000)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': test_input},
                                                   y=test_labels,
                                                   shuffle=False)
class Timer(tf.train.SessionRunHook):
    def __init__(self):
        self.runs = 0

    def before_run(self, run_context):
        self.run_start = time.time()

    def after_run(self, run_context, run_values):
        self.runs += time.time() - self.run_start

    def runtime(self):
        return self.runs

for i in range(runs):
    start = time.time()
    hook = Timer()
    model.train(train_input_fn, hooks=[hook])
    e = model.evaluate(eval_input_fn)
    end = time.time()
    t = hook.runtime()
    print("Runtime: {}".format(t))
    print("Accuracy: ", e['accuracy'])
    times += [t]

print('batch_size', batch_size)
print('Average training time', sum(times[1:]) / (runs-1))
