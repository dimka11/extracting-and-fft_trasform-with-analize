import pandas as pd

import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split

from pyCode.fourierTransform import data_transform, fft_transform
from pyCode.get_Data import  make_one_DataArray, make_array_Labels

RANDOM_SEED = 42
N_CLASSES = 5
N_FEATURES=1
N_HIDDEN_UNITS = 64
N_TIME_STEPS=450 ## количество примеров//данных НЕОБХОДИМО УЗНАТЬ И ПОДСТАВИТЬ КОЛИЧЕСТВО ДАННЫХ ДЛЯ ОБУЧЕНИЯ


batch_size = 10
kernel_size = 30
depth = 20
num_hidden = 50

learning_rate = 0.0025
training_epochs = 20

cpath = os.path.dirname(__file__)  # pycode folder (should folder where script is run)
dpath = cpath + "/../DATA/"  # path to data that above pycode folder


def create_segments():
    with open(dpath + "Down.csv", 'r') as f_obj:
        data = data_transform(f_obj)
        freq_Of_ShapesDown = fft_transform(data)
        labelDown = make_array_Labels('Down',len(freq_Of_ShapesDown))
    with open(dpath + "Run(soft).csv", 'r') as f_obj2:
        data1 = data_transform(f_obj2)
        freq_Of_ShapesRun = fft_transform(data1)
    with open(dpath + "Run9.csv", 'r') as f_obj3:
        data2 = data_transform(f_obj3)
        freq_Of_ShapesRun2 = fft_transform(data2)
        runArray = make_one_DataArray(freq_Of_ShapesRun, freq_Of_ShapesRun2)
        labelRun = make_array_Labels('Run', len(runArray))
    with open(dpath + "Up.csv", 'r') as f_obj4:
        data3 = data_transform(f_obj4)
        freq_Of_ShapesUp = fft_transform(data3)
        labelUp = make_array_Labels('Up', len(freq_Of_ShapesUp))
    with open(dpath + "Walk(soft).csv", 'r') as f_obj5:
        data4 = data_transform(f_obj5)
        freq_Of_Shapes4 = fft_transform(data4)
    with open(dpath + "Walking7.csv", 'r') as f_obj8:
        data7 = data_transform(f_obj8)
        freq_Of_Shapes7 = fft_transform(data7)
    with open(dpath + "Walking8.csv", 'r') as f_obj9:
        data8 = data_transform(f_obj9)
        freq_Of_Shapes8 = fft_transform(data8)
        walking = make_one_DataArray(freq_Of_Shapes4, freq_Of_Shapes7, freq_Of_Shapes8)
        labelWalk = make_array_Labels('Walk', len(walking))
    with open(dpath + "Standing.csv", 'r') as f_obj10:
        data9 = data_transform(f_obj10)
        freq_Of_Stand = fft_transform(data9)
        labelStand = make_array_Labels('Standing', len(freq_Of_Stand))

    segments = make_one_DataArray(freq_Of_ShapesDown, runArray, freq_Of_ShapesUp, walking, freq_Of_Stand)
    labels = make_one_DataArray(labelDown, labelRun, labelUp, labelWalk, labelStand)
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
    labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
    return (np.array(reshaped_segments), labels)

segments,labels=create_segments()

X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2, random_state=RANDOM_SEED)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x, W, [1, 1, 1, 1], padding='VALID')


def apply_depthwise_conv(x, kernel_size, N_FEATURES, depth):
    weights = weight_variable([1, kernel_size, N_FEATURES, depth])
    biases = bias_variable([depth * N_FEATURES])
    return tf.nn.relu(tf.add(depthwise_conv2d(x, weights), biases))


def apply_max_pool(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                          strides=[1, 1, stride_size, 1], padding='VALID')
total_batches = X_train.shape[0] // batch_size

X = tf.placeholder(tf.float32, shape=[None, N_TIME_STEPS, N_FEATURES], name="x_input")
X_reshaped = tf.reshape(X, [-1, 1, N_TIME_STEPS, N_FEATURES])
Y = tf.placeholder(tf.float32, shape=[None, N_CLASSES])

c = apply_depthwise_conv(X_reshaped, kernel_size, N_FEATURES, depth)
p = apply_max_pool(c, 20, 2)
c = apply_depthwise_conv(p, 6, depth*N_FEATURES, depth//10)

shape = c.get_shape().as_list()
c_flat = tf.reshape(c, [-1, shape[1] * shape[2] * shape[3]])

f_weights_l1 = weight_variable([shape[1] * shape[2] * depth * N_FEATURES * (depth//10), num_hidden])
f_biases_l1 = bias_variable([num_hidden])
f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1), f_biases_l1))

out_weights = weight_variable([num_hidden, N_CLASSES])
out_biases = bias_variable([N_CLASSES])
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases, name="labels_output")

loss = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cost_history = np.empty(shape=[1], dtype=float)

saver = tf.train.Saver()

if __name__ == '__main__':

    with tf.Session() as session:
        # tf.global_variables_initializer().run()
        session.run(tf.global_variables_initializer())
        # save the graph
        tf.train.write_graph(session.graph_def, '.', 'session.pb', False)

        for epoch in range(training_epochs):
            for b in range(total_batches):
                offset = (b * batch_size) % (y_train.shape[0] - batch_size)
                batch_x = X_train[offset:(offset + batch_size), :, :]
                batch_y = y_train[offset:(offset + batch_size), :]
                _, c = session.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
                cost_history = np.append(cost_history, c)
            print("Epoch: ", epoch, " Training Loss: ", c, " Training Accuracy: ",
                 session.run(accuracy, feed_dict={X: X_train, Y: y_train}))

        print("Testing Accuracy:", session.run(accuracy, feed_dict={X: X_test, Y: y_test}))
        saver.save(session, './session.ckpt')