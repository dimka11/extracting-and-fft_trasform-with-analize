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
N_TIME_STEPS=500 ## количество примеров//данных НЕОБХОДИМО УЗНАТЬ И ПОДСТАВИТЬ КОЛИЧЕСТВО ДАННЫХ ДЛЯ ОБУЧЕНИЯ

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
    lendat = len(freq_Of_ShapesDown) + len(runArray) + len(freq_Of_ShapesUp) + len(walking)+len(freq_Of_Stand)

    segments=0
    labels=0
    for i in range(0, lendat):
        segments=make_one_DataArray(freq_Of_ShapesDown,runArray,freq_Of_ShapesUp,walking,freq_Of_Stand)
        labels=make_one_DataArray(labelDown,labelRun,labelUp,labelWalk,labelStand)
    print(np.array(segments).shape)
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, N_TIME_STEPS)
    labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
    return (np.array(reshaped_segments),labels)

segments,labels=create_segments()

X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2, random_state=RANDOM_SEED)

def create_model(inputs):
     W = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN_UNITS])),####Создание весов
        'output': tf.Variable(tf.random_normal([N_HIDDEN_UNITS, N_CLASSES]))
     }
     biases = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN_UNITS], mean=1.0)),
        'output': tf.Variable(tf.random_normal([N_CLASSES]))
        }

        ##создание первого слоя с использованием функциии релу - элемент нелинейности
     X = tf.transpose(inputs, [1, 0])
     X = tf.reshape(X, [-1, N_FEATURES])
     hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
     hidden = tf.split(hidden, N_TIME_STEPS, 0)
     ##Создание второго слоя

     lstm_layers = [tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(2)]
     lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)

     outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)
     lstm_last_output = outputs[-1]
     return tf.matmul(lstm_last_output, W['output']) + biases['output']

##Создание наполнения для модели
## заполнение плейсхолдеров- те места, куда будут подставляться значения входных-выходных переменных
x = tf.placeholder(tf.float32, [None, N_TIME_STEPS], name="input")###хранит в себе данные по ускорению
y = tf.placeholder(tf.float32, [None, N_CLASSES])## Хранит в себе данные по виду деятельности


##Создание входного тензора
pred_Y = create_model(x)
pred_softmax = tf.nn.softmax(pred_Y, name="y_")

##Определение функции потерь
L2_LOSS = 0.0015

l2 = L2_LOSS * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred_Y, labels = y)) + l2

##Определение оптимизатора

LEARNING_RATE = 0.0025 ##Коэффициент обучения, скорость обучения
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)##Оптимизатор, использует алгоритм Адама???!!!!
correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))


if  __name__  ==  " __main__ " :

    N_EPOCHS = 50
    BATCH_SIZE = 1024
    saver = tf.train.Saver()

    history = dict(train_loss=[], train_acc=[],test_loss=[],test_acc=[])

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    train_count = len(X_train)

    for i in range(1, N_EPOCHS + 1):
     for start, end in zip(range(0, train_count, BATCH_SIZE),
                          range(BATCH_SIZE, train_count + 1, BATCH_SIZE)):
        sess.run(optimizer, feed_dict={x: X_train[start:end],
                                       y: y_train[start:end]})

     _, acc_train, loss_train = sess.run([pred_softmax, accuracy, loss], feed_dict={
        x: X_train, y: y_train})

     _, acc_test, loss_test = sess.run([pred_softmax, accuracy, loss], feed_dict={
        x: X_test, y: y_test})

     history['train_loss'].append(loss_train)
     history['train_acc'].append(acc_train)
     history['test_loss'].append(loss_test)
     history['test_acc'].append(acc_test)

     if i != 1 and i % 10 != 0:
        continue

     print(f'epoch: {i} test accuracy: {acc_test} loss: {loss_test}')

    predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={x: X_test, y: y_test})

    print()
    print(f'final results: accuracy: {acc_final} loss: {loss_final}')