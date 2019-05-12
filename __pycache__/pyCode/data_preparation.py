import tensorflow as tf
from tensorflow import keras
import numpy as np

from pyCode.fourierTransform import fft_transform, data_transform

N_CLASSES = 6
N_HIDDEN_UNITS = 64
N_TIME_STEPS=0 ## количество примеров//данных НЕОБХОДИМО УЗНАТЬ И ПОДСТАВИТЬ КОЛИЧЕСТВО ДАННЫХ ДЛЯ ОБУЧЕНИЯ


def create_segments(file_obj):
    getData=data_transform(file_obj)
    freq_data=fft_transform(getData)


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
    X = inputs
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

