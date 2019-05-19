import pandas as pd

import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split

RANDOM_SEED = 42
N_CLASSES = 4
N_FEATURES=1
N_HIDDEN_UNITS = 64
N_TIME_STEPS=71 ## количество примеров//данных НЕОБХОДИМО УЗНАТЬ И ПОДСТАВИТЬ КОЛИЧЕСТВО ДАННЫХ ДЛЯ ОБУЧЕНИЯ

def create_segments():
     with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\pyCode\Data.csv", 'r') as f_obj:
        segments = []
        columns = ['label','frequencies']
        df = pd.read_csv(f_obj, header=None, names=columns)
        for i in range(0,len(df)):
            seg=df['frequencies']
            segments.append(np.float32(seg[i]))
        reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, N_TIME_STEPS)
        return np.array(reshaped_segments)

def create_labels():
         with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\pyCode\Data.csv", 'r') as f_obj:
            labels = []
            columns = ['label', 'frequencies']
            df = pd.read_csv(f_obj, header=None, names=columns)
            for i in range(0, len(df)):
                label = df['label']
                labels.append(label[i])
            labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
            return (labels)

segments=create_segments()
labels=create_labels()

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