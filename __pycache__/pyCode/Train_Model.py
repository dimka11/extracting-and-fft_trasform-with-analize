from typing import Dict, List, Any

import tensorflow as tf

from pyCode.data_preparation import optimizer, pred_softmax, accuracy, loss, x, y

N_EPOCHS = 50
BATCH_SIZE = 1024

history = dict(train_loss=[], train_acc=[],test_loss=[],test_acc=[])

def train_model():
    sess =tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    train_count = len(x_train)### Подставляется тренировочный массив данных
                            #### Необходимо дооопределить тренировочные и тестовые входные данные!!!!!!!!!!!

    for i in range(1, N_EPOCHS + 1):
        for start, end in zip(range(0, train_count, BATCH_SIZE),
                           range(BATCH_SIZE, train_count + 1 ,BATCH_SIZE)):
         sess.run(optimizer, feed_dict={x: x_train[start:end],y: y_train[start:end]})

        _, acc_train, loss_train = sess.run([pred_softmax, accuracy, loss], feed_dict={x: x_train, y: y_train})

        _, acc_test, loss_test = sess.run([pred_softmax, accuracy, loss], feed_dict={x: x_test, y: y_test})

        history['train_loss'].append(loss_train)
        history['train_acc'].append(acc_train)
        history['test_loss'].append(loss_test)
        history['test_acc'].append(acc_test)

        if i != 1 and i % 10 != 0:
             continue

        print(f'epoch: {i} test accuracy: {acc_test} loss: {loss_test}')

        predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={x: x_test, y: y_test})

        print()
        print(f'final results: accuracy: {acc_final} loss: {loss_final}')