import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random


def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def conv_layer(x, W, b):
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x


def batch_norm_layer(x):
    mean, variance = tf.nn.moments(x, axes=[0])
    x = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=1e-8)
    return x


def max_pool_layer(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')


def fc_layer(x, W, b):
    return tf.add(tf.matmul(x, W), b)


def CNN_model(x):
    # define weights
    weights = {
        'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
        'wd1': tf.get_variable('W1', shape=(32*14*14,784), initializer = tf.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('W2', shape=(784,10), initializer=tf.contrib.layers.xavier_initializer()),
    }
    biases = {
        'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
        'bd1': tf.get_variable('B1', shape=(784), initializer=tf.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('B2', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
    }

    x = conv_layer(x, weights['wc1'], biases['bc1'])
    x = tf.nn.relu(x)
    x = batch_norm_layer(x)
    x = max_pool_layer(x)
    x = tf.reshape(x, [-1, weights['wd1'].get_shape().as_list()[0]])
    x = fc_layer(x, weights['wd1'], biases['bd1'])
    #dropout
    x = tf.nn.dropout(x, 0.5)

    x = tf.nn.relu(x)
    x = fc_layer(x, weights['out'], biases['out'])

    return tf.nn.softmax(x), weights, biases


def train(trainData,
          validData,
          testData,
          trainTarget,
          validTarget,
          testTarget,
          num_epochs,
          batch_size):

    x = tf.placeholder("float", [None, 28, 28, 1])
    target = tf.placeholder("float", [None, 10])

    prediction, weights, biases = CNN_model(x)

    # no reg
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=target))

    # with reg
#     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=target))
#     reg = (tf.nn.l2_loss(weights['wc1']) + tf.nn.l2_loss(weights['wd1']) +
#            tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(biases['bc1']) +
#            tf.nn.l2_loss(biases['bd1']) + tf.nn.l2_loss(biases['out']))
#     cost = cost + 0.5 * reg

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=target))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    # calculate accuracy
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tr_loss = []
    te_loss = []
    tr_accuracy = []
    te_accuracy = []

    num_batches = len(trainData)//batch_size

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        train_loss_epoch = []
        train_acc_epoch = []

        for epoch in range(num_epochs):
            #Shuffle
            combined = list(zip(trainData, trainTarget))
            random.shuffle(combined)
            trainData[:], trainTarget[:] = zip(*combined)

            for batch in range(num_batches):
                batch_x = trainData[batch*batch_size:min((batch+1)*batch_size,len(trainData))]
                batch_y = trainTarget[batch*batch_size:min((batch+1)*batch_size,len(trainData))]
                opt, loss, acc = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_x, target: batch_y})

                train_loss_epoch.append(loss)
                train_acc_epoch.append(acc)

            loss = sum(train_loss_epoch) / len(train_loss_epoch)
            acc = sum(train_acc_epoch) / len(train_acc_epoch)

            test_acc,test_loss = sess.run([accuracy,cost], feed_dict={x: testData,target : testTarget})
            print("Epoch", "{:3d}".format(epoch), "| acc", "{:.4f}".format(acc), "| loss", "{:.4f}".format(loss), "| test acc", "{:.4f}".format(test_acc), "| test loss", "{:.4f}".format(test_loss))

            tr_loss.append(float(loss))
            te_loss.append(float(test_loss))
            tr_accuracy.append(float(acc))
            te_accuracy.append(float(test_acc))

    return tr_loss, te_loss, tr_accuracy, te_accuracy


trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData, validData, testData = np.expand_dims(trainData, 3), np.expand_dims(validData, 3), np.expand_dims(testData, 3)
trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)


(train_loss, test_loss, train_accuracy, test_accuracy) = train(trainData,
                                                               validData,
                                                               testData,
                                                               trainTarget,
                                                               validTarget,
                                                               testTarget,
                                                               num_epochs=50,
                                                               batch_size=32)


plt.plot(train_loss, 'b', label='Training loss')
plt.plot(test_loss, 'r', label='Test loss')
plt.legend()
plt.figure()
plt.show()

plt.plot(train_accuracy, 'b', label='Training accuracy')
plt.plot(test_accuracy, 'r', label='Test accuracy')
plt.legend()
plt.figure()
plt.show()
