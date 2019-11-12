# prepared by Juan Egas

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

x_train, x_val, x_test, y_train, y_val, y_test = loadData()

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

y_train, y_val, y_test = convertOneHot(y_train, y_val, y_test)

def shuffle(trainData, trainTarget):
    np.random.seed(521)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target

def ReLU(X):
    """
    Apply ReLU activation to X
    ReLU(X) = max(X,0)
    Args: 
        X (np.array): input matrix
    Output: 
        X (np.array): ReLU(x)
    """
    X = X*(X>0)
    return X

def ReLU_prime(X):
    """
    Apply the derivative of ReLU activation to X
    ReLU_prime(X) = 1 if x>0 or = 0 if x <= 0
    Args: 
        X (np.array): input matrix
    Output: 
        x (np.array): ReLU_prime(X)
    """
    X = 1*(X>0)
    return X

def softmax(X):
    """
    Computes the softmax activation of each of the inputs
    Outputs a probability distribution of a list of potential outcomes
    softmax(x)i = e^xi/sum(e^xj) where j =1, 2, .., K classes
    Args: 
        x (np.array): input matrix (logits)
    Output: 
        probs (np.array): distribution of probabilities
    If input X is not a matrix but a vector expand one dim
    """
    # case for a vector 
    if X.ndim == 1: 
        X = np.expand_dims(X, axis = 0)
    # avoid overflowing by substracting max(x)
    
    Smat = np.exp(X - X.max(axis=1).reshape(X.shape[0],1)) 
    probs = Smat/Smat.sum(axis=1).reshape(X.shape[0],1)
    
    return probs

def averageCE(target, prediction):
    """
    Computes the average Cross-Entropy for the dataset
    averageCE = (1/N)sum(sum(target*log(pred)))
    Args: 
        target (np.array): A matrix with one-hot vectors as labels
        prediction (np.array): A matrix with softmax 
    Output: 
        averageCE (float): average Cross-Entropy
    """ 
    
    # number of samples
    N = target.shape[0]
    # applying log to prediction matrix
    log_pred = np.log(prediction)
    temp = np.sum(np.multiply(target, log_pred), axis=1)
    averageCE = -(1/N)*np.sum(temp)
    
    return averageCE

def computeLayer(X, W, b):
    """
    Computes W_t*X + b and outputs the 
    prediction for a given layer
    Args: 
        W (np.matrix): Weight matrix
        X (np.array): input vector
        b (np.array): bias vector
    Output: 
        pred (np.array): prediction for a given layer
    """ 
    pred = np.add(np.matmul(X,W), b)
    
    return pred

def gradCrossEntropy(target, output, gradType = 'unit'):
    """
    Computes gradient of the Cross-Entropy Loss
    Args: 
        target (np.array): A matrix with one-hot vectors as labels
        output (np.array): A matrix with logits
    Output: 
        gradCE (np.array): gradient of the output layer
    """ 
    N = target.shape[0]
    
    Sof = softmax(output)
    if gradType == 'unit':
        gradCE = np.subtract(Sof, target)
    elif gradType == 'average':
        gradCE = (1/N)*np.subtract(Sof, target)
        
    return gradCE

NN_architecture = [
                   {"input_dim": 28*28, "output_dim": 1000, "activation": "ReLU"},
                   {"input_dim": 1000, "output_dim": 10, "activation": "Softmax"}
                  ]

def initialize_layers(NN_architecture):
    np.random.seed(521)
    parameters = {}
    mean = 0.0
    
    for index, layer in enumerate(NN_architecture):
        index += 1
        layer_input_size = layer['input_dim']
        layer_output_size = layer['output_dim']
        std = np.sqrt(2/(layer_input_size + layer_output_size))
        parameters["W"+str(index)] = np.random.normal(mean, std, (layer_input_size, 
                                                    layer_output_size))
        parameters["b"+str(index)] = np.random.normal(mean, std, (1, layer_output_size))
    return parameters

def single_FeedForward(X, W, b, activation):
    S = computeLayer(X, W, b)

    if activation == 'Softmax':
        activation = softmax
    elif activation == 'ReLU':
        activation = ReLU
    else: 
        raise Exception("Activation function not supported")

    return activation(S), S

def forward_propagation(X, NN_architecture, parameters):
    node_curr = X
    mem = {}

    for index, layer in enumerate(NN_architecture):
          index = index + 1
          node_prev = node_curr
          activation = layer['activation']
          W_curr = parameters['W'+ str(index)] 
          b_curr = parameters['b' + str(index)]
          node_curr, S_curr = single_FeedForward(node_prev, W_curr, b_curr, activation)

          mem['X' + str(index)] = node_curr
          mem['S' + str(index)] = S_curr
    
    pred = node_curr
    output = S_curr 
    return pred, output, mem, parameters

def backward_propagation(X, target, pred, output, mem, parameters):
    gradients = {}
    N = X.shape[1]
    
    backward_first = gradCrossEntropy(target, output, gradType = 'average')
    Wo = parameters['W2']
    reluBack = ReLU_prime(mem['S1'])
    # average terms
    Sum_reluBack = np.sum(reluBack, axis=0).reshape(reluBack.shape[1],1)

    term1 = np.matmul(Wo,backward_first.T)
    back_second = np.multiply(term1,Sum_reluBack)
    
    grad_Wh = np.matmul(back_second, X)
    grad_bh =  np.sum(back_second, axis=1)
    gradients["dW1"] = grad_Wh.T
    gradients["db1"] = grad_bh
    
    #second layer
    H = mem['X1']
    grad_Wo = np.matmul(H.T, backward_first)
    grad_bo = np.sum(backward_first,axis=0)
    gradients["dW2"] = grad_Wo
    gradients["db2"] = grad_bo.reshape(1, grad_bo.shape[0])
    

    return gradients

def get_accuracy(target, pred):
    N = target.shape[0]
    
    pred_idx = np.argmax(pred, axis=1)
    target_idx = np.argmax(target, axis=1)
    
    acc = np.sum(pred_idx == target_idx)/N
    
    return acc

def initialize_momentum(NN_architecture, factor_mom=10e-5):
    momentum = {}
    
    for index, layer in enumerate(NN_architecture):
        index += 1
        layer_input_size = layer['input_dim']
        layer_output_size = layer['output_dim']
        momentum["v"+str(index)] = np.ones((layer_input_size, layer_output_size))*factor_mom
    return momentum

def flatten_out(X):
  N = X.shape[0]
  X = X.reshape(N, 784)
  return X

x_trainf = flatten_out(x_train)
x_valf = flatten_out(x_val)
x_testf = flatten_out(x_test)

def update_step(parameters, gradients, momentum, lr=10e-5, gamma=0.99):
    
    v_old_Wh = momentum["v1"]
    v_old_Wo = momentum["v2"]
    
    v_new_Wh = gamma*v_old_Wh + lr*gradients["dW1"]
    v_new_Wo = gamma*v_old_Wo + lr*gradients["dW2"]
    
    parameters["W1"] -= v_new_Wh
    parameters["b1"] -= lr*gradients["db1"]
    parameters["W2"] -= v_new_Wo
    parameters["b2"] -= lr*gradients["db2"]
    
    momentum["v1"] = v_new_Wh
    momentum["v2"] = v_new_Wo

    
    return momentum, parameters

training = train(x_trainf, x_valf, x_testf, y_train, y_val, y_test, NN_architecture, 200, n_layers = 1000)

def train(x_train, x_val, x_test, y_train, y_val, y_test, NN_architecture, epochs, n_layers, lr=10e-5, gamma=0.999, factor_mom=10e-5):
    # initilizing the momentum matrix
    momentum = initialize_momentum(NN_architecture,factor_mom)
    parameters = initialize_layers(NN_architecture)
    # loss and accuracy 
    train_loss, val_loss, test_loss = [], [], []
    train_acc, val_acc, test_acc = [], [], []
    
    for epoch in range(epochs):
        x_train, y_train = shuffle(x_train, y_train)
        pred, output, cache, parameters_forward = forward_propagation(x_train, NN_architecture, parameters)
        
        # validation and  accuracy feed forward
        pred_val, _, _, _ = forward_propagation(x_val, NN_architecture, parameters)
        pred_test, _, _, _ = forward_propagation(x_test, NN_architecture, parameters)
        # loss and accuracy computations
        point_loss_train = averageCE(y_train, pred)
        point_loss_val = averageCE(y_val, pred_val)
        point_loss_test = averageCE(y_test, pred_test)
        
        point_acc_train = get_accuracy(y_train, pred)
        point_acc_val = get_accuracy(y_val, pred_val)
        point_acc_test = get_accuracy(y_test, pred_test)
        
        # gradient calculation and update step
        gradients = backward_propagation(x_train, y_train, pred, output, cache, parameters_forward)
        momentum, parameters = update_step(parameters_forward, gradients, momentum, lr, gamma)
    
        if epoch%10 == 0:
            file_name = 'Epoch {} lr = {} n_layers = {}.csv'.format(epoch, lr, n_layers)
            np.savetxt(file_name, [parameters_forward], fmt='%s')
            print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.3f}".format(point_loss_train),
                 "train_accuracy=", "{:.3f}".format(point_acc_train), "val_accuracy=", "{:.3f}".format(point_acc_val),
                 "test_accuracy=", "{:.3f}".format(point_acc_test))
    
        train_loss.append(point_loss_train)
        val_loss.append(point_loss_val)
        test_loss.append(point_loss_test)
        
        train_acc.append(point_acc_train)
        val_acc.append(point_acc_val)
        test_acc.append(point_acc_test)
    print(point_acc_test)
    plt.title("Train, Validation, and Test Loss")
    plt.plot(range(len(train_loss)), train_loss, label="Train")
    plt.plot(range(len(val_loss)), val_loss, label="Validation")
    plt.plot(range(len(test_loss)), test_loss, label="Test")
    plt.xlabel("Epoch") # curve is too steep to visualize loss before 100 epochs
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()
    
    # plotting train and validation accuracy
    plt.title("Train, Validation, and Test accuracy")
    plt.plot(range(len(train_acc)), train_acc, label = "Train")
    plt.plot(range(len(val_acc)), val_acc, label = "Validation")
    plt.plot(range(len(test_acc)), test_acc, label = "Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    
    return parameters
# prepared by Daniel Gershanik
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