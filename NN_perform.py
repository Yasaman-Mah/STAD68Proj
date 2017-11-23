import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, average_precision_score
from FCNN_softmax import FCNN_SOFTMAX
from FCNN_SVM import FCNN_SVM
import matplotlib.pyplot as plt


test_fraud = np.load('test_fraud.npy')
test_legitimate = np.load('test_legitimate.npy')

test_data = np.concatenate((test_legitimate, test_fraud))
test_labels = np.zeros([test_data.shape[0], 2])
test_labels[:,0] = test_data[:,30]
test_labels[:,1] = 1 - test_data[:,30]

softmax_graph = tf.Graph()
with softmax_graph.as_default():
    transactions = tf.placeholder(tf.float32, [None, 30])
    labels = tf.placeholder(tf.float32, [None, 2])
    softmax_NN = FCNN_SOFTMAX(transactions, labels)
sess1 = tf.Session(graph=softmax_graph)

with sess1.as_default():
    with softmax_graph.as_default():
        saver = tf.train.Saver()
        saver.restore(sess1, "FF_SOFTMAX.ckpt")

        # need class probabilities to calculate AUPRC
        _ , probabilities = sess1.run(softmax_NN.loss, feed_dict={transactions:test_data[:,0:30], labels:test_labels})
 
        precision_softmax, recall_softmax, thresholds_softmax = precision_recall_curve(test_labels[:,0], probabilities[:,0])
        avg_precision_softmax = average_precision_score(test_labels[:,0], probabilities[:,0])

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.step(recall_softmax, precision_softmax)
ax1.set_xlabel('recall')
ax1.set_ylabel('precision')
ax1.set_title('precision recall curve for feed forward neural net with cross entropy loss')
plt.show()

print("average precision score for softmax classifier", avg_precision_softmax)


##### neural net with svm layer
svm_graph = tf.Graph()

with svm_graph.as_default():
    transactions = tf.placeholder(tf.float32, [None, 30])
    labels = tf.placeholder(tf.float32, [None, 1])
    SVM_NN = FCNN_SVM(transactions, labels)

sess2= tf.Session(graph=svm_graph)
with sess2.as_default():
    with svm_graph.as_default():
        saver = tf.train.Saver()
        saver.restore(sess2, "FF_SVM.ckpt")

        # need class probabilities to calculate AUPRC
        _ , probabilities = sess2.run(SVM_NN.loss, feed_dict={transactions:test_data[:,0:30], labels:test_labels[:,0:1]})
 
        precision_svm, recall_svm, thresholds_svm = precision_recall_curve(test_labels[:,0], probabilities)
        avg_precision_svm = average_precision_score(test_labels[:,0], probabilities)

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.step(recall_svm, precision_svm)
ax2.set_xlabel('recall')
ax2.set_ylabel('precision')
ax2.set_title('precision recall curve for feed forward neural net with SVM loss')
plt.show()
print("average precision score for svm classifier", avg_precision_svm)
