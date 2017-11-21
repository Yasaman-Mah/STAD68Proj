import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, average_precision_score
from FCNN_softmax import FCNN_SOFTMAX
import matplotlib.pyplot as plt


test_fraud = np.load('test_fraud.npy')
test_legitimate = np.load('test_legitimate.npy')

test_data = np.concatenate((test_legitimate, test_fraud))
test_labels = np.zeros([test_data.shape[0], 2])
test_labels[:,0] = test_data[:,30]
test_labels[:,1] = 1 - test_data[:,30]


transactions = tf.placeholder(tf.float32, [None, 30])
labels = tf.placeholder(tf.float32, [None, 2])

sess = tf.Session()
softmax_NN = FCNN_SOFTMAX(transactions, labels)

saver = tf.train.Saver()
saver.restore(sess, "FF_SOFTMAX.ckpt")

# need class probabilities to calculate AUPRC
_ , probabilities = sess.run(softmax_NN.loss, feed_dict={transactions:test_data[:,0:30], labels:test_labels})
 
precision_softmax, recall_softmax, thresholds_softmax = precision_recall_curve(test_labels[:,0], probabilities[:,0])

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(recall_softmax, precision_softmax)
ax1.set_xlabel('recall')
ax1.set_ylabel('precision')
ax1.set_title('precision recall curve for feed forward neural net with cross entropy loss')
plt.show()

