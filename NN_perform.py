import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, auc
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
        aucpr_sft = auc(recall_softmax, precision_softmax)
        print("area under precision recall curve for neural network with cross entropy loss", aucpr_sft)
        
        #bootstrapping
        k = 100
        auprcs = np.zeros(k)
        num_samples = test_data.shape[0]

        for i in range(k):
            resampled_test = test_data[np.random.choice(num_samples, num_samples, replace=True)]
            # don't care about labels here 
            _ , prob = sess1.run(softmax_NN.loss, feed_dict={transactions:resampled_test[:,0:30], labels:test_labels})
            # but care about labels here
            prec, recall, _ = precision_recall_curve(resampled_test[:,30], prob[:,0])
            auprcs[i] = auc(recall, prec)

        # get 95 and 5 percentiles
        percentiles = np.percentile(auprcs, (2.5, 97.5))
        print("95% confidence interval for aucpr of NN with softmax",percentiles)



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
        aucpr_svm = auc(recall_svm, precision_svm)
        print("area under precision recall curve for neural network with svm layer", aucpr_svm)
        
        #bootstrapping
        k = 100
        auprcs = np.zeros(k)
        num_samples = test_data.shape[0]

        for i in range(k):
            resampled_test = test_data[np.random.choice(num_samples, num_samples, replace=True)]
            # don't care about labels here 
            _ , prob = sess2.run(SVM_NN.loss, feed_dict={transactions:resampled_test[:,0:30], labels:test_labels[:, 0:1]})
            # but care about labels here
            prec, recall, _ = precision_recall_curve(resampled_test[:,30], prob)
            auprcs[i] = auc(recall, prec)

        # get 95 and 5 percentiles
        percentiles = np.percentile(auprcs, (2.5, 97.5))
        print("95% confidence interval for aucpr of NN with SVM",percentiles)
        
##### plot all precision recall curves in one window
# restore precision recall stores for random forest
rf_prec_recall = np.load("rf_prec_recall.npz")

 
 
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.step(recall_softmax, precision_softmax)
ax2.step(recall_svm, precision_svm)
ax2.step(rf_prec_recall['recall'], rf_prec_recall['prec'])
ax2.set_xlabel('recall')
ax2.set_ylabel('precision')
ax2.set_title('precision recall curves')
ax2.legend(["NN with softmax", "NN with SVM", "random forest"])
plt.show()
