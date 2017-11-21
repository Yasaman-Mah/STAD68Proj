import functools 
import tensorflow as tf
import numpy as np
from sample import *


# this idea is from https://danijar.com/structuring-your-tensorflow-models/
def scoped_property(func):
	attribute = '_cache_' + func.__name__

	@property
	@functools.wraps(func)
	def decorator(self):
		if not hasattr(self, attribute):
			with tf.variable_scope(func.__name__):
				setattr(self, attribute, func(self))
		return getattr(self, attribute)
	return decorator


def weights(shape):
	initial = tf.truncated_normal(shape, stddev=0.01)
	return tf.Variable(initial)

def bias(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


class FCNN_SOFTMAX():

    def __init__(self, transactions, labels, num_neurons=200):
        self.x = transactions
        self.labels = labels
        self.layer_size = num_neurons
        self.input_size = int(self.x.get_shape()[1])
        self.score
        self.loss
        self.optimize


    @scoped_property
    def score(self):
        w1 = weights([self.input_size, self.layer_size])
        b1 = bias([self.layer_size])
        w2 = weights([self.layer_size, 2])
        b2 = bias([2])
        
        h1 = tf.nn.relu(tf.matmul(self.x, w1) + b1)
        h2 = tf.matmul(h1, w2) + b2
        
        return h2  
        
        
    @scoped_property
    def loss(self):
        scores = self.score
        
        probabilities = tf.nn.softmax(scores)
        
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=scores))
        
        # Not sure yet if class probabilities are needed
        return cross_entropy_loss



    @scoped_property
    def optimize(self):
        loss = self.loss
        optimize = tf.train.AdagradOptimizer(0.01).minimize(loss)
        
        return optimize


if __name__ == "__main__":
    train_fraud = np.load('train_fraud.npy')
    train_legitimate = np.load('train_legitimate.npy')
    samp = get_sample(train_legitimate, train_fraud, 100, 30)
    print(sum(samp[:, 30])/samp.shape[0])
    

    transactions = tf.placeholder(tf.float32, [None, 30])
    labels = tf.placeholder(tf.float32, [None, 2])
    ffnn = FCNN_SOFTMAX(transactions, labels)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    for i in range(50000):
        batch = get_sample(train_legitimate, train_fraud, 100, 30)
        true_labels = np.zeros([100, 2])
        true_labels[:,0] = batch[:,30]
        true_labels[:,1] = 1 - batch[:,30]
        
        sess.run(ffnn.optimize, feed_dict={transactions:batch[:,0:30], labels:true_labels})
        loss = sess.run(ffnn.loss, feed_dict={transactions:batch[:,0:30], labels:true_labels})
        
        if (i%100 == 0):
            print("iteration", i, "loss", loss)

    saver.save(sess, "FF_SOFTMAX.ckpt"
    
        
        
        
        
