from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sample import *

train_fraud = np.load('train_fraud.npy')
train_legitimate = np.load('train_legitimate.npy')

# will not feed data in batches
samp = get_sample(train_legitimate, train_fraud, 10000, 100)

rfc = RandomForestClassifier(class_weight={0:2, 1:1})
rfc.fit(samp[:,0:30], samp[:,30])

print(rfc.score(samp[:,0:30], samp[:,30]))

#test_fraud = np.load('test_fraud.npy')
#test_legitimate = np.load('test_legitimate.npy')

#test_samp = get_sample(test_legitimate, test_fraud, 1000, 300, replc=True)

#print(rfc.score(test_samp[:,0:30], test_samp[:,30]))


