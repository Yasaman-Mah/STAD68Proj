import csv
import numpy as np
import math
import random
# split data into training and test sets

TRAIN_RATIO = 0.6

#train = csv.DictReader(open('creditcard.csv'), fieldnames=('Time','V1','V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class'))
data = csv.reader(open('creditcard.csv', newline=''), delimiter=',')

num_frauds = 0
num_txn = 0

fraud = []
legitimate = []

# sort them into fraud and legitimate
for entry in data:
    if (entry[30] == '1'):
        fraud.append(entry)
    else:
        legitimate.append(entry)

# make training and test sets
train_fraud = []
train_legitimate = []

print(legitimate.pop(0))

num_train_fraud = math.floor(TRAIN_RATIO * len(fraud))
train_fraud_idx = random.sample(range(len(fraud)), num_train_fraud)
num_train_legitimate = math.floor(TRAIN_RATIO * len(legitimate))
train_legitimate_idx = random.sample(range(len(legitimate)), num_train_legitimate)


train_fraud = [fraud[i] for i in train_fraud_idx]
train_legitimate = [legitimate[i] for i in train_legitimate_idx]

#test_fraud = [fraud[i] for i in range(len(fraud)) if i not in train_fraud_idx]
#test_legitimate = [legitimate[i] for i in range(len(legitimate)) if i not in train_legitimate_idx]

train_fraud_idx.sort(reverse=True)
train_legitimate_idx.sort(reverse=True)


for i in train_fraud_idx:
    fraud.pop(i)

for i in train_legitimate_idx:
    legitimate.pop(i)



np.save('train_fraud', np.asarray(train_fraud, dtype='float32'))
np.save('train_legitimate', np.asarray(train_legitimate, dtype='float32'))
np.save('test_fraud', np.asarray(fraud, dtype='float32'))
np.save('test_legitimate', np.asarray(legitimate, dtype='float32'))


