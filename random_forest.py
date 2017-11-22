from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sample import *
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

train_fraud = np.load('train_fraud.npy')
train_legitimate = np.load('train_legitimate.npy')

# will not feed data in batches
samp = get_sample(train_legitimate, train_fraud, 10000, 200)

rfc = RandomForestClassifier(class_weight={0:2, 1:1})
rfc.fit(samp[:,0:30], samp[:,30])

print(rfc.score(samp[:,0:30], samp[:,30]))

test_fraud = np.load('test_fraud.npy')
test_legitimate = np.load('test_legitimate.npy')

test_data = np.concatenate((test_legitimate, test_fraud))

probabilities = rfc.predict_proba(test_data[:,0:30])
print(sum(probabilities[:,0]), test_data[0,30], rfc.classes_, "num classes", rfc.n_classes_)

precision_rf, recall_rf, thresholds_rf = precision_recall_curve(test_data[:,30], probabilities[:,1])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.step(recall_rf, precision_rf)
ax.set_xlabel('recall')
ax.set_ylabel('precision')
ax.set_title('precision recall curve for random forest (Gini loss)')
plt.show()





