from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sample import *
from sklearn.metrics import precision_recall_curve, auc
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
auprc = auc(recall_rf, precision_rf)

np.savez("rf_prec_recall.npz", prec=precision_rf, recall=recall_rf)

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.step(recall_rf, precision_rf)
#ax.set_xlabel('recall')
#ax.set_ylabel('precision')
#ax.set_title('precision recall curve for random forest')
#plt.show()

print("area under precision recall curve", auprc)

# bootstrapping
k = 100
auprcs = np.zeros(k)
num_samples = test_data.shape[0]

for i in range(k):
    resampled_test = test_data[np.random.choice(num_samples, num_samples, replace=True)]
    prob = rfc.predict_proba(resampled_test[:,0:30])
    prec, recall, _ = precision_recall_curve(resampled_test[:,30], prob[:,1])
    
    auprcs[i] = auc(recall, prec)

# get 95 and 5 percentiles
percentiles = np.percentile(auprcs, (2.5, 97.5))
print("95% confidence interval for auprc",percentiles)


