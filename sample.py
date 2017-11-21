import numpy as np


## return stratified samples, oversampling minority class

def get_sample(all_legitimate, all_fraud, num_sample, fraud_num, replc=False):
    ''' return num_sample random samples, where fraud_frac is total
    number of frauds in the sample.
    '''
    sample_fr = all_fraud[np.random.choice(all_fraud.shape[0], size=fraud_num, replace=replc)]
    # add samples from legitimate class
    sample_leg = all_legitimate[np.random.choice(all_legitimate.shape[0], size=(num_sample-fraud_num), replace=replc)]
    
    # shuffle
    sample = np.concatenate((sample_fr, sample_leg))
    np.random.shuffle(sample)
    return sample
