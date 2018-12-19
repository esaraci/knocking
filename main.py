import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# loading source file
mat = sio.loadmat("traffic.mat")



# setting X (data) and y (target)
# X = [x[0][0] for x in mat['vectors']]
data = []
X = [pd.Series(x[0][0]) for x in mat['vectors']]
target = [x[0][0] for x in mat['labels']]

for series in X:
    features = []
    features.append(len(series))                                # number of packets
    
    # take from (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.73.8695&rep=rep1&type=pdf)
    features.append(series.max())                               # max
    features.append(series.min())                               # min
    features.append(series.max() - series.min())                # amplitude
    features.append(series.mean())                              # mean
    # features.append(series.std()**2)                            # covariance
    print(series.std())
    features.append(series.std())                               # std deviation

    data.append(features)

# splitting train and test
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=456)


rfc = RandomForestClassifier(n_estimators=10, random_state=456, n_jobs=-1)
rfc.fit(X_train, y_train)
print(rfc.score(X_test, y_test))
