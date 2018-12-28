import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



dataset = pd.read_csv("./dataset.csv")

n_clusters = len(dataset.columns) - 1
data    = dataset.loc[:, "C1":"C{}".format(n_clusters)].values
target  = dataset.loc[:, "action"].values

X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=1337)


rfc = RandomForestClassifier(random_state=1337)