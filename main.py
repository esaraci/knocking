import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from dtw import dtw
from fastdtw import fastdtw
import matplotlib.pyplot as plt

df = pd.read_csv("./apps_total_plus_filtered.csv")

def str_to_list(s):
    return [int(sub.replace("[", "").replace("]", "")) for sub in s.split(",")]

def _abs(x1, x2):
    return abs(x1-x2)

# extracting useful data
fb_data = df.loc[:5000,["action_start", "packets_length_total", "flow_length", "action"]].values

# converting flows to actual lists
flows = [str_to_list(row[1]) for row in fb_data]

# condensed = upper triangular matrix
n_flows = len(flows)
condensed_dist = [dtw(flows[j], flows[k], _abs)[0]
                      for j in range(n_flows)
                          for k in range(j+1, n_flows)]

Z = linkage(condensed_dist)

"""
# dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
"""

n_clusters = 100
res = fcluster(Z, n_clusters, criterion='maxclust')


# preparing values 
dataset             = []
features            = np.zeros(n_clusters)
prev_action_id      = 1383129102.11      # setting first value, => no need to handle None case
prev_action_label   = "open facebook"   # setting first value, => no need to handle None case

for idx, row in enumerate(fb_data):
    # saving current action_{label, id}
    cur_action_label    = row[3]
    cur_action_id       = row[0]
    # TODO: attenzione al caso base in cui action_{id, label} sono none.
    if prev_action_id != cur_action_id:
        # if we skipped to the next action
        # we have to save the current data and reset
        # save data retrieved till now (action, features)
        dataset.append((prev_action_label, features))
        # reset features for next action
        features = np.zeros(n_clusters)

    
    # res[idx] = cluster associated with the idx-th flow
    # features[res[idx]] number of elements in the res[idx]-th cluster for this action
    # NB: clusters are numbered starting from 1, so we need res[idx] to be res[idx]-1  when used as index
    features[res[idx]-1] += 1
    prev_action_label = cur_action_label
    prev_action_id = cur_action_id

    if idx+1 == n_flows:
        # no more flow to analyze
        # saving this one
        dataset.append((prev_action_label, features))


# writing dataset to csv file
with open('./dataset.csv', 'w') as f:
    # header row
    # C1, C2,..., Cn, action
    headers = ""
    for i in range(n_clusters):
        headers += "C{},".format(i+1)
    headers += "action\n"
    f.write(headers)

    # data rows
    # firs clusters, then action label
    # row[0] = action label
    # row[1] = list of cluters (features)
    for row in dataset:
        for c in row[1]:
            f.write("{},".format(int(c)))
        f.write("\"{}\"\n".format(row[0]))




