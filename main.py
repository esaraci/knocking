import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from dtw import dtw
# from fastdtw import fastdtw
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import itertools
import time

"""
"send message selection"            send message
"open facebook"                     open facebook
"status post selection"             post on own wall
"post button selection"             post on friends wall
"user profile selection"            open user profile
"first conversation selection"      open message
"status selection"                  status button
"""


"""
facebook (0, 9830)
twitter (50319, 60742)
gmail (86578, 96502)
"gplus" :(96502,107293),
"tumblr":(129973, 140532),
"dropbox"   :(186675, 196682),
"evernote"  :(235137, 246620)
"""



# ugly but easier
_CONSTS = {"facebook"     :(0, 9829),
             "twitter"      :(50319, 60742),
             "gmail"        :(86578, 96501),
             "gplus"        :(96502, 107293),
             "tumblr"       :(129973, 140532),
             "dropbox"      :(186675, 196682),
             "evernote"     :(235137, 246620)
             }

def _str_to_list(s):
    return [int(sub.replace("[", "").replace("]", "")) for sub in s.split(",")]

def _abs(x1, x2):
    return abs(x1-x2)


# TODO: this has to moved in the classification algorithm
def _rename_if_non_relevant(s):
    D =  ["send message selection",
          "open facebook" ,
          "status post selection",
          "post button selection",
          "user profile selection",
          "first conversation selection",
          "status selection"]

    return "other" if s not in D else s

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

def load_raw_data(path):
    df = pd.read_csv(path)
    # extracting useful data
    fb_data = df.loc[ENV_STARTING_INDEX:ENV_LAST_INDEX,["action_start", "packets_length_total", "flow_length", "action"]].values
    # converting flows to actual lists
    return [_str_to_list(row[1]) for row in fb_data], fb_data

def cdm(flows, dist_func=_abs):
    return [dtw(flows[i], flows[j], dist_func)[0]
                          for i in range(N_FLOWS)
                              for j in range(i+1, N_FLOWS)]

def _triu_indexes():
    # return [(i, j) for i in range(N_FLOWS) for j in range(i+1, N_FLOWS)]
    return [(0, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, N_FLOWS)]
    # return [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]

def concurrent_cdm(flows, dist_func, limits):
    # return dtw(flows[pair[0]], flows[pair[1]], dist_func)[0]
    i_start = limits[0]
    i_end   = limits[1]

    return [dtw(flows[i], flows[j], dist_func)[0]
                          for i in range(i_start, i_end)
                              for j in range(i+1, N_FLOWS)]


def clustering(cdm, linkage_metric="average"):
    # cdm: precomputed condensed distance matrix
    Z = linkage(cdm, method=linkage_metric)
    return fcluster(Z, N_CLUSTERS, criterion='maxclust')

def prepare_dataset(res, fb_data):

    dataset             = []
    features            = np.zeros(N_CLUSTERS)
    # prev_action_id      = 1383129102.11     # setting first value, => ugly but makes it easier
    # prev_action_label   = "open facebook"   # setting first value, => ugly but makes it easier
    prev_action_id      = fb_data[0][0]
    prev_action_label   = fb_data[0][3]

    for idx, row in enumerate(fb_data):
        # saving current action_{label, id}
        cur_action_id       = row[0]
        cur_action_label    = row[3]
        # TODO: attenzione al caso base in cui action_{id, label} sono none <== not needed since hardcoded values set before loop
        if prev_action_id != cur_action_id:
            # if we skipped to the next action
            # we have to save the current data and reset
            # save data retrieved till now (action, features)
            dataset.append((prev_action_label, features))
            # reset features for next action
            features = np.zeros(N_CLUSTERS)

        # res[idx] = cluster associated with the idx-th flow
        # features[res[idx]] number of elements in the res[idx]-th cluster for this action
        # NB: clusters are numbered starting from 1, so we need res[idx] to be res[idx]-1  when used as index
        features[res[idx]-1] += 1
        prev_action_label = cur_action_label
        prev_action_id = cur_action_id

        if idx+1 == N_FLOWS:
            # no more flow to analyze
            # saving this one
            dataset.append((prev_action_label, features))

    return dataset

def save_dataset(dataset):
    # writing dataset to csv file
    with open('./{}_dataset.csv'.format(ENV_TASK), 'w') as f:
        # header row
        # C1, C2,..., Cn, action
        headers = ""
        for i in range(N_CLUSTERS):
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


if __name__ == '__main__':
    
    print("STARTING")
    # CONSTS
    N_CLUSTERS      = 50
    PATH            = "./apps_total_plus_filtered.csv"



    # F: flows loaded from dataset
    # D: same cardinality as F but with more info needed later
    # X: condensed distance matrix (upper triangular matrix)
    # C: list with cluster assignments for each flow in D
    
    # task should be red from args
    ENV_TASK = "twitter"
    ENV_STARTING_INDEX  = _CONSTS[ENV_TASK][0]
    ENV_LAST_INDEX      = _CONSTS[ENV_TASK][1]

    F, D = load_raw_data(path=PATH)
    N_FLOWS = len(F)
   

    start_time = time.time()

    CONCURRENT_EXEC = True
    if CONCURRENT_EXEC:
        print("Building CDM concurrently...")
        pool = Pool(5)
        X_indexes = _triu_indexes()
        caller = partial(concurrent_cdm, F, _abs)
        X = pool.map(caller, X_indexes)
        X = list(itertools.chain(*X))

        # print("CDM build...")
        # np.savetxt("./async.txt", X)
    
    else:
        X = cdm(flows=F, dist_func=_abs)
        # np.savetxt("./sync.txt", X)




    print("[INFO] cdm took {:.3f}s]".format(time.time() - start_time))
    start_time = time.time()
    print("Clustering and linking started")
    C = clustering(cdm=X, linkage_metric="average")
    print("Clustering finished")
    
    print("[INFO] clustering + linkage=average took {:.3f}s ".format(time.time() - start_time))


    # DATASET CREATION RELATED
    start_time = time.time()
    print("Bulding dataset...")
    dataset = prepare_dataset(res=C, fb_data=D)
    print("[INFO] prepare_dataset took {:.3f}s".format(time.time() - start_time))
    
    start_time = time.time()

    print("Saving dataset...")
    save_dataset(dataset)
    print("[INFO] save_dataset took {:.3f}s".format(time.time() - start_time))

    print("FINISHED")
