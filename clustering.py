#! /usr/bin/env python

import itertools
import time
import sys
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dtw import dtw
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

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
KEEP THESE LINES AS A "BACKUP"
_CONSTS = {"facebook"     :(0, 9829),
             "twitter"      :(50319, 60742),
             "gmail"        :(86578, 96501),
             "gplus"        :(96502, 107293),
             "tumblr"       :(129973, 140532), --> not ready for classificaiton.
             "dropbox"      :(186675, 196682),
             "evernote"     :(235137, 246620)
             }

"""

# very ugly
_CONSTS = dict(facebook=(0, 9829), twitter=(50319, 60742), gmail=(86578, 96501), gplus=(96502, 107293),
               tumblr=(129973, 140532), dropbox=(186675, 196682), evernote=(235137, 246620))


def _str_to_list(s):
    """
    This functions expects strings in the form of s = '[n1, n2, ...]'
    as loaded from the original dataset. With these kind of variables
    we have that s[0] is '[' while we want it to be n1.
    To make them useful i need to convert them to actual lists.
    :param s: list-looking-like string
    :return: the actual list represented by the string
    """
    return [int(sub.replace("[", "").replace("]", "")) for sub in s.split(",")]


# manhattan distance
def _abs(x1, x2):
    """
    function used by DTW, it's just the absolute value of the difference
    also known as cityblock distance or manhattan distance
    :param x1: first element
    :param x2: second element
    :return: absolute value of x1-x2
    """
    return abs(x1 - x2)


"""
# this has been used to print the dendrogram in the report
# leaving it here
def print_dendrogram(Z):
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
    plt.show()
"""

def load_raw_data(path):
    """
    Loads the dataset using Pandas
    We just need some of the columns, for clusering purposes we just need 'packets_length_total',
    the others are needed to link flows to their respective actions
    :param path: dataset path
    :return: list of flows,
    """
    df = pd.read_csv(path)
    # extracting useful data
    data = df.loc[ENV_STARTING_INDEX:ENV_LAST_INDEX, ["action_start", "packets_length_total", "action"]].values
    # converting flows to actual lists

    return [_str_to_list(row[1]) for row in data], data[:, [0, 2]]


def _triu_indexes(start=0, end=1523):
    """
    function name is misleading, it was initially used for other purposes.
    it is only used when CONCURRENT_EXEC = True, it returns
    a list of pairs denoting the staring index and the last index of flows
    to be analyzed. It's used to split the jobs among threads.
    :return:
    """
    # return [(i, j) for i in range(N_FLOWS) for j in range(i+1, N_FLOWS)]
    return [(0, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, N_FLOWS)]
    # return [(0, 4000), (4000, 8000), (8000, 12000), (12000, 16000), (16000, N_FLOWS)]


def concurrent_cdm(flows, dist_func, limits):
    """
    Concurrently building the Condensed Distance Matrix (CDM)
    Executed when CONCURRENT_EXEC = True
    :param flows: the list of flows loaded from the dataset
    :param dist_func: the dist function that will be passed to DTW
    :param limits: pair containing the staring index and the last index. look at `_triu_indexes` above
    :return: a CDM, ideally every entry (i, j) contains dtw(i, j).
    """
    # return dtw(flows[pair[0]], flows[pair[1]], dist_func)[0]
    i_start = limits[0]
    i_end = limits[1]

    return [dtw(flows[i], flows[j], dist_func)[0]
            for i in range(i_start, i_end)
            for j in range(i + 1, N_FLOWS)]


def cdm(flows, dist_func=_abs):
    """
    Same as `concurrent_cdm` but no pair of indexes are involved.
    Executed when CONCURRENT_EXEC = False
    :param flows:
    :param dist_func:
    :return:
    """
    return [dtw(flows[i], flows[j], dist_func)[0]
            for i in range(N_FLOWS)
            for j in range(i + 1, N_FLOWS)]


def clustering(cdm, linkage_metric="average"):
    # cdm: precomputed condensed distance matrix
    # memory issues for large number of flows
    Z = linkage(cdm, method=linkage_metric)
    return fcluster(Z, N_CLUSTERS, criterion='maxclust')


def prepare_samples(clusters, data):
    """
    :param clusters: the output of `clustering`, its length is N_FLOWS, each entry has a value
    in [1, N_CLUSTERS] which denotes the cluster assigned to the i-th entry (or flow).
    :param data: data containing "action ids" and action labels, the i-th entry of data contains the action_label
    and "action id" of the i-th entry of `clusters`. It is needed to link flows to their actions.
    :return: samples, action representation using clustering info, read comments for more info.
    """
    samples = []
    features = np.zeros(N_CLUSTERS)

    # settings these two values allows me not to check for None values
    prev_action_id = data[0][0]
    prev_action_label = data[0][1]

    for idx, row in enumerate(data):
        # saving current action_{label, id}
        cur_action_id = row[0]
        cur_action_label = row[1]
        if prev_action_id != cur_action_id:
            # if we just passed to a new action
            # we have to save the current features and reset
            # save data retrieved till now (action, features)
            samples.append((prev_action_label, features))
            # reset features for next action
            features = np.zeros(N_CLUSTERS)

        # res[idx] = cluster associated with the idx-th flow
        # features[res[idx]] number of elements in the res[idx]-th cluster for this action
        # NB: clusters are numbered starting from 1, so we need res[idx] to be res[idx]-1  when used as index
        features[clusters[idx] - 1] += 1

        # updating
        prev_action_label = cur_action_label
        prev_action_id = cur_action_id

        if idx+1 == N_FLOWS:
            # no more flows to analyze
            # saving this one
            samples.append((prev_action_label, features))

    return samples


def save_dataset(samples):
    """
    Function that actually saves the dataset and adds the headers to it, basically converts a numpy array to a csv file.
    :param samples: output of `prepare_samples`, a list of samples (actions) with |N_CLUSTERS| integer features plus
    the "target" being the action label.
    :return: None
    """
    # writing dataset to csv file
    with open('./{}_dataset_{}.csv'.format(ENV_TASK, N_CLUSTERS), 'w') as f:
        # building headers
        # C1, C2,..., Cn, action
        headers = ""
        for i in range(N_CLUSTERS):
            headers += "C{},".format(i + 1)
        headers += "action\n"
        f.write(headers)

        # data rows
        # first clusters, then action label
        # row[0] = action label
        # row[1] = list of cluters (features)
        for row in samples:
            for c in row[1]:
                f.write("{},".format(int(c)))
            f.write("\"{}\"\n".format(row[0]))


if __name__ == '__main__':

    print("STARTING")
    N_CLUSTERS = 250  # --> 250 seems to be optimal
    PATH = "./apps_total_plus_filtered.csv"

    # F: flows loaded from dataset
    # D: F but with more features, needed later
    # X: condensed distance matrix (upper triangular matrix)
    # C: list with cluster assignments for each flow in D

    # task should be red from args

    # [MAIN]
    if len(sys.argv) < 2:
         exit("Usage: ./clustering.py APPNAME")
    else:
        ENV_TASK = sys.argv[1]
    #    
    # ENV_TASK = "gmail"
    ENV_STARTING_INDEX = _CONSTS[ENV_TASK][0]
    ENV_LAST_INDEX = _CONSTS[ENV_TASK][1]

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
    dataset = prepare_samples(clusters=C, data=D)
    print("[INFO] prepare_dataset took {:.3f}s".format(time.time() - start_time))

    start_time = time.time()

    print("Saving dataset...")
    save_dataset(dataset)
    print("[INFO] save_dataset took {:.3f}s".format(time.time() - start_time))

    print("FINISHED")
