import pandas as pd
import main as main
import numpy as np

path = "./apps_total_plus_filtered.csv"

    # F: flows loaded from dataset
    # D: same cardinality as F but with more info needed later
    # X: condensed distance matrix (upper triangular matrix)
    # C: list with cluster assignments for each flow in D


df = pd.read_csv(path)
print(df.loc[0]["action"])

# print(np.unique(df["action"].values, return_counts=True))

# ugly but easier
_SETTINGS = {"facebook"     :(0, 9829),
             "twitter"      :(50319, 60742),
             "gmail"        :(86578, 96501),
             "gplus"        :(96502, 107293),
             "tumblr"       :(129973, 140532),
             "dropbox"      :(186675, 196682),
             "evernote"     :(235137, 246620)
             }

