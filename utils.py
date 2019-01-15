import numpy as np
import pandas as pd
import main as m



def load_raw_data(path):
    df = pd.read_csv(path)
    # extracting useful data
    fb_data = df.loc[:,["action"]].values
    # converting flows to actual lists
    return fb_data

for app in ['dropbox', 'evernote', 'facebook', 'gmail', 'gplus', 'tumblr', 'twitter']:
    data = load_raw_data("./datasets/{}_dataset.csv".format(app))
    print("Actions for {}: {}".format(app, len(np.unique(data))))


exit()
print(np.unique(data[:, 0], return_counts=True))
