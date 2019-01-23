#! /usr/bin/env python

import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

D = {

    "facebook": ["send message selection", "open facebook", "status post selection",
                 "post button selection", "user profile selection", "first conversation selection",
                 "status selection"],

    "gmail": ["sending mail", "reply selection", "chats selection", "sending mail reply"],

    "twitter": ["open Twitter", "contact selection", "refresh home",
                "send selection", "send tweet selection", "direct messages selection",
                "tweets selection"],

    "dropbox": ["open dropbox", "file favorite", "favorites",
                "delete file confirm", "delete folder confirm", "folder creation",
                "open file", "open folder", "file text input"],

    "gplus": ["open gplus", "refresh", "user page", "new selection",
              "send new post", "post selection", "post delete confirm",
              "plus post", "comment send", "back to main"],

    "evernote": ["open evernote", "market selection", "new note title input",
                 "done text note", "done note edit", "done audio note"]

    # tumblr actions are not clear, cant find the righ association with the paper
    # "tumblr":   ["open tumblr", "user page", "user likes", "following page", "home page"]

}


def _rename_if_non_relevant(s, app):
    """
    :param s: string to be checked
    :param app: string used as a key for @D, it's the app name
    :return: 'other' when @s is not in @D[s] else returns @s itself
    """

    # the authors identified that some actions are fairly similar
    # it is useful to put them under the same target class
    if app == "twitter" and (s == "send tweet selection" or s == "send selection"):
        return "send tweet/message"

    if app == "dropbox" and (s == "open file" or s == "open folder"):
        return "open file/folder"

    return "other" if s not in D[app] else s


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=60, fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=8)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.tight_layout()


if __name__ == '__main__':

    # [MAIN] command line execution
    if len(sys.argv) < 2:
        exit("Usage: ./classifier.py APPNAME")
    else:
        ENV_TASK = sys.argv[1]

    # ENV_TASK = "evernote"

    # [LOADING DATASET]
    dataset = pd.read_csv("./datasets/{}_dataset_250.csv".format(ENV_TASK))

    # [DATASET 'PREPROCESSING']
    n_clusters = len(dataset.columns) - 1
    data = dataset.loc[:, "C1":"C{}".format(n_clusters)].values
    target = [_rename_if_non_relevant(s, ENV_TASK) for s in dataset.loc[:, "action"].values]

    # [HOLDOUT]
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=1337)

    # [TRAINING AND TESTING]
    rfc = RandomForestClassifier(random_state=1337)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    # [COMPUTING CONFUSION MATRIX]
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # [PLOTTING CONFUSION MATRIX]
    # make sure that the target path exists
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=np.unique(target), normalize=True,
                          title='Normalized confusion matrix for {}'.format(ENV_TASK))
    # plt.savefig("./images/{}_cm.png".format(ENV_TASK))
    # plt.show()

    # [EVALUATION]
    # 
    # for action-wise performances uncomment the next lines
    # prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, warn_for=())
    #     print(np.unique(target))
    #     print("P: {}".format(prec))
    #     print("R: {}".format(rec))
    #     print("F1: {}".format(f1))

    
    # weighted averages, app wise performances
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", warn_for=())

    print("---[scores for {}]---".format(ENV_TASK))
    print("P: {:.2f}".format(prec))
    print("R: {:.2f}".format(rec))
    print("F1: {:.2f}".format(f1))
