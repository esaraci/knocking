import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import itertools


# i dont think i need this structure, but it's helpful.
D = {


    "facebook": ["send message selection", "open facebook" , "status post selection",
                "post button selection", "user profile selection", "first conversation selection",
                "status selection"],

    "gmail":    ["sending mail", "reply selection", "chats selection", "sending mail reply"],

    "twitter":  ["open Twitter", "contact selection", "refresh home",
                "send selection", "send tweet selection", "direct messages selection",
                "tweets selection"],

    "dropbox":  ["open dropbox", "file favorite", "favorites",
                "delete file confirm", "delete folder confirm", "folder creation", 
                "open file", "open folder", "file text input"],

    "gplus":    ["open gplus", "refresh", "user page", "new selection",
                "send new post", "post selection", "post delete confirm",
                "plus post", "comment send", "back to main"],

    "evernote": ["open evernote", "market selection", "new note title input",
                "done text note", "done note edit", "done audio note"]

    }


def _rename_if_non_relevant(s, app):

    
    # extra very boring things

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
    plt.xticks(tick_marks, classes, rotation=45, fontsize=5)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.tight_layout()


# choose application
task = "dropbox"
dataset = pd.read_csv("./datasets/{}_dataset.csv".format(task))

n_clusters = len(dataset.columns)-1
data    = dataset.loc[:, "C1":"C{}".format(n_clusters)].values
target  = [_rename_if_non_relevant(s, task) for s in dataset.loc[:, "action"].values]

X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=1337)

rfc = RandomForestClassifier(random_state=1337)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
cm_classes = D[task]
cm_classes.append("other")
cm_classes.sort()

plot_confusion_matrix(cnf_matrix, classes=cm_classes, normalize=True,
                     title='Not Normalized confusion matrix')

# plt.savefig("cm.png")
# plt.show()

print("F1:", f1_score(y_test, y_pred, average="weighted"))
print("P:", precision_score(y_test, y_pred, average="weighted"))
print("R:", recall_score(y_test, y_pred, average="weighted"))
# print("ACC:", accuracy_score(y_test, y_pred))
# print(rfc.score(X_test, y_test))
