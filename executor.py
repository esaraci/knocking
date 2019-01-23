#! /usr/bin/env python
import os

# remove 'tumblr' from APPS when classifying, you can keep it when clustering
APPS = ['dropbox', 'evernote', 'facebook', 'gmail', 'gplus', 'twitter'] 

for app in APPS:
    os.system("./classifier.py {}".format(app))
