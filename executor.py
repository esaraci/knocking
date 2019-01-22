#! /usr/bin/env python
import os

APPS = ['dropbox', 'evernote', 'facebook', 'gmail', 'gplus', 'twitter'] # removed tumblr

for app in APPS:
    os.system("./classifier.py {}".format(app))
