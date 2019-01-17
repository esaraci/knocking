#! /usr/bin/env python
import os

APPS = ['dropbox', 'evernote', 'facebook', 'gmail', 'gplus', 'twitter'] # tumblr missing

for app in APPS:
    os.system("./classifier.py {}".format(app))
