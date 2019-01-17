#! /usr/bin/env python
import os

APPS = ['dropbox', 'evernote', 'facebook', 'gmail', 'gplus', 'tumblr', 'twitter']

for app in APPS:
    os.system("./clustering.py {}".format(app))
