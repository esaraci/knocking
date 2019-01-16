import os

APPS = ['dropbox', 'evernote', 'facebook', 'gmail', 'gplus', 'tumblr', 'twitter']

for app in APPS:
    os.system("./main.py {}".format(app))
