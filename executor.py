import os

APPS = ['dropbox', 'evernote', 'facebook'] 
# APPS = ['gmail', 'gplus', 'tumblr', 'twitter']

for app in APPS:
    os.system("./clustering.py {}".format(app))
