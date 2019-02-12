#Cleans the CalTech256 data set
import shutil
import constants as c
import os

# Module to clean the data
def clean_data():
    # There are objects in the files that are not images
    # Delete them
    # First check to see if they are there.
    path = os.path.join(c.dir_train, '198.spider/RENAME2')
    if os.path.exists(path):
        os.remove(path)
    # os.remove does not remove directories
    # check out https://stackoverflow.com/questions/303200/how-do-i-remove-delete-a-folder-that-is-not-empty-with-python
    # I do not understand the intention of the empty folder 056.dog/greg
    path = os.path.join(c.dir_train, '056.dog/greg')
    if os.path.isdir(path):
        shutil.rmtree(path)

    # testing to attempt to increase accuracy
    #removing the clutter

    path = os.path.join(c.dir_train, '257.clutter')
    if os.path.exists(path):
        shutil.rmtree(path)

if __name__ == '__main__':
    clean_data()