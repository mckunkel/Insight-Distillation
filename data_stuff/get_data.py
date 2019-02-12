#checks to see if data exists, if not it will download and extract data

import os, tarfile
import download_data as dl
import constants as c

# Checking to see if the directory exists already
def check_dir_existance():
    return os.path.isdir(c.dir_train)

# Check if the directory is empty
def check_if_dir_empty():
    len_of_dir = 0
    if check_dir_existance():
        len_of_dir = len(list(os.walk(c.dir_train))[1:])
    return len_of_dir

# Check if the .tar file is in existance
def check_tar_existance():
    return os.path.exists(c.file_train)

# Modele to extract tar file
def untar(fname):
    if fname.endswith("tar"):
        tar = tarfile.open(fname, "r:")
        tar.extractall()
        tar.close()
        print ("Extracted in Current Directory")

# Checks to see if the proper set up on folders has been accomplished
# if not, will create folders and download the file
# if .tar file exists will untart file
def run_data_checker():
    if check_dir_existance() and check_if_dir_empty() > 200:
        print('Files already in folder {}'.format(c.dir_train))
    elif check_tar_existance():
        print('The tar file is already downloaded. Extracting tar file')
        untar(c.file_train)
    else:
        print('No data or .tar file exists')
        print('Downloading the .tar file')
        dl.get_data()
        untar(c.file_train)

if __name__ == '__main__':
    run_data_checker()