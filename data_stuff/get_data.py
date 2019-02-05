#checks to see if data exists, if not it will download and extract data

import os, tarfile
import download_data as dl
import constants as c


def check_dir_existance():
    return os.path.isdir(c.dir_train)


def check_if_dir_empty():
    len_of_dir = 0
    if check_dir_existance():
        len_of_dir = len(list(os.walk(c.dir_train))[1:])
    return len_of_dir


def check_tar_existance():
    return os.path.exists(c.file_train)


def untar(fname):
    if fname.endswith("tar"):
        tar = tarfile.open(fname, "r:")
        tar.extractall()
        tar.close()
        print ("Extracted in Current Directory")


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