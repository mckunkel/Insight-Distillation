#creates metadata

import os
import constants as c
from tqdm import tqdm
import matplotlib.pyplot as plt

def make_metadata():
    sub_dirs = list(os.walk(c.dir_train))[1:]

    # collect data_stuff metadata
    train_metadata = []

    for dir_path, _, files in tqdm(sub_dirs):

        dir_name = dir_path.split('/')[-1]

        for file_name in files:
            if not file_name.startswith('.'):
                # read the image with matplotlib
                image = plt.imread(os.path.join(dir_path, file_name))

                    # collect and store the image metadata
                image_metadata = []
                image_metadata.extend([dir_name, file_name])
                image_metadata.extend(
                    list(image.shape) if len(image.shape) == 3
                    else [image.shape[0], image.shape[1], 1]
                )
                image_metadata.extend([image.nbytes, image.dtype])
                # append image metadata to list
                train_metadata.append(image_metadata)
    #return the metadata
    return train_metadata

if __name__ == '__main__':
    make_metadata()
