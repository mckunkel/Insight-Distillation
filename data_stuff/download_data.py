#Gets the Caltech256 data set from http://www.vision.caltech.edu/Image_Datasets/Caltech256/

import requests, sys
import constants as c


def get_data():
    with open(c.file_train, "wb") as f:
        print("Downloading %s" % c.file_train)
        response = requests.get(c.data_url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                sys.stdout.flush()