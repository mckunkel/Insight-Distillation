#used to create a decoder the metadata for later use

import numpy as np
from create_dataframe import create_dataframe


def create_decoder():
    meta_data = create_dataframe()
    decode = {n: i for i, n in meta_data.groupby('category_name').category_number.first().iteritems()}
    np.save('decode.npy', decode)