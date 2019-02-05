#used to create a dataframe of the metadata for later use
import pandas as pd
from make_metadata import make_metadata as md


def create_dataframe():
    meta_data = pd.DataFrame(md())
    meta_data.columns = ['directory', 'img_name', 'height', 'width', 'channels', 'byte_size', 'bit_depth']

    meta_data['category_name'] = meta_data.directory.apply(lambda x: x.split('.')[-1].lower())
    meta_data['img_extension'] = meta_data.img_name.apply(lambda x: x.split('.')[-1])
    meta_data['category_number'] = meta_data.directory.apply(lambda x: int(x.split('.')[0]))

    # some of the category names are appended with '101'
    # remove '101'
    meta_data.category_name = meta_data.category_name.apply(lambda x: x[:-4] if '101' in x else x)

    return meta_data


if __name__ == '__main__':
    create_dataframe()