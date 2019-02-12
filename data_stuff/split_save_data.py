from create_dataframe import create_dataframe
import numpy as np


# Must split the data into training and testing
# Validation should have 25 images per class
def split_save():
    # 25 images per class
    meta_data = create_dataframe()
    value = meta_data.groupby('category_name', group_keys=False).apply(lambda x: x.sample(n=25, replace=False))
    value.sort_index(inplace=True)
    meta_data.drop(value.index, axis=0, inplace=True)

    meta_data.to_csv('train_metadata.csv', index=False)
    value.to_csv('val_metadata.csv', index=False)

    # deleted the definition to decode because the global meta_data = create_dataframe() was causing a crash
    # think about using classes
    decode = {n: i for i, n in meta_data.groupby('category_name').category_number.first().iteritems()}
    np.save('decode.npy', decode)
