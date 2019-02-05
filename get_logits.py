import numpy as np
from tqdm import tqdm
import sys

# use non standard flow_from_directory
from utils.image_preprocessing_v1 import ImageDataGenerator
# it outputs not only x_batch and y_batch but also image names

from keras.models import Model
from models.xception import Xception, preprocess_input
import constants as c


data_dir = c.data_dir

data_generator = ImageDataGenerator(
    data_format='channels_last',
    preprocessing_function=preprocess_input
)
data_generator2 = ImageDataGenerator(
    data_format='channels_last',
    preprocessing_function=preprocess_input
)
train_generator = data_generator.flow_from_directory(
    data_dir + 'train',
    target_size=(299, 299),
    batch_size=64, shuffle=False
)

val_generator = data_generator2.flow_from_directory(
    data_dir + 'val',
    target_size=(299, 299),
    batch_size=64, shuffle=False
)

# Load model and remove last layer of parent model
model = Xception()
model.load_weights('xception_weights.hdf5')
# remove softmax
model.layers.pop()
model = Model(model.input, model.layers[-1].output)
# now model outputs logits

batches = 0
train_logits = {}

for x_batch, _, name_batch in tqdm(train_generator):

    batch_logits = model.predict_on_batch(x_batch)

    for i, n in enumerate(name_batch):
        train_logits[n] = batch_logits[i]

    batches += 1
    if batches >= 400:
        break

batches = 0
val_logits = {}

for x_batch, _, name_batch in tqdm(val_generator):

    batch_logits = model.predict_on_batch(x_batch)

    for i, n in enumerate(name_batch):
        val_logits[n] = batch_logits[i]

    batches += 1
    if batches >= 400:
        break


np.save(data_dir + 'train_logits.npy', train_logits)
np.save(data_dir + 'val_logits.npy', val_logits)