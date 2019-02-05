import sys
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from models.squeezenet import SqueezeNet, preprocess_input
from utils.plot_utils import plot_utils as plt_uts
from utils.history_utils import history_utils as hist_uts
from utils.save_utils import save_utils as save_uts
import constants as c

data_dir = c.data_dir

data_generator = ImageDataGenerator(
    data_format='channels_last',
    preprocessing_function=preprocess_input
)

train_generator = data_generator.flow_from_directory(
    data_dir + 'train',
    target_size=(299, 299),
    batch_size=64
)

val_generator = data_generator.flow_from_directory(
    data_dir + 'val', shuffle=False,
    target_size=(299, 299),
    batch_size=64
)

model = SqueezeNet(weight_decay=1e-4, image_size=299)
model.count_params()
orig_stdout = sys.stdout
f = open('modelSummary.txt', 'w')
sys.stdout = f
print(model.summary())
sys.stdout = orig_stdout
f.close()
model.compile(
    optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=True),
    loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy']
)

model.fit_generator(
    train_generator,
    steps_per_epoch=400, epochs=30, verbose=1,
    callbacks=[
        EarlyStopping(monitor='val_acc', patience=4, min_delta=0.01),
        ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, epsilon=0.007)
    ],
    validation_data=val_generator, validation_steps=80, workers=4
)

plt_uts(model, 'Originalsqueezenet', 0, 0)

hist_uts(model, 'Originalsqueezenet', 0, 0)

save_uts(model, 'Originalsqueezenet', 0, 0)
