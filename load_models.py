from keras.models import model_from_json
from keras.models import model_from_yaml
import numpy
import os, sys
from models.squeezenet_model import SqueezeNet, preprocess_input
from keras.losses import categorical_crossentropy as loss
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
lambda_const = 0.02
temperature = 15
# fix random seed for reproducibility
numpy.random.seed(7)
base_dir = '/Volumes/MacStorage/InSight/Analysis/SqueezeNet/'
json_dir = os.path.join(base_dir,'JSONFiles')
yaml_dir = os.path.join(base_dir,'YAMLFiles')
model_dir = os.path.join(base_dir,'HDF5Files')

json_path = os.path.join(json_dir,'distilled_squeezenet_model_T_{}_lambda_{}.json'.format(temperature,lambda_const))
yaml_path = os.path.join(yaml_dir,'distilled_squeezenet_model_T_{}_lambda_{}.yaml'.format(temperature,lambda_const))
model_path = os.path.join(model_dir,'distilled_squeezenet_model_T_{}_lambda_{}.h5'.format(temperature,lambda_const))

data_dir = '../data_stuff/'

data_generator_val = ImageDataGenerator(
    data_format='channels_last',
    preprocessing_function=preprocess_input
)

val_generator = data_generator_val.flow_from_directory(
    data_dir + 'val', shuffle=False,
    target_size=(299, 299),
    batch_size=64
)
model = SqueezeNet(weight_decay=1e-4, image_size=299)
model.load_weights(model_path, by_name=True)
model.compile(
    optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=True),
    loss=loss, metrics=['categorical_crossentropy', 'accuracy', 'top_k_categorical_accuracy']
)
orig_stdout = sys.stdout
f = open('squeezenetSummary.txt', 'w')
sys.stdout = f
print(model.summary())
sys.stdout = orig_stdout
f.close()
test_image = image.load_img('/Volumes/MacStorage/WorkCodes/GitHub/DistillingObjectDetector/data_stuff/val/1/001_0002.jpg', target_size = (299, 299))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

#predict the result
result = model.predict(test_image)
print(result[0].argmax())
#score = model.evaluate(val_generator,80)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))