# This script runs on the pi 3 B+

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
from threading import Thread
from squeezenet_model import SqueezeNet, preprocess_input
import numpy as np
import imutils
import time
import cv2
import os

mean = np.array([0.485, 0.456, 0.406], dtype='float32')
std = np.array([0.229, 0.224, 0.225], dtype='float32')


def preprocess_input(frame):
    # prepare the image to be classified by our deep learning network
    image = cv2.resize(frame, (299, 299))
    image = image.astype("float") / 255.0
    image -= mean
    image /= std
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image


data = np.load('decode.npy')[()]
# load the model
print("[INFO] loading model...")
original_model = load_model('model_distilled_Originalsqueezenet_model_T_0_lambda_0.h5')
model = SqueezeNet(weight_decay=1e-4, image_size=299)
model.load_weights('distilled_squeezenet_model_T_5_lambda_0.5.h5', by_name=True)
# model.load_weights('squeezenet_weights.hdf5', by_name=True)

print(model.summary())

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)


def get_frame(this_frame, this_model):
    this_frame = imutils.rotate_bound(this_frame, 180)
    this_frame = imutils.resize(this_frame, width=400)

    # prepare the image to be classified by our deep learning network
    image = preprocess_input(this_frame)

    # classify the input image and initialize the label and
    # probability of the prediction
    result = this_model.predict(image)
    temps = result[0].tolist()
    top_5_list = [temps.index(w) for w in sorted(temps)[-5:]][::-1]
    # for i in top_5_list:
    #	print('value at {} is {}'.format(i, data[i]))

    # print('##############')
    # update the label and prediction probability
    label_1 = data[top_5_list[0]]
    proba_1 = temps[top_5_list[0]]
    label_2 = data[top_5_list[1]]
    proba_2 = temps[top_5_list[1]]
    label_3 = data[top_5_list[2]]
    proba_3 = temps[top_5_list[2]]
    label_4 = data[top_5_list[3]]
    proba_4 = temps[top_5_list[3]]
    label_5 = data[top_5_list[4]]
    proba_5 = temps[top_5_list[4]]
    # build the label and draw it on the frame
    label_1 = "{}: {:.2f}%".format(label_1, proba_1 * 100)
    label_2 = "{}: {:.2f}%".format(label_2, proba_2 * 100)
    label_3 = "{}: {:.2f}%".format(label_3, proba_3 * 100)
    label_4 = "{}: {:.2f}%".format(label_4, proba_4 * 100)
    label_5 = "{}: {:.2f}%".format(label_5, proba_5 * 100)

    this_frame = cv2.putText(this_frame, label_1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    this_frame = cv2.putText(this_frame, label_2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    this_frame = cv2.putText(this_frame, label_3, (210, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    this_frame = cv2.putText(this_frame, label_4, (210, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    this_frame = cv2.putText(this_frame, label_5, (210, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return this_frame

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame_in = vs.read()
    frame = get_frame(frame_in, model)
    frame2 = get_frame(frame_in,original_model)
    # show the output frame
    cv2.imshow("With Distillation", frame)
    cv2.imshow("Without Distillation", frame2)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()



