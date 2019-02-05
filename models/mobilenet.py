import keras, sys
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Activation, GlobalAveragePooling2D, Dropout, Dense, Input


def get_mobilenet(input_size, alpha, weight_decay, dropout):
    input_shape = (input_size, input_size, 3)
    base_model = MobileNet(
        include_top=False, weights='imagenet',
        input_shape=input_shape, alpha=alpha
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    logits = Dense(256, kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    probabilities = Activation('softmax')(logits)
    model = Model(base_model.input, probabilities)

    for layer in model.layers[:-2]:
        layer.trainable = False

    orig_stdout = sys.stdout
    f = open('mobilenetSummary.txt', 'w')
    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout
    f.close()

    return model

if __name__ == '__main__':
    get_mobilenet(224, alpha=0.25, weight_decay=1e-5, dropout=0.1)