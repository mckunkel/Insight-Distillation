from keras.losses import categorical_crossentropy as logloss
from keras import backend as K


def knowledge_distillation_loss(y_true, y_pred, lambda_const,temperature):
    # split in
    #    onehot hard true targets
    #    logits from xception
    y_true, logits = y_true[:, :256], y_true[:, 256:]

    # convert logits to soft targets
    y_soft = K.softmax(logits / temperature) * temperature

    # split in
    #    usual output probabilities
    #    probabilities made softer with temperature
    y_pred, y_pred_soft = y_pred[:, :256], y_pred[:, 256:]

    return lambda_const * logloss(y_true, y_pred) + logloss(y_soft, y_pred_soft)