import numpy as np
import constants as c

data_dir = c.data_dir

# save plots of model
def history_utils(model, student_name,temperature, lambda_const, num_residuals=0):
    #Lets make some save strings first, some models will not depend on num_residuals
    if num_residuals == 0:
        categorical_crossentropy='student_{}_categorical_crossentropy_T_{}_lambda_{}.npy'.format(student_name, temperature, lambda_const)
        val_categorical_crossentropy='student_{}_val_categorical_crossentropy_T_{}_lambda_{}.npy'.format(student_name, temperature, lambda_const)
        accuracy = 'student_{}_accuracy_T_{}_lambda_{}.npy'.format(student_name, temperature,lambda_const)
        val_accuracy = 'student_{}_val_accuracy_T_{}_lambda_{}.npy'.format(student_name, temperature, lambda_const)
        top_5_accuracy = 'student_{}_top_5_accuracy_T_{}_lambda_{}.npy'.format(student_name, temperature, lambda_const)
        val_top_5_accuracy = 'student_{}_val_top_5_accuracy_T_{}_lambda_{}.npy'.format(student_name, temperature, lambda_const)

    else:
        categorical_crossentropy = 'student_{}_categorical_crossentropy_T_{}_lambda_{}_numResiduals_{}.npy'.format(student_name,
                                                                                               temperature,
                                                                                               lambda_const, num_residuals)
        val_categorical_crossentropy = 'student_{}_val_categorical_crossentropy_T_{}_lambda_{}_numResiduals_{}.npy'.format(student_name,
                                                                                                       temperature,
                                                                                                       lambda_const, num_residuals)
        accuracy = 'student_{}_accuracy_T_{}_lambda_{}_numResiduals_{}.npy'.format(student_name, temperature, lambda_const, num_residuals)
        val_accuracy = 'student_{}_val_accuracy_T_{}_lambda_{}_numResiduals_{}.npy'.format(student_name, temperature, lambda_const, num_residuals)
        top_5_accuracy = 'student_{}_top_5_accuracy_T_{}_lambda_{}_numResiduals_{}.npy'.format(student_name, temperature, lambda_const, num_residuals)
        val_top_5_accuracy = 'student_{}_val_top_5_accuracy_T_{}_lambda_{}_numResiduals_{}.npy'.format(student_name, temperature,
                                                                                   lambda_const, num_residuals)

    np.save(categorical_crossentropy, model.history.history['categorical_crossentropy'])
    np.save(val_categorical_crossentropy, model.history.history['val_categorical_crossentropy'])

    np.save(accuracy, model.history.history['accuracy'])
    np.save(val_accuracy, model.history.history['val_accuracy'])

    np.save(top_5_accuracy, model.history.history['top_5_accuracy'])
    np.save(val_top_5_accuracy, model.history.history['val_top_5_accuracy'])
