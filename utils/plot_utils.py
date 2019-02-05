import matplotlib.pyplot as plt


def plot_utils(model, student_name,temperature, lambda_const, num_residuals=0):
    #Lets make some save strings first, some models will not depend on num_residuals
    if num_residuals == 0:
        logloss_vs_epoch='student_{}_logloss_vs_epoch_T_{}_lambda_{}.png'.format(student_name, temperature, lambda_const)
        accuracy_vs_epoch='student_{}_accuracy_vs_epoch_T_{}_lambda_{}.png'.format(student_name, temperature, lambda_const)
        top5_accuracy_vs_epoch = 'student_{}_top5_accuracy_vs_epoch_T_{}_lambda_{}.png'.format(student_name, temperature,lambda_const)

    else:
        logloss_vs_epoch='student_{}_logloss_vs_epoch_T_{}_lambda_{}_numResiduals_{}.png'.format(student_name, temperature, lambda_const, num_residuals)
        accuracy_vs_epoch='student_{}_accuracy_vs_epoch_T_{}_lambda_{}_numResiduals_{}.png'.format(student_name, temperature, lambda_const, num_residuals)
        top5_accuracy_vs_epoch='student_{}_top5_accuracy_vs_epoch_T_{}_lambda_{}_numResiduals_{}.png'.format(student_name, temperature, lambda_const, num_residuals)


    # metric plots
    plt.plot(model.history.history['categorical_crossentropy'], label='train')
    plt.plot(model.history.history['val_categorical_crossentropy'], label='val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('logloss')
    plt.savefig(logloss_vs_epoch)
    plt.gcf().clear()

    plt.plot(model.history.history['accuracy'], label='train')
    plt.plot(model.history.history['val_accuracy'], label='val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(accuracy_vs_epoch)
    plt.gcf().clear()

    plt.plot(model.history.history['top_5_accuracy'], label='train')
    plt.plot(model.history.history['val_top_5_accuracy'], label='val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('top5_accuracy')
    plt.savefig(top5_accuracy_vs_epoch)
    plt.gcf().clear()