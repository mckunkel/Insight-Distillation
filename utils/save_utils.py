
#Module to save the model
def save_utils(model, student_name,temperature, lambda_const, num_residuals=0):
    # Lets make some save strings first, some models will not depend on num_residuals
    if num_residuals == 0:
        model_name = "model_distilled_{}_model_T_{}_lambda_{}.h5".format(student_name, temperature, lambda_const)

    else:
        model_name = "model_distilled_{}_model_T_{}_lambda_{}_numResiduals_{}.h5".format(student_name, temperature, lambda_const,num_residuals)


    model.save(model_name)

    print("Saved model to disk")