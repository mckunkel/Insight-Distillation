import os.path

def check_utils(student_name,temperature, lambda_const, num_residuals=0):
    if num_residuals == 0:
        weight_name = "distilled_{}_model_T_{}_lambda_{}.h5".format(student_name, temperature, lambda_const)
    else:
        weight_name = 'distilled_{}_model_T_{}_lambda_{}_numResiduals_{}.h5'.format(student_name, temperature, lambda_const, num_residuals)

    return os.path.isfile(weight_name)

