

def save_utils(model, student_name,temperature, lambda_const, num_residuals=0):
    # Lets make some save strings first, some models will not depend on num_residuals
    if num_residuals == 0:
        json_name = "distilled_{}_model_T_{}_lambda_{}.json".format(student_name, temperature, lambda_const)
        yaml_name = "distilled_{}_model_T_{}_lambda_{}.yaml".format(student_name, temperature, lambda_const)
        weight_name = "distilled_{}_model_T_{}_lambda_{}.h5".format(student_name, temperature, lambda_const)
        model_name = "model_distilled_{}_model_T_{}_lambda_{}.h5".format(student_name, temperature, lambda_const)

    else:
        json_name = 'distilled_{}_model_T_{}_lambda_{}_numResiduals_{}.json'.format(student_name, temperature, lambda_const, num_residuals)
        yaml_name = 'distilled_{}_model_T_{}_lambda_{}_numResiduals_{}.yaml'.format(student_name, temperature, lambda_const, num_residuals)
        weight_name = 'distilled_{}_model_T_{}_lambda_{}_numResiduals_{}.h5'.format(student_name, temperature, lambda_const, num_residuals)
        model_name = "model_distilled_{}_model_T_{}_lambda_{}_numResiduals_{}.h5".format(student_name, temperature, lambda_const,num_residuals)



    # serialize model to JSON
    model_json = model.to_json()
    with open(json_name, "w") as json_file:
        json_file.write(model_json)


    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(yaml_name, "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights(weight_name)
    model.save(model_name)

    print("Saved model to disk")