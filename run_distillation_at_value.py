import sys
from distillsqueezenet import distill as ds
from distill_model import distill as dsXcept
from utils.check_file_exists import check_utils as check_file

def run(type, part):
    if type == 'squeeze':
        distill_squeeze()
    elif type == 'xception':
        distill_Xception(part)

def distill_squeeze():
    temps = [2.5, 5, 10, 15]
    lamdas = [0.02, 0.2, 0.5, 1]

    list3 = [(x, y) for x in temps for y in lamdas]

    for temperature, lambda_constant in list3:
        ds(temperature, lambda_constant)

def distill_Xception(part):
    if part == 1:
        temps = [2.5, 5]
        lamdas = [0.02, 0.2, 0.5, 1]
        residuals = range(7)
        list3 = [(x, y, z) for x in temps for y in lamdas for z in residuals]
        for temperature, lambda_constant, residual in list3:
            if check_file('miniXception',temperature, lambda_constant, residual):
                print('File exists already. Will not redo')
                continue
            dsXcept(temperature, lambda_constant, residual)
    elif part == 2:
        temps = [10, 15]
        lamdas = [0.02, 0.2, 0.5, 1]
        residuals = range(7)
        list3 = [(x, y, z) for x in temps for y in lamdas for z in residuals]
        for temperature, lambda_constant, residual in list3:
            if check_file('miniXception',temperature, lambda_constant, residual):
                print('File exists already. Will not redo')
                continue
            dsXcept(temperature, lambda_constant, residual)
    else:
        print('The value {} is not valid please choose 1 or 2'.format(part))



if __name__ == '__main__':
    _type = sys.argv[1]
    _part = int(sys.argv[2])
    run(_type, _part)