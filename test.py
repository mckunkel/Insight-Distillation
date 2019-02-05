temps = [2.5, 5, 10, 15]
lamdas = [0.02, 0.2, 0.5, 1]
residuals = range(7)
list3 = [(x, y, z) for x in temps for y in lamdas for z in residuals]
for temperature, lambda_constant, residual in list3:
    print(temperature, lambda_constant, residual)
