import math
import random
import sys

loss_name_lst = ['mae', 'mse', 'rmse']

def is_int(x):
    return x.isnumeric()

def check_loss_name(loss_name):
    return loss_name.lower() in loss_name_lst

def mae(x):
    sum_loss = 0
    predict_arr = []
    target_arr = []
    for i in range(x):
        predict_arr.append(random.uniform(0, 10))
        target_arr.append(random.uniform(0, 10))
        sum_loss += abs(target_arr[i] - predict_arr[i])
    return predict_arr, target_arr, sum_loss / x

def mse(x):
    sum_loss = 0
    predict_arr = []
    target_arr = []
    for i in range(x):
        predict_arr.append(random.uniform(0, 10))
        target_arr.append(random.uniform(0, 10))
        sum_loss += (target_arr[i] - predict_arr[i]) ** 2
    return predict_arr, target_arr, sum_loss / x

def rmse(x):
    result = mse(x)
    return result[0], result[1], math.sqrt(result[2])

x = input('Nhập x: ')
if not is_int(x):
    print('number of samples must be an integer number')
    sys.exit()

x = int(x)

loss_name = input('Nhập loss name: ')
if not check_loss_name(loss_name):
    print(f'{loss_name.lower()} is not supported')
    sys.exit()

function_lst = [mae, mse, rmse]

result = function_lst[loss_name_lst.index(loss_name.lower())](x)

predict_arr = result[0]
target_arr = result[1]
loss = result[2]

print('')

for i in range(x):
    print(f'loss name: {loss_name.lower()} - sample: {i} - predict: {predict_arr[i]} - target: {target_arr[i]}')

print(f'\nloss: {loss}')
