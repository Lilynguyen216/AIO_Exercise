#câu 2
import numpy as np
import math
import sys

function_name_array = ['sigmoid', 'elu', 'relu']

def is_number(x):
  try:
    float(x)
    return True
  except:
    print(f'{x} must be a number')
    return False

def check_activation_function(function_name):
  for x in function_name_array:
    if x == function_name.lower():
      return True
      return True
  print(f'{function_name.lower()} is not supported ')
  return False

def sigmoid(x):
  return 1/(1+np.exp(-x))

def relu(x):
  return max(0, x)

def elu(x):
  if x > 0:
    return x
  else:
    return 0.01*(math.exp(x)-1)

x = input('Nhập x: ')
if is_number(x) == False:
  sys.exit()

activation_function = input('Nhập activation function: ')
if check_activation_function(activation_function) == False:
  sys.exit()

if is_number(x) == True and check_activation_function(activation_function) == True:
  if activation_function.lower() == 'sigmoid':
    print(f'sigmoid: f({x}) = {sigmoid(float(x))}')
  elif activation_function.lower() == 'relu':
    print(f'elu: f({x}) = {elu(float(x))}')
  elif activation_function.lower() == 'elu':
    print(f'relu: f({x}) = {relu(float(x))}')
