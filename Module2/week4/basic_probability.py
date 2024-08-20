import numpy as np
import math

def compute_mean(X):
  sum = 0
  for _ in X:
    sum = sum + _
  return sum/len(X)

def compute_median(X):
  size = len(X)
  X = np.sort(X)
  if size % 2 == 0:
    return (X[size // 2] + X[size // 2 - 1]) / 2
  else:
    return X[size // 2]

def compute_variance(X):
  N = len(X)
  mean = compute_mean(X)
  sum = 0
  for _ in X:
    sum = sum + (_ - mean) ** 2
  return sum / N

def compute_std(X):
  return round(math.sqrt(compute_variance(X)), 2)

def compute_correlation_cofficient(X, Y):
  N = len(X)
  numerator = N * np.sum(X * Y) - np.sum(X) * np.sum(Y)
  denominator = math.sqrt(N * np.sum(X ** 2) - np.sum(X) ** 2) \
                * math.sqrt(N * np.sum(Y ** 2) - np.sum(Y) ** 2)
  return round(numerator / denominator, 2)

X = [2, 0, 2, 2, 7, 4, -2, 5, -1, -1]
print('mean:', compute_mean(X))
X = [1, 5, 4, 4, 9, 13]
print('median:', compute_median(X))
X = [ 171 , 176 , 155 , 167 , 169 , 182]
print('std:', compute_std(X))
X = np.array([-2 , -5 , -11 , 6 , 4 , 15 , 9])
Y = np.array([4 , 25 , 121 , 36 , 16 , 225 , 81])
print('correlation:', compute_correlation_cofficient(X, Y))

