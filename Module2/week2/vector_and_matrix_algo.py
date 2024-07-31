import numpy as np

def compute_vector_length(vector):
  len_of_vector = np.sqrt(np.sum(np.square(vector)))
  return len_of_vector

def compute_dot_product(vector1, vector2):
  result = np.dot(vector1, vector2)
  return result

def matrix_multi_vector(matrix, vector):
  if matrix.shape[1] != vector.shape[0]:
        raise ValueError("Matrix columns must match vector size")
  result = np.dot(matrix, vector)
  return result

def matrix_multi_matrix(matrix1, matrix2):
  result = np.dot(matrix1, matrix2)
  return result

def inverse_matrix(matrix):
  determinant = np.linalg.det(matrix)

  if determinant == 0:
    raise ValueError("Singular matrix, cannot compute its inverse")

  result = np.linalg.inv(matrix)
  return result

v = np.array([0, 1, -1, 2])
u = np.array([2, 5, 1, 0])

matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix2 = np.array([[1, 3, 7], [2, 9, 10], [5, 5, -1]])

print(compute_vector_length(v))
print(compute_dot_product(v, u))
print()

try:
  print(matrix_multi_vector(matrix1, v))
except ValueError as e:
  print(e)

print('\n', matrix_multi_matrix(matrix1, matrix2), '\n')

try:
    print(inverse_matrix(matrix1))
except ValueError as e:
    print(e)

