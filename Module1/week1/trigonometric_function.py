#câu 4
def is_int(x):
  return x.isnumeric()

x = input('Nhập x: ')

n = input('Nhập n: ')

if not is_int(n):
  print('number of samples must be an integer number')
  sys.exit()

if is_int(n) and int(n) <= 0:
  print('number of samples must be greater than 0')
  sys.exit()

x = int(x)
n = int(n)

def factorial(n):
  if n == 0:
    return 1
  return n * factorial(n-1)

def sin(x, n):
  res = 0
  for i in range(n):
    res += ((-1)**i) * (x**(2*i+1)/(factorial(2*i+1)))
  return res

def cos(x, n):
  res = 0
  for i in range(n):
    res += ((-1)**i) * (x**(2*i)/(factorial(2*i)))
  return res

def sinh(x,n):
  res = 0
  for i in range(n):
    res += (x**(2*i+1)/(factorial(2*i+1)))
  return res

def cosh(x,n):
  res = 0
  for i in range(n):
    res += (x**(2*i)/(factorial(2*i)))
  return res

cosh(x, n)
