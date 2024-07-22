#câu 1
tp = 0.8
fp = -3
fn = 4.1

def calculate(tp, fp, fn):
  arr = [tp, fp, fn]
  name = ['tp', 'fp', 'fn']
  strArr = []
  for x in arr:
    if x <= 0:
      strArr.append(f'{name[arr.index(x)]} must be greater than 0')
    if type(x) != int:
      strArr.append(f'{name[arr.index(x)]} must be int')
  if len(strArr) > 0:
    for _ in strArr:
      print(_)
    return
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1 = 2*(precision*recall)/(precision+recall)
  print(f'Precision: {precision}')
  print(f'Recall: {recall}')
  print(f'F1: {f1}')

calculate(tp, fp, fn)
