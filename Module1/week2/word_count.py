file_path = './week2/P1_data.txt'
with open(file_path, 'r') as f:
  data = f.read()
data = data.lower()
data = data.split()

def get_num_count_list(data):
  num_count_list = {}
  for word in data:
    num_count_list[word] = num_count_list.get(word, 0) + 1
  return num_count_list

print(get_num_count_list(data))
