import math

def max_element_in_slider(num_list, k):
  result_lst = []
  for i in range(len(num_list)):
    if i + k <= len(num_list):
      sub_list = []
      for j in range(3):
        sub_list.append(num_list[i])
      result_lst.append(max(sub_list))
    else:
      sub_list = [num_list[-3], num_list[-2], num_list[-1]]
      result_lst.append(max(sub_list))
  return result_lst

if __name__ == '__main__':
    num_list = [3, 4, 5, 1, -44 , 5 ,10, 12 ,33, 1]
    k = 3
    print(max_element_in_slider(num_list, k))
