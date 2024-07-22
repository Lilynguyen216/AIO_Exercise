# Download data
!gdown '1iA0WmVfW88HyJvTBSQDI5vesf-pgKabq'

import pandas as pd
import numpy as np

df = pd.read_csv('./advertising.csv')

data = df.to_numpy()

sales = data[:, 3]  
max_sales = np.max(sales)
max_index = np.argmax(sales)

print(f"Max value: {max_sales} - Index: {max_index}")

tv = data[:, 0]  
mean_tv = np.mean(tv)

print(f"\nMean TV: {mean_tv}")

sales_more_than_20_count = np.sum(sales >= 20)
print(f"\nSales more than 20: {sales_more_than_20_count}")

conditional_radio_lst = data[:, 1][sales >= 15]
conditional_radio_avg = np.mean(conditional_radio_lst)
print(f"\nConditional radio: {conditional_radio_avg}")

conditonal_sales_lst = sales[data[:,2] > np.mean(data[:,2])]
conditonal_sales_sum = np.sum(conditonal_sales_lst)
print(f"\nConditional sales: {conditonal_sales_sum}")

A = np.mean(sales)

# Điều kiện lọc
scores = ['Good' if sale > A else 'Average' if sale == A else 'Bad' for sale in sales]
df['Scores'] = scores
data = df.to_numpy()

print(scores[7:10])

#bài cuối
A1 = min(sales - A)
def classify_sales(value, average):
    if value > average:
        return 'Good'
    elif value < average:
        return 'Bad'
    else:
        return 'Average'

# Apply the classification function to the Sales column
df['Classification'] = df['Sales'].apply(lambda x: classify_sales(x, A1))

print(df['Classification'][7:10].values)
