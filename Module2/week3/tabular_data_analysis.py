#Download data
!gdown 1iA0WmVfW88HyJvTBSQDI5vesf-pgKabq

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('advertising.csv')

def correlation(X, Y):
  N = len(X)
  numerator = N * np.sum(X * Y) - np.sum(X) * np.sum(Y)
  denominator = math.sqrt(N * np.sum(X ** 2) - np.sum(X) ** 2) \
                * math.sqrt(N * np.sum(Y ** 2) - np.sum(Y) ** 2)
  return round(numerator / denominator, 2)

X = data['TV']
Y = data['Radio']
corr_xy = correlation(X, Y)
print(f'correlation between TV and Sales: {round(corr_xy, 2)}\n')

features = ['TV', 'Radio', 'Newspaper']

for feature_1 in features:
  for feature_2 in features:
    correlation_value = correlation(data[feature_1], data[feature_2])
    print(f'correlation between {feature_1} and {feature_2}: {round(correlation_value, 2)}')
print()

x = data['Radio']
y = data['Newspaper']
result = np.corrcoef(x, y)
print(result)

#Visualize data
sns.heatmap(data.corr(), fmt='.2f', annot=True)
