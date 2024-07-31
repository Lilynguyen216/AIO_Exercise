import numpy as np
import math

# Load dữ liệu
data = np.loadtxt('iris.data.txt', delimiter=',', dtype=str)

numeric_cols = data[:, :4].astype(float)
str_col = data[:, -1:].reshape((-1, 1))

train_data = np.empty(data.shape, dtype=object)

train_data[:, :4] = numeric_cols
train_data[:, -1:] = str_col

train_data = np.array(train_data, dtype=object)


def compute_prior_probabilities(train_data):
    unique_targets, counts = np.unique(train_data[:, -1], return_counts=True)
    prior_probabilities = dict(zip(unique_targets, counts / len(train_data)))
    return list(prior_probabilities.values())


def compute_conditional_probabilities(train_data):
  target_col_unique = np.unique(train_data[:,4]) # 0 for Setosa, 1 for Versicolour, 2 for Virginica
  x_feature = 4
  conditional_probability = []
  for i in range(0,train_data.shape[1]-1):
    x_conditional_probability = np.zeros((len(target_col_unique), 2))
    for j in range(0,len(target_col_unique)):
        mean = np.mean((train_data[:,i][np.where(train_data[:,4] == target_col_unique[j])]))
        sigma =  np.std((train_data[:,i][np.where(train_data[:,4] == target_col_unique[j])]))
        sigma = sigma * sigma
        x_conditional_probability[j]= [mean, sigma]

    conditional_probability.append(x_conditional_probability)
  return conditional_probability


def train_gaussian_naive_bayes(train_data):
    # Step 1: Calculate Prior Probability
    prior_probability = compute_prior_probabilities(train_data)

    # Step 2: Calculate Conditional Probability
    conditional_probability = compute_conditional_probabilities(train_data)

    return prior_probability,conditional_probability


def gauss(x, mean, sigma):
  result = (1.0 / (np.sqrt(2*math.pi*sigma))) \
  * (np.exp(-(float(x) - mean) ** 2 / (2 * sigma)))
  return result


# Prediction
def prediction_iris(X,  prior_probability, conditional_probability):

    p0=prior_probability[0] \
    *gauss(X[0], conditional_probability[0][0][0],conditional_probability[0][0][1])  \
    *gauss(X[1], conditional_probability[1][0][0],conditional_probability[1][0][1])  \
    *gauss(X[2], conditional_probability[2][0][0],conditional_probability[2][0][1])  \
    *gauss(X[3], conditional_probability[3][0][0],conditional_probability[3][0][1])

    p1=prior_probability[1] \
    *gauss(X[0], conditional_probability[0][1][0],conditional_probability[0][1][1])  \
    *gauss(X[1], conditional_probability[1][1][0],conditional_probability[1][1][1])  \
    *gauss(X[2], conditional_probability[2][1][0],conditional_probability[2][1][1])  \
    *gauss(X[3], conditional_probability[3][1][0],conditional_probability[3][1][1])

    p2=prior_probability[2] \
    *gauss(X[0], conditional_probability[0][2][0],conditional_probability[0][2][1])  \
    *gauss(X[1], conditional_probability[1][2][0],conditional_probability[1][2][1])  \
    *gauss(X[2], conditional_probability[2][2][0],conditional_probability[2][2][1])  \
    *gauss(X[3], conditional_probability[3][2][0],conditional_probability[3][2][1])

    # print(p0, p1)

    list_p = [p0, p1, p2]

    return list_p.index(np.max(list_p))

X = [6.3 , 3.3, 6.0,  2.5]
prior_probability, conditional_probability = train_gaussian_naive_bayes(train_data)
idx =  prediction_iris(X, prior_probability, conditional_probability)
pred = np.unique(train_data[:,-1])[idx]
print(pred)
