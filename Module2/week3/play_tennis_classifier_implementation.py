import numpy as np

data=[['Sunny','Hot', 'High', 'Weak', 'no'],
        ['Sunny','Hot', 'High', 'Strong', 'no'],
        ['Overcast','Hot', 'High', 'Weak', 'yes'],
        ['Rain','Mild', 'High', 'Weak', 'yes'],
        ['Rain','Cool', 'Normal', 'Weak', 'yes'],
        ['Rain','Cool', 'Normal', 'Strong', 'no'],
        ['Overcast','Cool', 'Normal', 'Strong', 'yes'],
        ['Overcast','Mild', 'High', 'Weak', 'no'],
        ['Sunny','Cool', 'Normal', 'Weak', 'yes'],
        ['Rain','Mild', 'Normal', 'Weak', 'yes']
        ]

train_data = np.array(data)

def compute_prior_probabilities(train_data, target_column):
    unique_targets, counts = np.unique(train_data[:, target_column], return_counts=True)
    prior_probabilities = dict(zip(unique_targets, counts / len(train_data)))
    return list(prior_probabilities.values())

#compute_prior_probabilities(train_data, -1)

def compute_conditional_probabilities(train_data, feature_column, target_column):
  feature_column_unique = np.unique(train_data[:, feature_column])
  target_column_unique = np.unique(train_data[:, target_column])
  conditional_probabilities = np.zeros((len(target_column_unique), len(feature_column_unique)))

  for j in range(0,len(target_column_unique)):
    for i in range(0,len(feature_column_unique)):
      conditional_probabilities[j, i] = len(np.where((train_data[:, feature_column] == feature_column_unique[i]) & (train_data[:, target_column] == target_column_unique[j]))[0]) / len(np.where(train_data[:, target_column] == target_column_unique[j])[0])
  
  return conditional_probabilities, feature_column_unique

def compute_full_conditional_probabilities(train_data):
  full_conditional_probabilities = []
  full_feature_columns_unique = []

  for i in range(0, len(train_data[0])-1):
    res = compute_conditional_probabilities(train_data, i, -1)

    full_conditional_probabilities.append(res[0])
    full_feature_columns_unique.append(res[1])
  
  return full_conditional_probabilities, full_feature_columns_unique

compute_full_conditional_probabilities(train_data)


def train_naive_bayes(train_data):
    # Step 1: Calculate Prior Probability
    prior_probability = compute_prior_probabilities(train_data, -1)

    # Step 2: Calculate Conditional Probability
    conditional_probability, list_x_name  = compute_full_conditional_probabilities(train_data)

    return prior_probability,conditional_probability, list_x_name


def get_index_from_value(feature_name, list_features):
  return np.where(list_features == feature_name)[0][0]


def prediction_play_tennis(X, list_x_name, prior_probability, conditional_probability):

    x1=get_index_from_value(X[0],list_x_name[0])
    x2=get_index_from_value(X[1],list_x_name[1])
    x3=get_index_from_value(X[2],list_x_name[2])
    x4=get_index_from_value(X[3],list_x_name[3])

    p0=prior_probability[0] \
    *conditional_probability[0][0,x1] \
    *conditional_probability[1][0,x2] \
    *conditional_probability[2][0,x3] \
    *conditional_probability[3][0,x4]

    p1=prior_probability[1]\
    *conditional_probability[0][1,x1]\
    *conditional_probability[1][1,x2]\
    *conditional_probability[2][1,x3]\
    *conditional_probability[3][1,x4]

    # print(p0, p1)

    if p0>p1:
        y_pred=0
    else:
        y_pred=1

    return y_pred


#compute_prior_probabilities(train_data, -1)

X = ['Sunny','Cool', 'High', 'Strong']
prior_probability, conditional_probability, list_x_name = train_naive_bayes(train_data)
pred =  prediction_play_tennis(X, list_x_name, prior_probability, conditional_probability)

if(pred):
  print("Ad should go!")
else:
  print("Ad should not go!")
