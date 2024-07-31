def compute_cosine(v1, v2):
  cos_sim = np.dot(v1, v2) / (compute_vector_length(v1) * compute_vector_length(v2))
  return cos_sim

x = [1, 2, 3, 4]
y = [1, 0, 3, 0]

print(round(compute_cosine(x, y), 3))
