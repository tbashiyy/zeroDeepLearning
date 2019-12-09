import numpy as np

def step_function(x):
  return np.array(x>0, dtype=np.int)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def softmax(x):
  c = np.max(x)
  exp_a = np.exp(x-c)
  sum_exp_a = np.sum(exp_a)
  return exp_a/sum_exp_a