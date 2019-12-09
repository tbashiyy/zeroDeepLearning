import numpy as np

def step_function(x):
  return np.array(x>0, dtype=np.int)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))