import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

X, y = datasets.make_moons(n_samples=200, shuffle=True, noise=0.2, random_state=1234)
y = np.reshape(y, (len(y), 1))
