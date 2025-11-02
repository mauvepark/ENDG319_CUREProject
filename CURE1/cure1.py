
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

emissions = DataFrame()

emissions["Low Altitude"] = [1.50, 1.48, 2.98, 1.40, 3.12, 0.25, 6.73, 5.30, 9.30, 6.69, 7.21, 0.87, 1.06, 7.39, 1.37]
emissions["High Altitude"] = [7.59, 2.06, 8.86, 8.67, 5.61, 6.28, 4.04, 4.40, 9.52, 1.50, 6.07, 17.11, 3.57, 2.68, 6.46]


emissions.plot.box(column=['Low Altitude', 'High Altitude'])

# task 3

dataset1 = DataFrame()
dataset1["x"] = [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0]
dataset1["y"] = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]

dataset2 = DataFrame()
dataset2["x"] = [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0]
dataset2["y"] = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]

print(dataset1.describe())
print(dataset2.describe())

digits_dataset = load_breast_cancer()

print(digits_dataset.keys())
print(digits_dataset.DESCR)
