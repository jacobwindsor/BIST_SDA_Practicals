import pandas as pd # Read data and work with data frames
import matplotlib.pyplot as plt # Boxplot and histogram
import statsmodels.api as sm # QQ plot
import scipy.stats as stats # Basic statistics
import numpy as np # Scientific computing
from pathlib import Path
from sample_from_simulated import sample_from_simulated

data = pd.read_table(Path.cwd() / "Practical1/spider_web.txt", sep=' ')

def binomial(column, sampleSize = 100, numSamples = 1000):
    mean = np.mean(data[column])
    std = np.std(data[column])

    np.random.seed(10)
    distribution = np.random.normal(size=sampleSize, loc=mean, scale=std)

    sample_from_simulated(distribution, sampleSize, numSamples)

for column in data.columns.delete(0):
    print(f"Performing simulated binomial sampling for {column}.")
    print("======================================================\n\n")
    binomial(column)
    binomial(column, 3, 10)
    binomial(column, 10, 50)