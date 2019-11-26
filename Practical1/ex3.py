import pandas as pd # Read data and work with data frames
import matplotlib.pyplot as plt # Boxplot and histogram
import statsmodels.api as sm # QQ plot
import scipy.stats as stats # Basic statistics
import numpy as np # Scientific computing
from pathlib import Path
from sample_from_simulated import sample_from_simulated

data = pd.read_table(Path.cwd() / "Practical1/spider_web.txt", sep=' ')

def uniform(sampleSize = 100, numSamples = 1000):

    np.random.seed(10)
    distribution = np.random.uniform(low=0, high=9, size=sampleSize)

    sample_from_simulated(distribution, sampleSize, numSamples, True, "uniform")

print(f"Performing simulated uniform sampling.")
print("======================================================\n\n")
uniform()
uniform(100, 10)
uniform(100, 100)
uniform(3)
uniform(10)
uniform(50)