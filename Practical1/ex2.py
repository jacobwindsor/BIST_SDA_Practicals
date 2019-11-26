import pandas as pd # Read data and work with data frames
import matplotlib.pyplot as plt # Boxplot and histogram
import statsmodels.api as sm # QQ plot
import scipy.stats as stats # Basic statistics
import numpy as np # Scientific computing
from pathlib import Path
from sample_from_simulated import sample_from_simulated

def binomial(sampleSize = 100, numSamples = 1000):
    np.random.seed(10)
    distribution = np.random.binomial(1000, 0.5, sampleSize)

    sample_from_simulated(distribution, sampleSize, numSamples)

print(f"Performing simulated binomial sampling.")
print("======================================================\n\n")
binomial()
binomial(3, 10)
binomial(10, 50)