import pandas as pd # Read data and work with data frames
import matplotlib.pyplot as plt # Boxplot and histogram
import statsmodels.api as sm # QQ plot
import scipy.stats as stats # Basic statistics
import numpy as np # Scientific computing
from pathlib import Path

def _doSampling(dat, numSamples, sampleSize):
    # calculate and plot a sampling distribution of the mean # parameters given are:
    # dat = data to sample from (numeric vector),
    # numSamples = number of times you want to sample (number)
    # sampleSize = size of the sample (number)
    for i in range(numSamples):
        # simple random sampling
        get_sample = np.random.choice(dat, size = sampleSize, replace = True) 
        # calculate mean and return it
        yield np.mean(get_sample)


def sample_from_simulated(distribution, sampleSize = 100, numSamples = 1000, showGraph = True, prefix=""):    
    print(f"number of samples = {numSamples}, sample size = {sampleSize}")
    print(f"===================================================== \n")

    samples = np.fromiter(_doSampling(distribution, sampleSize, numSamples), dtype=float)

    plt.hist(np.fromiter(samples, dtype=float))
    showGraph and plt.savefig(Path.cwd() / f"Practical1/graphs/{prefix}_hist_sampleSize{sampleSize}_numSamples{numSamples}.png")

    sample_mean = np.mean(samples)
    pop_mean = np.mean(distribution)
    similiar_text = ("not ", "")[abs(pop_mean - sample_mean) < 10]
    print(f"The mean of the population is {pop_mean} and the mean of the sampling distribution of the mean is {sample_mean}. They are {similiar_text}similiar.")

    sample_std = np.std(samples)
    pop_std = np.std(distribution)
    similiar_std_text = ("not ", "")[abs(pop_std - sample_std) < 5]
    print(f"The std of the population is {pop_std} and the std of the sampling distribution of the mean is {sample_std}. They are {similiar_std_text}similiar.")

    estimated_std = pop_std / np.sqrt(numSamples)
    clt_holds_txt = ("not ", "")[abs(estimated_std - sample_std) < 1]
    print(f"The estimated std of the population is {estimated_std}. So, the CLT does {clt_holds_txt}estimate the sample standard deviation.")

    print("\n")