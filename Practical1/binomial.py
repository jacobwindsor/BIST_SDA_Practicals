import pandas as pd # Read data and work with data frames
import matplotlib.pyplot as plt # Boxplot and histogram
import statsmodels.api as sm # QQ plot
import scipy.stats as stats # Basic statistics
import numpy as np # Scientific computing
from pathlib import Path

data = pd.read_table(Path.cwd() / "Practical1/spider_web.txt", sep=' ')


def sample_from_simulated(column, sizeSamples = 100, numSamples = 1000, showGraph = False):
    print(f"Performing simulated binomial sampling for {column}.")
    print(f"numSamples = {numSamples}, sizeSample = {sizeSamples}")
    print(f"===================================================== \n")
    mean = np.mean(data[column])
    std = np.std(data[column])

    np.random.seed(10)
    distribution = np.random.normal(size=sizeSamples, loc=mean, scale=std)
    samples = np.random.choice(a=distribution, size=numSamples, replace=True)

    plt.hist(np.fromiter(samples, dtype=float))
    showGraph and plt.show()

    sample_mean = np.mean(samples)
    similiar_text = ("not ", "")[abs(mean - sample_mean) < 10]
    print(f"The mean of the population is {mean} and the mean of the sampling distribution of the mean is {sample_mean}. They are {similiar_text}similiar.")

    sample_std = np.std(samples)
    similiar_std_text = ("not ", "")[abs(std - sample_std) < 5]
    print(f"The std of the population is {std} and the std of the sampling distribution of the mean is {sample_std}. They are {similiar_std_text}similiar.")

    estimated_std = sample_std / np.sqrt(sizeSamples)
    clt_holds_txt = ("not ", "")[abs(estimated_std - sample_std) < 1]
    print(f"The estimated std of the population is {estimated_std}. So, the CLT does {clt_holds_txt}estimate the sample standard deviation.")

    print("\n\n")


for column in ['HORIZDIM', 'VERTDIM', 'HORIZLIG', 'VERTLIG']:
    sample_from_simulated(column)
    # sample_from_simulated(column, 3, 10)
    # sample_from_simulated(column, 10, 50)