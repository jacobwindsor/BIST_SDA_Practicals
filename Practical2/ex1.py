import pandas as pd
import statsmodels as sm
import statsmodels.api as statsmodels
from statsmodels.formula.api import ols
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(10)

""" Excercise 1
Normal distribution ANOVA
"""

def experiment():
    # Sample 4 (n=20) samples from normal distribution
    samples =  [{},{},{},{}]
    # Calculate mean and variance
    for idx, sample in enumerate(samples):
        sample["index"] = idx # Bit of a hack to get the head looking good for ols
        sample["data"] = np.random.normal(loc=16, scale=24, size=20)
        sample["mean"] = np.mean(sample["data"])
        sample["variance"] = np.var(sample["data"])

    # Convert to DF
    samples = pd.DataFrame(samples)

    # Calculate MSbetween and MSwithin
    mod = ols("mean ~ index", data=samples).fit()
    return sm.stats.anova.anova_lm(mod)

print("Performing ANOVA for one repeat of experiment \n")
print(experiment())

print("MSW and MSB are similiar but both are far from the variance of 24 given to the normal distribution. \n")

print("Performing ANOVA for 1000 repeats of experiment \n")

fvalues = []
for x in range(0, 999):
    anova = experiment()
    fvalues.append(anova["F"]["index"])

plt.hist(np.fromiter(fvalues, dtype=float))
plt.savefig(Path.cwd() / "Practical2/graphs/hist_ex1.png")

print("Note that distribution is very uniform since numpy.random has been seeded. \n")
