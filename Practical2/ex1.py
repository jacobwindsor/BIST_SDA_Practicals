import pandas as pd
import statsmodels as sm
import statsmodels.api as statsmodels
from statsmodels.formula.api import ols
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

np.random.seed(10)

""" Excercise 1
Normal distribution ANOVA
"""

def experiment():
    # Sample 4 (n=20) samples from normal distribution
    index_column = []
    data_column = []
    for i in range(1,5):
        index_column.extend([str(i)] * 20) # Convert i to string so is treated as a categorical variable
        data_column.extend(np.random.normal(loc=164, scale=math.sqrt(24), size=20))

    samples = pd.DataFrame(
        data={"index": index_column, "data": data_column}
    )

    sample_stats = []
    for i in range(1,5):
        sample = samples.loc[samples["index"] == str(i)]
        sample_stats.append({
            "sample_label": str(i),
            "mean": np.mean(sample["data"]),
            "variance": np.var(sample["data"])
        })

    # Calculate MSbetween and MSwithin
    mod = ols("data ~ index", data=samples).fit()
    return { 
        "a_tab": sm.stats.anova.anova_lm(mod),
        "sample_stats": sample_stats
    }

print("Performing ANOVA for one repeat of experiment \n")
one_experiment = experiment()
print(one_experiment["a_tab"])
print("\n")

for sample in one_experiment["sample_stats"]:
    print(f"Mean for sample {sample['sample_label']} is {sample['mean']} and the variance is {sample['variance']}")

print("Given the p value. Sample means are likely not statistically different. \n")
print("MSB and MSW are not similiar. Neither are good estimates of the variance of the normal distribution. Although, MSW is closer to 24. \n")

print("Performing ANOVA for 1000 repeats of experiment \n")

fvalues = []
for x in range(0, 999):
    anova = experiment()["a_tab"]
    fvalues.append(anova["F"]["index"])

plt.hist(
    fvalues,
    density=True, # plot a density function for probabilities
    bins=np.arange(min(fvalues), max(fvalues) + 0.3, 0.3)
)
plt.xlabel("F value")
plt.ylabel("Probability")
fig_path = Path.cwd() / "Practical2/graphs/fdist_ex1.png"
print(f"See {fig_path} for output f distribution \n")
plt.savefig(fig_path)

print("Using this distribution, given alpha of 0.5, the Fc would be approximately 3.5-4")

cdf = stats.f.cdf(3.5, dfn=3, dfd=26)

print(f"For a theoretical T distribution with the same degrees of freedom, {round(cdf * 100, 3)}% of F values lie to the left of 3.5.")
print("Therefore, the given Fc value of 3.5-4.0 is quite accurate.")