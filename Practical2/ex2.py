import pandas as pd
import statsmodels as sm
import statsmodels.api as statsmodels
from statsmodels.formula.api import ols
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.sandbox.stats.multicomp import tukeyhsd

"""Excercise 2
Zinc contamination
"""

data = pd.read_table(Path.cwd() / "Practical2/contamination.txt", sep=' ')
print(data.head())

# Make box plot
data.boxplot(column="DIVERSITY", by="ZINC")
plt.show()

# h0 = group means are all equal
mod = ols("DIVERSITY ~ ZINC", data=data).fit()
results = sm.stats.anova.anova_lm(mod)
print("ANOVA TABLE for diversity versus zinc level groups")
print(results)

print("P is below alpha of 0.05. So reject null hypothesis. The means are not equal.")

# Perform post hoc test
print("Performing post hoc test...")
mc = MultiComparison(data["DIVERSITY"], data["ZINC"])
print(mc.tukeyhsd())
print("Can conclude that group2 with LOW zinc contamination is the group with a significantly different mean")

