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
import math
from statsmodels.graphics.factorplots import interaction_plot

""""Ex 1 ANOVA 2 (ex 4, practical 2)
Limpets ANOVA
"""

# Create the data 
spring = [1.167, 0.5, 1.667, 1.5, 0.833, 1, 0.667, 0.667, 0.75]
summer = [4, 3.83, 3.83, 3.33, 2.58, 2.75, 2.54, 1.83, 1.63]
density = [6] * 3 + [12] * 3 + [24] * 3

data = {
    "DENSITY": density * 2,
    "SEASON": ["SPRING"] * len(spring) + ["SUMMER"] * len(summer),
    "EGGS": spring + summer
}
df = pd.DataFrame(data)
df["DENSITY"] = df["DENSITY"].astype(object)

# Look at dispesion of eggs of each factor
df.boxplot(column="EGGS", by="DENSITY")
df.boxplot(column="EGGS", by="SEASON")

# And together
df.boxplot(column="EGGS", by=["DENSITY", "SEASON"])

# Perform two way ANOVA
print("Performing two way ANOVA")
mod = ols('EGGS ~ DENSITY + SEASON + DENSITY:SEASON', data = df).fit()
print(sm.stats.anova.anova_lm(mod))
print("Both the density and season affect the eggs and there IS an interaction between the two factors.")

# Create interaction plot
print("Creating interaction plot")
interaction_plot(df['DENSITY'], df['SEASON'], df['EGGS'])
plt.show()

print("More eggs are laid during spring")
print("Lines are not parallel so an interaction occurs.")

