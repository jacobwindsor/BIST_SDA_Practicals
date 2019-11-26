import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns # statistical data visualization
from scipy.stats import pearsonr
from pathlib import Path

droso = pd.read_table(Path.cwd() / "Practical3/droso_survival.txt", delim_whitespace = True)
droso_log = np.log(droso[['size', 'egg_rate', 'longv']])

def calc_residuals(var1, var2):
    print(f"Calculating residuals between {var1} and {var2}")
    mod = ols(f"{var1}~{var2}", data=droso_log).fit()
    return mod.resid

def pairwise(var1, var2):
    corr = pearsonr(droso_log[var1], droso_log[var2])
    print(f"Pairwise correlation between {var1} and {var2}: {corr}")
    return corr

# Compute pairwise correlations
longv_size = pairwise("longv", "size")
longv_eggrate = pairwise("longv", "egg_rate")
size_eggrate = pairwise("size", "egg_rate")

# Caluclate residuals of regrssion between survival and body size
resid1 = calc_residuals("longv", "size")
resid2 = calc_residuals("egg_rate", "size")

correlation = pearsonr(resid1, resid2)
print(f"Correlation between residuals: {correlation}")
