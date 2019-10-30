import pandas as pd # Read data and work with data frames
import matplotlib.pyplot as plt # Boxplot and histogram
import statsmodels.api as sm # QQ plot
import scipy.stats as stats # Basic statistics
import numpy as np # Scientific computing
from pathlib import Path

data = pd.read_table(Path.cwd() / "Practical1/spider_web.txt", sep=' ')

def ttest(keyword, human_readable):
    result = stats.ttest_ind(data[keyword + 'DIM'], data[keyword + 'LIG'], equal_var = True)
    print(f"The test statistic for {human_readable} is {result.statistic} with a pvalue of {result.pvalue}.")
 
ttest("HORIZ", "web width")
ttest("VERT", "web height")