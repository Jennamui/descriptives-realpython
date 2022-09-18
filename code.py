#import packages
import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt

##Use Python’s statistics for the most important Python statistics functions.
##Use NumPy to handle arrays efficiently.
##Use SciPy for additional Python statistics routines for NumPy arrays.
##Use Pandas to work with labeled datasets.
##Use Matplotlib to visualize data with plots, charts, and histograms.

#create data
x = [8, 1, 2.5, 4, 28]
x_with_nan = [8, 1, 2.5, 4, 28, math.nan]
x
x_with_nan

#Create np.ndarray and pd.series
y, y_with_nan = np.array(x), np.array(x_with_nan)
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)

#Caluate mean
mean_ = sum(x) / len(x)
mean_ = statistics.mean(x)
mean_ = statistics.mean(x_with_nan)
mean_ = np.mean(y)
mean_ = y.mean()
np.mean(y_with_nan)
np.nanmean(y_with_nan)
mean_ = z.mean()
z_with_nan.mean()

#weighted mean
x = [8, 1, 2.5, 4, 28]
w = [0.1, 0.2, 0.3, 0.25, 0.15]
wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)

y, z, w = np.array(x), pd.Series(x), np.array(w)
wmean = np.average(y, weights=w)

w = np.array([0.1, 0.2, 0.3, 0.0, 0.2, 0.1])
(w * y_with_nan).sum() / w.sum()

#harmonic mean
hmean = len(x) / sum(1 / item for item in x)
hmean = statistics.harmonic_mean(x)
statistics.harmonic_mean(x_with_nan)
scipy.stats.hmean(y)
scipy.stats.hmean(z)

#geometric mean
gmean = statistics.geometric_mean(x)
gmean = statistics.geometric_mean(x_with_nan)
scipy.stats.gmean(y)
scipy.stats.gmean(z)

#Median
n = len(x)
if n % 2:
    median = sorted(x)[round(0.5*(n-1))]
else:
    x_ord, index = sorted(x), round(0.5 * n)
    median = 0.5 * (x_ord[index-1] + x_ord[index])

median_ = statistics.median(x)
median_ = statistics.median(x[:-1])
statistics.median_low(x[:-1])
statistics.median_high(x[:-1])

statistics.median(x_with_nan)

median_ = np.median(y)
median_ = np.median(y[:-1])

np.nanmedian(y_with_nan)
np.nanmedian(y_with_nan[:-1])

z.median()
z_with_nan.median()

#Mode
u = [2, 3, 2, 8, 12]
mode_ = max((u.count(item), item) for item in set(u))[1]

mode_ = statistics.mode(u)
mode_ = statistics.multimode(u)

v = [12, 15, 12, 15, 21, 15, 12]
statistics.mode(v)
statistics.multimode(v)

statistics.mode([2, math.nan, 2])
statistics.multimode([2, math.nan, 2])

u, v = np.array(u), np.array(v)
mode_ = scipy.stats.mode(u)

#sample variance
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
var_

var_ = statistics.variance(x)
var_

statistics.variance(x_with_nan)

var = np.var(y, ddof=1)
var = y.var(ddof=1)

np.nanvar(y_with_nan, ddof=1)

#Standard Deviation
std_ = var_ ** 0.5
std_ = statistics.stdev(x)
np.std(y, ddof=1)
y.std(ddof=1)
np.nanstd(y_with_nan, ddof=1)
z.std(ddof=1)
z_with_nan.std(ddof=1)

#sample skewness
x = [8.0, 1, 2.5, 4, 28.0]
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
std_ = var_ ** 0.5
skew_ = (sum((item - mean_)**3 for item in x)
 * n / ((n - 1) * (n - 2) * std_**3))

y, y_with_nan = np.array(x), np.array(x_with_nan)
scipy.stats.skew(y, bias=False)

z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
z_with_nan.skew()

#percentile
y = np.array(x)
np.percentile(y, 5)
np.percentile(y, 95)

y_with_nan = np.insert(y, 2, np.nan)
np.nanpercentile(y_with_nan, [25, 50, 75])

#range
np.ptp(y)
np.ptp(z)
np.ptp(y_with_nan)
np.ptp(z_with_nan)

#quartiles
quartiles = np.quantile(y, [0.25, 0.75])
quartiles[1] - quartiles[0]

quartiles = z.quantile([0.25, 0.75])
quartiles[0.75] - quartiles[0.25]

#summary
result = scipy.stats.describe(y, ddof=1, bias=False)
result.nobs
result.minmax[0]
result.minmax[1]
result.mean
result.variance
result.skewness
result.kurtosis

result = z.describe()

#Covariance
x = list(range(-10, 11))
y = [0, 2, 2, 2, 2, 3, 3, 6, 7, 4, 7, 6, 6, 9, 4, 5, 5, 10, 11, 12, 14]
x_, y_ = np.array(x), np.array(y)
x__, y__ = pd.Series(x_), pd.Series(y_)

cov_matrix = np.cov(x_, y_)
cov_xy = cov_matrix[0, 1]

#correlation coefficient
scipy.stats.linregress(x_, y_)

result = scipy.stats.linregress(x_, y_)
r = result.rvalue
r = x__.corr(y__)
r = x__.corr(x__)

##2D Data
#Axes
a = np.array([[1, 1, 1],
               [2, 3, 1],
               [4, 9, 2],
               [8, 27, 4],
               [16, 1, 1]])
np.mean(a)
a.mean()
np.median(a)
a.var(ddof=1)
np.mean(a, axis=0)
np.mean(a, axis=1)
scipy.stats.gmean(a)
scipy.stats.gmean(a, axis=1)
scipy.stats.gmean(a, axis=None)
scipy.stats.describe(a, axis=None, ddof=1, bias=False)

#Dataframes
row_names = ['first', 'second', 'third', 'fourth', 'fifth']
col_names = ['A', 'B', 'C']
df = pd.DataFrame(a, index=row_names, columns=col_names)

df.mean()
df.var()
df.mean(axis=1)
df.var(axis=1)
df['A']
df['A'].mean()
df.values
df.describe()

##Visualizing Data
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#Box plots
np.random.seed(seed=0)
x = np.random.randn(1000)
y = np.random.randn(100)
z = np.random.randn(10)

fig, ax = plt.subplots()
ax.boxplot((x, y, z), vert=False, showmeans=True, meanline=True,
           labels=('x', 'y', 'z'), patch_artist=True,
           medianprops={'linewidth': 2, 'color': 'purple'},
           meanprops={'linewidth': 2, 'color': 'red'})
plt.show()

#Histograms
hist, bin_edges = np.histogram(x, bins=10)

fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=False)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=True)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

#Pie charts
x, y, z = 128, 256, 1024
fig, ax = plt.subplots()
ax.pie((x, y, z), labels=('x', 'y', 'z'), autopct='%1.1f%%')
plt.show()

#Bar charts
x = np.arange(21)
y = np.random.randint(21, size=21)
err = np.random.randn(21)

fig, ax = plt.subplots()
ax.bar(x, y, yerr=err)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

#X-Y Plots
x = np.arange(21)
y = 5 + 2 * x + 2 * np.random.randn(21)
slope, intercept, r, *__ = scipy.stats.linregress(x, y)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'

fig, ax = plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Data points')
ax.plot(x, intercept + slope * x, label=line)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor='white')
plt.show()

#heatmaps
matrix = np.cov(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()