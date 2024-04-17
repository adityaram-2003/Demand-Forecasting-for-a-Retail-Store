import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

import lightgbm as lgb
import shap
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
# Import training and test data
train = pd.read_csv("C:/Users/Adityaram/OneDrive/Desktop/Demand Forecasting for a Retail Store/archive (4)/train.csv", parse_dates=['date'])
test = pd.read_csv("C:/Users/Adityaram/OneDrive/Desktop/Demand Forecasting for a Retail Store/archive (4)/train.csv", parse_dates=['date'])
df = pd.concat([train, test], sort=False)
print(train.shape, test.shape, df.shape, "\n")
train.head()
# How many stores and items are there?
train.store.nunique(), test.store.nunique(), train.item.nunique(), test.item.nunique()
# Time Range
train["date"].min(), train["date"].max(), test["date"].min(), test["date"].max()
# How many items are in the store?
df.groupby(["store"])["item"].nunique()
# Summary Stats for each store
df.groupby(["store"]).agg({"sales": ["count","sum", "mean", "median", "std", "min", "max"]})
# Summary Stats for each item
df.groupby(["item"]).agg({"sales": ["count","sum", "mean", "median", "std", "min", "max"]})
#histogram for store sales 

fig, axes = plt.subplots(2, 5, figsize=(20, 10))
for i in range(1,11):
    if i <= 5:
        train[train.store == i].sales.hist(ax=axes[0, i-1])
        axes[0,i-1].set_title("Store " + str(i), fontsize = 15)
        
    else:
        train[train.store == i].sales.hist(ax=axes[1, i - 6])
        axes[1,i-6].set_title("Store " + str(i), fontsize = 15)
plt.tight_layout(pad=4.5)
plt.suptitle("Histogram: Sales")
plt.show()

#distribution of sales of all items in store 1 
store = 1
sub = train[train.store == store].set_index("date")

fig, axes = plt.subplots(10, 5, figsize=(20, 35))
for i in range(1,51):
    row = (i - 1) // 5
    col = (i - 1) % 5
    sub[sub.item == i].sales.plot(ax=axes[row, col], legend=True, label = "Item "+str(i)+" Sales")
    axes[row, col].set_title("Item " + str(i) + " Sales")
    
plt.tight_layout(pad=4.5)
plt.suptitle("Store 1 Item Sales Diagram")
plt.show()

storesales = train.groupby(["date", "store"]).sales.sum().reset_index().set_index("date")
corr =  pd.pivot_table(storesales, values = "sales", columns="store", index="date").corr(method = "spearman")
plt.figure(figsize = (7,7))
sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.5)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 9}, square=True)
plt.title("Correlation Heatmap")
plt.show()
