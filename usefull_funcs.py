"""
This file is set od random usefull functions i use thruout most of my projects
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

a4_dims = (11.7, 8.27)


def hist_for_loop(df, columns=None):
    columns = df.columns
    for col in columns:
        fig, ax = plt.subplots(figsize=a4_dims)
        sns.histplot(df[col], kde=True)
        plt.title(col)
        plt.show()


def box_plot_for_loop(df, columns=None):
    columns = df.columns
    for col in columns:
        fig, ax = plt.subplots(figsize=a4_dims)
        sns.boxplot(df[col])
        plt.title(col)
        plt.show()


def outlier_treatment(datacolumn, iqr_multiplier):
    sorted(datacolumn)
    q1, q3 = np.percentile(datacolumn, [25, 75])
    iqr = q3 - q1
    lower_range = q1 - (iqr_multiplier * iqr)
    upper_range = q3 + (iqr_multiplier * iqr)
    return lower_range, upper_range


