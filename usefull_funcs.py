"""
This file is set od random usefull functions i use thruout most of my projects
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import recall_score, precision_score, \
        confusion_matrix, accuracy_score, roc_auc_score
from lazypredict.Supervised import LazyClassifier



a4_dims = (11.7, 8.27)


def initial_sss(df, label, test_size, out_file):

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=42)
    for train_index, test_index in sss.split(df, df[label]):
        print("TRAIN:", train_index, "TEST:", test_index)
        train_df = df.loc[train_index]
        test_df = df.loc[test_index]

    print(f'Train shape: {train_df.shape}')
    print(f'Train shape: {test_df.shape}')

    print(f'Train value counts: {train_df.Class.value_counts()}')
    print(f'Test value counts: {test_df.Class.value_counts()}')

    train_df.to_csv(f'{out_file}/train_df', index=False)
    test_df.drop(label, 1).to_csv(f'{out_file}/test_df', index=False)
    test_df[label].to_csv(f'{out_file}/test_df_y_true', index=False)


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


def iqr_removal(df, threshold, col_list):

    print(f'Before removal: \n {df["Class"].value_counts()}')
    def outlier_treatment(datacolumn):
        sorted(datacolumn)
        q1, q3 = np.percentile(datacolumn , [25,75])
        iqr = q3 - q1
        lower_range = q1 - (threshold * iqr)
        upper_range = q3 + (threshold * iqr)

        return lower_range,upper_range

    for col in col_list:
        lower_range, upper_range = outlier_treatment(df[col])
        outliers = df.loc[(df[col] > upper_range) | (df[col] < lower_range)]
        outliers_indexes = outliers.index
        df = df.drop(outliers_indexes)

    print(f'After removal: \n {df["Class"].value_counts()}')

    return df


def print_metrcis(y_test, y_pred):
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'ROC AUC: {roc_auc_score(y_test, y_pred)}')
    print(f'Precision: {precision_score(y_test, y_pred)}')
    print(f'Recall: {recall_score(y_test, y_pred)}')
    print(f'Confusion Matrix: \n {confusion_matrix(y_test, y_pred)}')


def lazy_cls(X, y, output_csv=False):
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
    )

    clf = LazyClassifier()
    models,predictions = clf.fit(X_train, X_test, y_train, y_test)

    if output_csv:
        models.to_csv('data/lazy_cls.csv')

    print(models)


