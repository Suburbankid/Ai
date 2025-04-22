import numpy as np
import pandas as pd

df = pd.read_csv('golf-dataset.csv')

def divide_data(data, feature):
    DATA = {}
    feat_values = list(df[feature].value_counts().index)  # Unique feature values
    occurence = list(df[feature].value_counts())  # Counts of each feature value

    for val in feat_values:
        DATA[val] = {'data': pd.DataFrame([], columns=data.columns), 'len': 0}

    for ix in range(data.shape[0]):
        val = data[feature].iloc[ix]
        DATA[val]['data'] = pd.concat([DATA[val]['data'], data.iloc[[ix]]])  # Replace .append()
        idx = feat_values.index(val)  # Fix typo
        DATA[val]['len'] = occurence[idx]

    return DATA

divide_data(df, 'Outlook')
