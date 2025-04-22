import numpy as np
import pandas as pd
df=pd.read_csv('golf-dataset.csv')
print(df)
def divide_data(data, feature):
    DATA={}
    feat_values=list(data[feature].value_counts().index)
    occurence=list(data[feature].value_counts())
    for val in feat_values:
        DATA[val]={'data':pd.DataFrame([], columns=data.columns),'len':0}
    for ix in range(data.shape[0]):
        val=data[feature].iloc[ix]
        DATA[val]['data']=DATA[val]['data'].append(data.iloc[ix])
        idx=feat_valuesindex(val)
        DATA[val]['len']=occurence[idx]
    return DATA

divide_data(df,'Outlook')
