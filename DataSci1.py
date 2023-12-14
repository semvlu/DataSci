import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from pandas.core.frame import DataFrame

df = pd.read_csv("train.csv")
df['Attribute20'] = df['Attribute20'].replace('Yes', 1)
df['Attribute20'] = df['Attribute20'].replace('No', 0)
df['Attribute21'] = df['Attribute21'].replace('Yes', 1)
df['Attribute21'] = df['Attribute21'].replace('No', 0)
df['Attribute1'] = pd.to_datetime(df['Attribute1']).dt.month
df = pd.get_dummies(df, columns=['Attribute8', 'Attribute9'])
df = df.dropna()
train = df.drop('Attribute21', axis=1)
label = df['Attribute21']

sm = SMOTE(sampling_strategy='minority', random_state=38)
train, label = sm.fit_resample(train, label)

reg = GaussianNB(var_smoothing=7.5e-4)
reg.fit(train, label)

test = pd.read_csv("test.csv")
test['Attribute20'] = test['Attribute20'].replace('Yes', 1)
test['Attribute20'] = test['Attribute20'].replace('No', 0)
test['Attribute1'] = pd.to_datetime(test['Attribute1']).dt.month
test = pd.get_dummies(test, columns=['Attribute8', 'Attribute9'])
test = test.dropna()
pred = reg.predict(test)
header = [['id', 'ans']]

for i in range(len(pred)):
    header.append([float(i), str(int(pred[i]))])

res = DataFrame(header)
res.to_csv("b10933001.csv", header=None, index=False, encoding='utf-8')
