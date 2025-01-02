import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from pandas.core.frame import DataFrame
import librosa

df = pd.read_csv("train.csv")
df['Feature 13'] = librosa.note_to_midi(df['Feature 13'])
df = df.dropna()

clu = DBSCAN(algorithm='kd_tree')
clu.fit(df)
c = clu.fit_predict(df)

test = pd.read_csv("test_4000.csv")
result = []
for i in range(test.shape[0]):
    id1 = test['col_1'][i]
    id2 = test['col_2'][i]
    pred1 = c[id1]
    pred2 = c[id2]
    if(pred1 == pred2):
        result.append(1)
    else:
        result.append(0)

header = [['id', 'ans']]
for i in range(len(result)):
    header.append([int(i), str(int(result[i]))])

res = DataFrame(header)
res.to_csv("out.csv", header=None, index=False, encoding='utf-8')
