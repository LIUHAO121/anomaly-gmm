import numpy as np
import pandas as pd
import pickle as pkl


# preprocess for SWaT. SWaT.A2_Dec2015, version 0
df = pd.read_excel('datasets/SWaT/SWaT_Dataset_Attack_v0.xlsx')
y = list(df['Normal/Attack'])
labels = []
for i in y:
    if i == 'Attack':
        labels.append(1)
    else:
        labels.append(0)
labels = np.array(labels)
assert len(labels) == 449919
pkl.dump(labels, open('datasets/SWaT/SWaT_test_label.pkl', 'wb'))
print('SWaT_test_label saved')

df = df.drop(columns=[' Timestamp', 'Normal/Attack'])
test = np.array(df)
assert test.shape == (449919, 51)
pkl.dump(test, open('datasets/SWaT/SWaT_test.pkl', 'wb'))
print('SWaT_test saved')

df = pd.read_excel('datasets/SWaT/SWaT_Dataset_Normal_v0.xlsx')
df = df.drop(columns=['Unnamed: 51'])
train = np.array(df[1:]) # 去掉第一行
assert train.shape == (496800, 51)
pkl.dump(train, open('datasets/SWaT/SWaT_train.pkl', 'wb'))
print('SWaT_train saved')