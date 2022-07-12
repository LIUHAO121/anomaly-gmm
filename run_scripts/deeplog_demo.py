from scipy.stats.mstats import zscore
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tods.sk_interface.detection_algorithm.Telemanom_skinterface import TelemanomSKI
from tods.sk_interface.detection_algorithm.DeepLog_skinterface import DeepLogSKI
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import datetime as dt
from datetime import datetime,tzinfo
import scipy, json, csv, time, pytz
from pytz import timezone
import numpy as np
import pandas as pd
import os

from run_scripts.plot_tools import plot_anomal_multi_columns,plot_anomal_multi_columns_3d,plot_multi_columns,plot_one_column,plot_one_column_dense,plot_predict



    
data_path = 'datasets/anomaly/raw_data/yahoo_sub_5.csv'
dataset_name = "yahoo_sub_5"
plot_root = "run_scripts/out/imgs"
train_cols = ['value_0','value_1','value_2','value_3','value_4']
anomal_col = 'anomaly'
plot_cols = ['value_0','value_1','value_4']


dataset = pd.read_csv(data_path)   
plot_anomal_multi_columns(dataset,train_cols,anomal_col,os.path.join(plot_root, f"{dataset_name}.png"))
plot_anomal_multi_columns_3d(dataset,plot_cols, anomal_col, os.path.join(plot_root, f"{dataset_name}_3d.png"))


data = dataset[train_cols]

scaler = StandardScaler()
np_scaled = scaler.fit_transform(data)
data = pd.DataFrame(np_scaled).to_numpy()


# train
transformer_DL = DeepLogSKI(window_size=10, stacked_layers=1, contamination=0.01, epochs=10)
transformer_DL.fit(data)
prediction_labels_DL = transformer_DL.predict(data)
prediction_score_DL = transformer_DL.predict_score(data)


# dataset['anomaly_DeepLog'] = pd.Series(prediction_labels_DL.flatten())
# dataset['anomaly_DeepLog'] = dataset['anomaly_DeepLog'].apply(lambda x: x == 1)
# dataset['anomaly_DeepLog'] = dataset['anomaly_DeepLog'].astype(int)
# dataset['anomaly_DeepLog'].value_counts()


# eval
y_true = dataset[anomal_col]
y_pred = pd.Series(prediction_labels_DL.flatten())
print('Accuracy Score: ', accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))