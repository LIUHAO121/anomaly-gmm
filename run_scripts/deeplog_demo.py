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




def plot_one_column(df,col_name,save_path):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    plt.plot(df[col_name], label=col_name)
    plt.legend()
    plt.savefig(save_path)
 
def plot_one_column_dense(df,col_name,save_path):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    sns.kdeplot(df[col_name],label=col_name)
    plt.legend()
    plt.savefig(save_path)

def plot_multi_columns(df,col_names,save_path):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    for col_name in col_names:
        plt.plot(df[col_name], label=col_name)
    plt.legend(col_names,title='multi columns')
    plt.savefig(save_path)
        
     
result = pd.read_csv('datasets/anomaly/raw_data/block_chain.csv')   
plot_root = "run_scripts/out/imgs"
plot_one_column(result, 'Blocks', os.path.join(plot_root,"Blocks.png"))
plot_one_column_dense(result, 'Blocks', os.path.join(plot_root,"Blocks_dense.png"))


plot_one_column(result, 'Output_Satoshis', os.path.join(plot_root,"Output_Satoshis.png"))
plot_one_column(result, 'Transactions', os.path.join(plot_root,"Transactions.png"))


data = result[['Output_Satoshis','Blocks','Transactions']]
outliers_fraction=0.05
scaler = StandardScaler()
np_scaled = scaler.fit_transform(data)
data = pd.DataFrame(np_scaled).to_numpy()

transformer_DL = DeepLogSKI(window_size=10, stacked_layers=1, contamination=0.1, epochs=1)
transformer_DL.fit(data)
prediction_labels_DL = transformer_DL.predict(data)
prediction_score_DL = transformer_DL.predict_score(data)


result['anomaly_DeepLog'] = pd.Series(prediction_labels_DL.flatten())
result['anomaly_DeepLog'] = result['anomaly_DeepLog'].apply(lambda x: x == 1)
result['anomaly_DeepLog'] = result['anomaly_DeepLog'].astype(int)
result['anomaly_DeepLog'].value_counts()

fig, ax = plt.subplots(figsize=(10,6))

#anomaly
a = result.loc[result['anomaly_DeepLog'] == 1]
outlier_index=list(a.index)
ax.plot(result['Transactions'], color='black', label = 'Normal', linewidth=1.5)
ax.scatter(a.index ,a['Transactions'], color='red', label = 'Anomaly', s=16)
ax.plot(pd.Series(prediction_score_DL.flatten()*10), color='blue', label = 'Score', linewidth=0.5)

plt.legend()
plt.title("Anamoly Detection Using DeepLog")
plt.xlabel('Date')
plt.ylabel('Transactions')
plt.savefig(os.path.join(plot_root,"predict.png"))
