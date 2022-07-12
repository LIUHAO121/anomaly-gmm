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
import pickle

from run_scripts.plot_tools import plot_anomal_multi_columns,plot_anomal_multi_columns_3d,plot_multi_columns,plot_one_column_with_label,plot_predict

machine_names = ["machine-1-1",  "machine-1-4",  "machine-1-7",  "machine-2-2",  "machine-2-5",  "machine-2-8",  "machine-3-10",  "machine-3-3",  "machine-3-6",  "machine-3-9",
                "machine-1-2",  "machine-1-5", "machine-1-8",  "machine-2-3",  "machine-2-6",  "machine-2-9", "machine-3-11",  "machine-3-4",  "machine-3-7",
                "machine-1-3",  "machine-1-6",  "machine-2-1",  "machine-2-4",  "machine-2-7",  "machine-3-1",  "machine-3-2",   "machine-3-5",  "machine-3-8"]

train_suffix = "_train.pkl"
test_suffix = "_test.pkl"
test_label_suffix = "_test_label.pkl"


train_args = {
    "window_size":100, 
    "stacked_layers":1,
    "contamination":0.15, 
    "epochs":6,
    "dataset_dir":'datasets/SMD',
    "dataset_name":"SMD",
    "dataset_dim":38,
    "batch_size":50,
    "anomal_col":"anomaly",
    "hidden_size":64,
    "plot":True,
    "plot_dir":"run_scripts/out/imgs",
    "imptant_cols":['1','12','15']

}


def plot_before_train(args, df):
    for col in df.columns[:-1]:
        plot_one_column_with_label(
            df=df,
            col_name=col,
            anomal_col=args['anomal_col'],
            save_path=os.path.join(args['plot_dir'],"{}_{}.png".format(args['dataset_name'],col)))
        


def prepare_data(args,machine_name):
    # path
    dataset_dir = args['dataset_dir']
    train_data_path = os.path.join(dataset_dir,machine_name + train_suffix) 
    test_data_path = os.path.join(dataset_dir,machine_name + test_suffix) 
    test_label_path = os.path.join(dataset_dir,machine_name + test_label_suffix) 
    # read
    train_data=pickle.load(open(train_data_path,'rb'))
    test_data=pickle.load(open(test_data_path,'rb'))
    test_label=pickle.load(open(test_label_path,'rb'))
    
    
    
    # convert to dataframe
    columns = [str(i+1) for i in range(args['dataset_dim'])]
    train_df = pd.DataFrame(train_data,columns=columns)
    test_df = pd.DataFrame(test_data,columns=columns) 
    test_label_df = pd.DataFrame(test_label,columns=[args['anomal_col']])
    test_label_df[args['anomal_col']] = test_label_df[args['anomal_col']].astype(int)
    
    columns = [str(i+1) for i in range(args['dataset_dim'])]
    test_with_label = np.concatenate([test_data,test_label.reshape(-1,1)],axis=1)
    test_with_label_df = pd.DataFrame(test_with_label)
    columns.append(args['anomal_col'])
    test_with_label_df.columns = columns
    test_with_label_df[args['anomal_col']] = test_with_label_df[args['anomal_col']].astype(int)
    
    # plot 
    if args["plot"]:
        plot_before_train(args, df=test_with_label_df)
    
    # normalize
    scaler = StandardScaler()
    train_np = scaler.fit_transform(train_df)
    test_np = scaler.transform(test_df)
    
    return train_np , test_np, test_label_df


def train(args,machine_name):
    train_np, test_np, test_label_df = prepare_data(args, machine_name)  # 已归一化
    test_anomal_num = int(np.sum(test_label_df[args['anomal_col']]))
    test_data_num = int(test_np.shape[0])
    
    transformer_DL = DeepLogSKI(
                                window_size=args['window_size'],
                                stacked_layers=args['stacked_layers'],
                                contamination=args['contamination'],
                                epochs=args['epochs'],
                                batch_size = args['batch_size'],
                                hidden_size=args['hidden_size']
                                )
    transformer_DL.fit(train_np)
    prediction_labels_DL = transformer_DL.predict(test_np)
    prediction_score_DL = transformer_DL.predict_score(test_np)
    
    y_true = test_label_df[args['anomal_col']]
    y_pred = pd.Series(prediction_labels_DL.flatten())
    
    print("processing machine {}".format(machine_name))
    print("test : anomal/total {}/{}".format(test_anomal_num, test_data_num))
    print('Accuracy Score: ', accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    

# for n in machine_names[0]:
train(args=train_args,machine_name=machine_names[0])