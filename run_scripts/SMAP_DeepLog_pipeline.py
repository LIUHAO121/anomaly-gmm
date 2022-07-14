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
from run_scripts.metric_tools import calc_point2point, adjust_predicts,multi_threshold_eval

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
    


train_args = {
    "window_size":100, 
    "stacked_layers":1,
    "contamination":0.1, 
    "contaminations":[0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2],
    "epochs":6,
    "dataset_dir":'datasets/SMAP',
    "dataset_name":"SMAP",
    "dataset_dim":25,
    "batch_size":50,
    "anomal_col":"anomaly",
    "hidden_size":64,
    "plot":True,
    "plot_dir": "run_scripts/out/imgs",
    "metrics_dir": "run_scripts/out/metric",
    "important_cols":['1','9','10','12','13','14','15','23'],
    "plot_cols":['9','10','12'],
    "use_important_cols":False,
    "model":"deeplog",
    "sub_dataset":"null"
}


def plot_before_train(args, df):
    # df = df.iloc[15000:17000,:]
    """
    df 必须包括标注列
    """
    for col in df.columns[:-1]:
        plot_one_column_with_label(
            df=df,
            col_name=col,
            anomal_col=args['anomal_col'],
            save_path=os.path.join(args['plot_dir'],"{}_{}.png".format(args['dataset_name'],col)))
        
    plot_anomal_multi_columns_3d(
                        df,
                        col_names=args['plot_cols'],
                        anomal_col=args['anomal_col'],
                        save_path=os.path.join(args['plot_dir'],'{}_multicols_3d.png'.format(args['dataset_name']))
                        )  

def plot_after_train(args,df,predict):
    
    """
    df 必须包括标注列
    """
    threshold =  np.percentile(predict, 100 * (1 - args['contamination']))
    max_score = np.max(predict)
    rescale_predict = predict / max_score
    rescale_threshod = threshold / max_score
    rescale_threshod_series = pd.Series([rescale_threshod for i in range(len(predict))])
    for col in df.columns[:-1]:
        plot_predict(df, 
                     col_name=col,
                     anomal_col=args['anomal_col'], 
                     predict=rescale_predict, 
                     threshold=rescale_threshod_series,
                     save_path=os.path.join(args['plot_dir'],'{}_{}_predict.png'.format(args['dataset_name'],col)))



def prepare_data(args):
    # path
    dataset_dir = args['dataset_dir']
    train_data_path = os.path.join(dataset_dir,"SMAP_train.pkl") 
    test_data_path = os.path.join(dataset_dir,"SMAP_test.pkl") 
    test_label_path = os.path.join(dataset_dir,"SMAP_test_label.pkl") 
    # read
    train_data=pickle.load(open(train_data_path,'rb'))
    test_data=pickle.load(open(test_data_path,'rb'))
    test_label=pickle.load(open(test_label_path,'rb'))
    
    
    
    # convert to dataframe
    columns = [str(i+1) for i in range(args['dataset_dim'])]
    train_df = pd.DataFrame(train_data,columns=columns)
    test_df = pd.DataFrame(test_data,columns=columns) 
    if args['use_important_cols']:
        train_df = train_df.loc[:,args['important_cols']]
        test_df = test_df.loc[:,args['important_cols']]
    
    # test_with_label
    columns = [str(i+1) for i in range(args['dataset_dim'])]
    test_with_label = np.concatenate([test_data, test_label.reshape(-1,1)], axis=1)
    test_with_label_df = pd.DataFrame(test_with_label)
    columns.append(args['anomal_col']) # inplace
    test_with_label_df.columns = columns
    test_with_label_df[args['anomal_col']] = test_with_label_df[args['anomal_col']].astype(int)
    
    # plot 
    if args["plot"]:
        plot_before_train(args, df=test_with_label_df)
        
    # normalize
    scaler = StandardScaler()
    train_np = scaler.fit_transform(train_df)
    test_np = scaler.transform(test_df)
    
    return train_np , test_np, test_with_label_df 


def train(args):

    train_np, test_np, test_with_label_df = prepare_data(args)  # 已归一化
    test_anomal_num = int(np.sum(test_with_label_df[args['anomal_col']]))
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
    prediction_labels_DL = transformer_DL.predict(test_np) # shape = (n,1)
    prediction_score_DL = transformer_DL.predict_score(test_np) # shape = (n,1)
    
    y_true = test_with_label_df[args['anomal_col']]
    y_pred = pd.Series(prediction_labels_DL.flatten())
    y_score = pd.Series(prediction_score_DL.flatten())
    

    # plot_after_train(
    #             args,
    #             df=test_with_label_df,
    #             predict=y_score
    #                  )
    multi_threshold_eval(args=args, pred_score=y_score, label=y_true)
    
    print("train_np.shape = ",train_np.shape)
    print("test : anomal/total {}/{}".format(test_anomal_num, test_data_num))
    

if __name__ == "__main__":
    train(args=train_args)