from scipy.stats.mstats import zscore
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tods.sk_interface.detection_algorithm.Telemanom_skinterface import TelemanomSKI
from tods.sk_interface.detection_algorithm.LSTMODetector_skinterface import LSTMODetectorSKI
from tods.sk_interface.detection_algorithm.DeepLog_skinterface import DeepLogSKI
from tods.sk_interface.detection_algorithm.AutoEncoder_skinterface import AutoEncoderSKI
from tods.sk_interface.detection_algorithm.LSTMAE_skinterface import LSTMAESKI
from tods.sk_interface.detection_algorithm.VariationalAutoEncoder_skinterface import VariationalAutoEncoderSKI
from tods.sk_interface.detection_algorithm.DAGMM_skinterface import DAGMMSKI
from tods.sk_interface.detection_algorithm.LSTMVAE_skinterface import LSTMVAESKI
from tods.sk_interface.detection_algorithm.LSTMVAEGMM_skinterface import LSTMVAEGMMSKI
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

from run_scripts.plot_tools import plot_anomal_multi_columns,plot_anomal_multi_columns_3d,plot_multi_columns,plot_one_column_with_label,plot_predict, plot_after_train, plot_before_train
from run_scripts.metric_tools import calc_point2point, adjust_predicts,multi_threshold_eval
from run_scripts.utils import train_step,eval_step


import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
    
    

    
machine_names = ["machine-1-1", "machine-1-2","machine-1-3", "machine-1-4",  "machine-1-5","machine-1-6",  "machine-1-7", "machine-1-8", 
                 "machine-2-1",  "machine-2-2", "machine-2-3",  "machine-2-4","machine-2-5", "machine-2-6", "machine-2-7",    "machine-2-8",  "machine-2-9", 
                "machine-3-1",  "machine-3-2",  "machine-3-3",  "machine-3-4", "machine-3-5",  "machine-3-6","machine-3-7",  "machine-3-8",   "machine-3-9", "machine-3-10","machine-3-11", ]


# machine_names = ['machine-1-4']

train_suffix = "_train.pkl"
test_suffix = "_test.pkl"
test_label_suffix = "_test_label.pkl"


dataset_name = "SMD"
dataset_dim = 38

deeplog_args = {
    "window_size":100, 
    "stacked_layers":1,
    "contamination":0.1, 
    "contaminations":[0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2],
    "epochs":6,
    "dataset_dir":'datasets/SMD',
    "dataset_name":"SMD",
    "dataset_dim":38,
    "batch_size":50,
    "anomal_col":"anomaly",
    "hidden_size":64,
    "plot":False,
    "plot_dir": "run_scripts/out/imgs",
    "metrics_dir": "run_scripts/out/metric",
    "model_dir":"run_scripts/out/models",
    "important_cols":['1','9','10','12','13','14','15','23'],
    "plot_cols":['9','10','12'],
    "use_important_cols":False,
    "model":"deeplog",
    "sub_dataset":"null"
}

lstmod_args = {
    "stacked_layers":1,
    "contamination":0.1, 
    "contaminations":[0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2],
    "epochs":6,
    "min_attack_time":5,
    "n_hidden_layer":2,
    "hidden_dim":8,
    "dataset_dir":'datasets/SMD',
    "dataset_name":"SMD",
    "dataset_dim": 38,
    "batch_size": 50,
    "hidden_size": 8,
    "anomal_col":"anomaly",
    "model": "lstmod",
    "plot": False,
    "plot_dir": "run_scripts/out/imgs",
    "metrics_dir": "run_scripts/out/metric",
    "important_cols":['1','9','10','12','13','14','15','23'],
    "plot_cols":['9', '10', '12'],
    "use_important_cols": False,
    "sub_dataset": "null"
}

telemanom_args = {
    "stacked_layers":1,
    "contamination":0.1, 
    "contaminations":[0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4],
    "epochs":3,
    "min_attack_time":5,
    "n_hidden_layer":2,
    "hidden_dim":8,
    "dataset_dir":'datasets/SMD',
    "dataset_name":"SMD",
    "dataset_dim": 38,
    "l_s":100,
    "layers":[64,64],   # No of units for the 2 lstm layers
    "n_predictions":2,
    "window_size_":1, # !!!!!!
    "anomal_col":"anomaly",
    "model": "telemanom",
    "plot": False,
    "plot_dir": "run_scripts/out/imgs",
    "metrics_dir": "run_scripts/out/metric",
    "important_cols":['1','9','10','12','13','14','15','23'],
    "plot_cols":['9', '10', '12'],
    "use_important_cols": False,
    "sub_dataset": "null"
}

ae_args = {
    "preprocessing":False,
    "batch_size":32,
    "epochs":20,
    "hidden_neurons":[20,10,20],
    "contamination":0.1, 
    "contaminations":[0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4],
    "epochs":3,
    "dataset_dir":'datasets/SMD',
    "dataset_name":"SMD",
    "dataset_dim": 38,
    "anomal_col":"anomaly",
    "model": "AE",
    "plot": False,
    "plot_dir": "run_scripts/out/imgs",
    "metrics_dir": "run_scripts/out/metric",
    "important_cols":['1','9','10','12','13','14','15','23'],
    "plot_cols":['9', '10', '12'],
    "use_important_cols": False,
    "sub_dataset": "null"
}



lstmae_args = {
    "preprocessing":False,
    "hidden_neurons":[16,3,16],
    "window_size":100, 
    "stacked_layers":1,
    "contamination":0.1, 
    "contaminations":[0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2],
    "contamination":0.01,
    "epochs":6,
    "dataset_dir":f'datasets/{dataset_name}',
    "dataset_name":dataset_name,
    "dataset_dim":dataset_dim,
    "batch_size":50,
    "anomal_col":"anomaly",
    "hidden_size":32,
    "plot":False,
    "plot_dir": "run_scripts/out/imgs",
    "metrics_dir": "run_scripts/out/metric",
    "important_cols":['1','9','10','12','13','14','15','23'],
    "plot_cols":['9','10','12'],
    "use_important_cols":False,
    "model":"LSTMAE",
    "sub_dataset":"null"
}

vae_args = {
    "model":"VAE",
    "preprocessing":False,
    "encoder_neurons":[128, 64, 32],
    "decoder_neurons":[32, 64, 128],
    "latent_dim":3,
    "contaminations":[0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2],
    "contamination":0.01,
    "epochs": 6,
    "dataset_dir":f'datasets/{dataset_name}',
    "dataset_name":dataset_name,
    "dataset_dim":dataset_dim,
    "batch_size":50,
    "anomal_col":"anomaly",
    "hidden_size":32,
    "plot":False,
    "plot_dir": "run_scripts/out/imgs",
    "metrics_dir": "run_scripts/out/metric",
    "important_cols":['1','9','10','12','13','14','15','23'],
    "plot_cols":['9','10','12'],
    "use_important_cols":False,
    "sub_dataset":"null"
}

dagmm_args = {
    "model":"DAGMM",
    "normalize":False,
    "comp_hiddens":[16,8,1],
    "est_hiddens":[8,4],
    "minibatch_size":1024,
    "epoch_size":100,
    "contaminations":[0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2],
    "contamination":0.01,
    "epochs": 6,
    "dataset_dir":f'datasets/{dataset_name}',
    "dataset_name":dataset_name,
    "dataset_dim":dataset_dim,
    "anomal_col":"anomaly",
    "plot":False,
    "plot_dir": "run_scripts/out/imgs",
    "metrics_dir": "run_scripts/out/metric",
    "important_cols":['1','9','10','12','13','14','15','23'],
    "plot_cols":['9','10','12'],
    "use_important_cols":False,
    "sub_dataset":"null"
}


lstmvae_args = {
    "model":"LSTMVAE",
    "preprocessing":False,
    "window_size":100, 
    "batch_size":32,
    "hidden_size":64,
    "encoder_neurons":[64,32,16],
    "decoder_neurons":[16,32,64],
    "latent_dim":2,
    "epoch_size":32,
    "contaminations":[0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2],
    "contamination":0.01,
    "epochs": 6,
    "dataset_dir":f'datasets/{dataset_name}',
    "dataset_name":dataset_name,
    "dataset_dim":dataset_dim,
    "anomal_col":"anomaly",
    "plot":False,
    "plot_dir": "run_scripts/out/imgs",
    "metrics_dir": "run_scripts/out/metric",
    "important_cols":['1','9','10','12','13','14','15','23'],
    "plot_cols":['9','10','12'],
    "use_important_cols":False,
    "sub_dataset":"null"
}

lstmvaegmm_args = {
    "model":"LSTMVAEGMM",
    "num_gmm":4,
    "preprocessing":False,
    "window_size":100, 
    "batch_size":64,
    "hidden_size":64,
    "encoder_neurons":[64,32,16],
    "decoder_neurons":[16,32,64],
    "latent_dim":2,
    "contaminations":[0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2],
    "contamination":0.01,
    "epochs": 2,
    "dataset_dir":f'datasets/{dataset_name}',
    "dataset_name":dataset_name,
    "dataset_dim":dataset_dim,
    "anomal_col":"anomaly",
    "plot":False,
    "plot_dir": "run_scripts/out/imgs",
    "metrics_dir": "run_scripts/out/metric",
    "model_dir": "run_scripts/out/models",
    "important_cols":['1','9','10','12','13','14','15','23'],
    "plot_cols":['9','10','12'],
    "use_important_cols":False,
    "sub_dataset":"null"
}

args = deeplog_args


def prepare_data(args,machine_name):
    # path
    dataset_dir = args['dataset_dir']
    train_data_path = os.path.join(dataset_dir,machine_name + train_suffix) 
    test_data_path = os.path.join(dataset_dir,machine_name + test_suffix) 
    test_label_path = os.path.join(dataset_dir,machine_name + test_label_suffix) 
    # read
    train_data = pickle.load(open(train_data_path,'rb'))
    test_data = pickle.load(open(test_data_path,'rb'))
    test_label = pickle.load(open(test_label_path,'rb'))
    
    
    # convert to dataframe
    columns = [str(i+1) for i in range(args['dataset_dim'])]
    train_df = pd.DataFrame(train_data,columns=columns)
    test_df = pd.DataFrame(test_data,columns=columns) 
    if args['use_important_cols']:
        train_df = train_df.loc[:,args['important_cols']]
        test_df = test_df.loc[:,args['important_cols']]
    
    # normalize
    scaler = StandardScaler()
    train_np = scaler.fit_transform(train_df)
    test_np = scaler.transform(test_df)
    
    # test_with_label
    columns = [str(i+1) for i in range(args['dataset_dim'])]
    test_with_label = np.concatenate([test_np, test_label.reshape(-1,1)], axis=1)
    test_with_label_df = pd.DataFrame(test_with_label)
    columns.append(args['anomal_col']) # inplace
    test_with_label_df.columns = columns
    test_with_label_df[args['anomal_col']] = test_with_label_df[args['anomal_col']].astype(int)
    
    # plot 
    if args["plot"]:
        plot_before_train(args, df=test_with_label_df)
    
    return train_np , test_np, test_with_label_df 


def train(args,machine_name):
    args['sub_dataset'] = machine_name
    train_np, test_np, test_with_label_df = prepare_data(args, machine_name)  # 已归一化
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
    
    # transformer_DL = LSTMODetectorSKI(
    #     min_attack_time = args['min_attack_time'],
    #     epochs = args['epochs'],
    #     batch_size = args['batch_size'],
    #     hidden_dim = args['hidden_dim'],
    #     n_hidden_layer = args['n_hidden_layer']
    # )
    
    # transformer_DL = TelemanomSKI(
    #     epochs = args['epochs'],
    #     l_s = args['l_s'],
    #     n_predictions = args['n_predictions'],
    #     layers = args['layers'],
    #     window_size_ = args['window_size_']
    # )
    
    # transformer_DL = AutoEncoderSKI(
    #     preprocessing = args["preprocessing"],
    #     batch_size = args["batch_size"],
    #     epochs = args["epochs"],
    #     hidden_neurons = args["hidden_neurons"],
    # )
    
    # transformer_DL = LSTMAESKI(
    #     window_size = args['window_size'],
    #     preprocessing = args["preprocessing"],
    #     batch_size = args["batch_size"],
    #     epochs = args["epochs"],
    #     hidden_neurons = args["hidden_neurons"],
    #     hidden_size=args['hidden_size']
    # )
    
    # transformer_DL = VariationalAutoEncoderSKI(
    #     preprocessing = args["preprocessing"],
    #     batch_size = args["batch_size"],
    #     epochs = args["epochs"],
    #     encoder_neurons = args["encoder_neurons"],
    #     decoder_neurons = args["decoder_neurons"],
    #     latent_dim = args["latent_dim"]
    # )
    
    # transformer_DL = DAGMMSKI(
    #     normalize = args["normalize"],
    #     comp_hiddens = args["comp_hiddens"],
    #     est_hiddens = args["est_hiddens"],
    #     minibatch_size = args["minibatch_size"],
    #     epoch_size = args["epoch_size"],
    # )
    
    # transformer_DL = LSTMVAESKI(
    #     window_size=args['window_size'],
    #     hidden_size = args['hidden_size'],
    #     preprocessing = args["preprocessing"],
    #     batch_size = args["batch_size"],
    #     epochs = args["epochs"],
    #     latent_dim = args["latent_dim"],
    #     encoder_neurons = args["encoder_neurons"],
    #     decoder_neurons = args["decoder_neurons"],
    # )
    
    
    # transformer_DL = LSTMVAEGMMSKI(
    #     num_gmm = args["num_gmm"],
    #     window_size=args['window_size'],
    #     hidden_size = args['hidden_size'],
    #     preprocessing = args["preprocessing"],
    #     batch_size = args["batch_size"],
    #     epochs = args["epochs"],
    #     latent_dim = args["latent_dim"],
    #     encoder_neurons = args["encoder_neurons"],
    #     decoder_neurons = args["decoder_neurons"]
    # )
    
    model_dir =  os.path.join(args['model_dir'],"{}_{}_{}".format(args['dataset_name'],args['model'],args['sub_dataset']))
    if not os.path.exists(model_dir):
        train_step(
            args,
            transformer_DL=transformer_DL,
            train_np=train_np,
            test_np=test_np,
            test_with_label_df=test_with_label_df
                )
    else:
        eval_step(
            args,
            transformer_DL=transformer_DL,
            test_np=test_np,
            test_with_label_df=test_with_label_df
        )
    

for n in machine_names:
    train(args=args,machine_name=n)