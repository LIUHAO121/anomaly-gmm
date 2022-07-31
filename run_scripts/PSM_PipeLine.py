from scipy.stats.mstats import zscore
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tods.sk_interface.detection_algorithm.Telemanom_skinterface import TelemanomSKI
from tods.sk_interface.detection_algorithm.DeepLog_skinterface import DeepLogSKI
from tods.sk_interface.detection_algorithm.LSTMODetector_skinterface import LSTMODetectorSKI
from tods.sk_interface.detection_algorithm.AutoEncoder_skinterface import AutoEncoderSKI
from tods.sk_interface.detection_algorithm.LSTMAE_skinterface import LSTMAESKI
from tods.sk_interface.detection_algorithm.DAGMM_skinterface import DAGMMSKI
from tods.sk_interface.detection_algorithm.LSTMVAE_skinterface import LSTMVAESKI
from tods.sk_interface.detection_algorithm.OCSVM_skinterface import OCSVMSKI
from tods.sk_interface.detection_algorithm.VariationalAutoEncoder_skinterface import VariationalAutoEncoderSKI
from tods.sk_interface.detection_algorithm.LSTMVAEGMM_skinterface import LSTMVAEGMMSKI
from tods.sk_interface.detection_algorithm.LSTMVAEDISTGMM_skinterface import LSTMVAEDISTGMMSKI
from tods.sk_interface.detection_algorithm.GRUVAEGMM_skinterface import GRUVAEGMMSKI
from tods.sk_interface.detection_algorithm.LSTMAEGMM_skinterface import LSTMAEGMMSKI
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

from run_scripts.plot_tools import plot_anomal_multi_columns,plot_anomal_multi_columns_3d,plot_multi_columns,plot_one_column_with_label,plot_predict, plot_after_train,plot_before_train
from run_scripts.metric_tools import calc_point2point, adjust_predicts,multi_threshold_eval
from run_scripts.utils import train_step,eval_step

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
      
dataset_name = "PSM"
dataset_dim = 25


dagmm_args = {
    "model":"DAGMM",
    "normalize":False,
    "comp_hiddens":[16,8,1],
    "est_hiddens":[8,4],
    "minibatch_size":1024,
    "epoch_size":100,
    "contaminations":[0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2],
    "epochs": 2,
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

lstmod_args = {
    "model": "lstmod",
    "stacked_layers":1,
    "contamination":0.1, 
    "contaminations":[0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2],
    "epochs":1,
    "min_attack_time":5,
    "n_hidden_layer":2,
    "hidden_dim":8,
    "dataset_dir":f'datasets/{dataset_name}',
    "dataset_name":dataset_name,
    "dataset_dim":dataset_dim,
    "batch_size": 64,
    "hidden_size": 8,
    "anomal_col":"anomaly",
    "plot": False,
    "plot_dir": "run_scripts/out/imgs",
    "metrics_dir": "run_scripts/out/metric",
    "important_cols":['1','9','10','12','13','14','15','23'],
    "plot_cols":['9', '10', '12'],
    "use_important_cols": False,
    "sub_dataset": "null"
}

lstmae_args = {
    "model":"LSTMAE",
    "preprocessing":False,
    "hidden_neurons":[16,3,16],
    "window_size":100, 
    "stacked_layers":1,
    "contamination":0.1, 
    "contaminations":[0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2],
    "epochs":2,
    "dataset_dir":f'datasets/{dataset_name}',
    "dataset_name":dataset_name,
    "dataset_dim":dataset_dim,
    "batch_size":64,
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

telemanom_args = {
    "model": "telemanom",
    "stacked_layers":1,
    "contamination":0.1, 
    "contaminations":[0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4],
    "epochs":3,
    "n_hidden_layer":2,
    "dataset_dir":f'datasets/{dataset_name}',
    "dataset_name":dataset_name,
    "dataset_dim":dataset_dim,
    "l_s":100,
    "layers":[64,64],   # No of units for the 2 lstm layers
    "n_predictions":2,
    "window_size_":1,
    "anomal_col":"anomaly",
    "plot": False,
    "plot_dir": "run_scripts/out/imgs",
    "metrics_dir": "run_scripts/out/metric",
    "important_cols":['1','9','10','12','13','14','15','23'],
    "plot_cols":['9', '10', '12'],
    "use_important_cols": False,
    "sub_dataset": "null"
}

deeplog_args = {
    "model":"deeplog",
    "window_size":100, 
    "stacked_layers":1,
    "contamination":0.1, 
    "contaminations":[0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2],
    "epochs":2,
    "dataset_dir":'datasets/SMAP',
    "dataset_dir":f'datasets/{dataset_name}',
    "dataset_name":dataset_name,
    "dataset_dim":dataset_dim,
    "batch_size":50,
    "anomal_col":"anomaly",
    "hidden_size":64,
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

lstmvaedistgmm_args = {
    "model":"LSTMVAEDISTGMM",
    "num_gmm":4,
    "preprocessing":False,
    "window_size":100, 
    "batch_size":64,
    "hidden_size":64,
    "encoder_neurons":[32,16],
    "decoder_neurons":[16,32],
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
    "important_cols":['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18'],
    "plot_cols":['9','10','12'],
    "use_important_cols":False,
    "sub_dataset":"null"
}

gruvaegmm_args = {
    "model":"GRUVAEGMM",
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

lstmaegmm_args = {
    "model":"LSTMAEGMM",
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

def prepare_data(args):
    args['model_dir'] = "run_scripts/out/models"
    # path
    dataset_dir = args['dataset_dir']
    train_data_path = os.path.join(dataset_dir,"train.csv") 
    test_data_path = os.path.join(dataset_dir,"test.csv") 
    test_label_path = os.path.join(dataset_dir,"test_label.csv") 
    # read
    train_df=pd.read_csv(train_data_path)
    test_df=pd.read_csv(test_data_path)
    test_label_df=pd.read_csv(test_label_path)
    
    train_np = train_df.values[:,1:] # (n,25)
    test_np = test_df.values[:,1:]  # (n,25)
    test_label_np = test_label_df.values[:,1:]  # (n,1)
    
    train_np = np.nan_to_num(train_np)
    test_np = np.nan_to_num(test_np)
    test_label_np = np.nan_to_num(test_label_np)
    
    # normalize
    scaler = StandardScaler()
    train_np = scaler.fit_transform(train_np)
    test_np = scaler.transform(test_np)

    # test_with_label
    columns = [str(i+1) for i in range(args['dataset_dim'])]
    
    test_with_label_df = pd.DataFrame(np.concatenate([test_np,test_label_np],axis=1))
    columns.append(args['anomal_col']) # inplace
    test_with_label_df.columns = columns
    test_with_label_df[args['anomal_col']] = test_with_label_df[args['anomal_col']].astype(int)    

    
    # plot 
    if args["plot"]:
        plot_before_train(args, df=test_with_label_df)
    
    return train_np, test_np, test_with_label_df 


def train(model):
    model2args = {
        "DAGMM":dagmm_args,
        "lstmod":lstmod_args,
        "LSTMAE":lstmae_args,
        "telemanom":telemanom_args,
        "deeplog":deeplog_args,
        "LSTMVAEGMM": lstmvaegmm_args,
        "LSTMVAEDISTGMM":lstmvaedistgmm_args,
        "GRUVAEGMM": gruvaegmm_args,
        "LSTMAEGMM": lstmaegmm_args
    }
    args = model2args[model]
    args['model_dir'] = "run_scripts/out/models"
    train_np, test_np, test_with_label_df = prepare_data(args)  # 已归一化

    
    model2ski = {
        "DAGMM":DAGMMSKI(
            normalize = model2args["DAGMM"]["normalize"],
            comp_hiddens = model2args["DAGMM"]["comp_hiddens"],
            est_hiddens = model2args["DAGMM"]["est_hiddens"],
            minibatch_size = model2args["DAGMM"]["minibatch_size"],
            epoch_size = model2args["DAGMM"]["epoch_size"],
        ),
        "lstmod":LSTMODetectorSKI(
            min_attack_time = model2args["lstmod"]['min_attack_time'],
            epochs = model2args["lstmod"]['epochs'],
            batch_size = model2args["lstmod"]['batch_size'],
            hidden_dim = model2args["lstmod"]['hidden_dim'],
            n_hidden_layer = model2args["lstmod"]['n_hidden_layer']
        ),
        "LSTMAE":LSTMAESKI(
            window_size = model2args["LSTMAE"]['window_size'],
            preprocessing = model2args["LSTMAE"]["preprocessing"],
            batch_size = model2args["LSTMAE"]["batch_size"],
            epochs = model2args["LSTMAE"]["epochs"],
            hidden_neurons = model2args["LSTMAE"]["hidden_neurons"],
            hidden_size=model2args["LSTMAE"]['hidden_size']
        ),
        "telemanom":TelemanomSKI(
            epochs = model2args["telemanom"]['epochs'],
            l_s = model2args["telemanom"]['l_s'],
            n_predictions = model2args["telemanom"]['n_predictions'],
            layers = model2args["telemanom"]['layers'],
            window_size_ = model2args["telemanom"]['window_size_']
        ),
        "deeplog":DeepLogSKI(
            window_size=model2args["deeplog"]['window_size'],
            stacked_layers=model2args["deeplog"]['stacked_layers'],
            contamination=model2args["deeplog"]['contamination'],
            epochs=model2args["deeplog"]['epochs'],
            batch_size = model2args["deeplog"]['batch_size'],
            hidden_size=model2args["deeplog"]['hidden_size']
        ),
        "LSTMVAEGMM":LSTMVAEGMMSKI(
            num_gmm = model2args["LSTMVAEGMM"]["num_gmm"],
            window_size=model2args["LSTMVAEGMM"]['window_size'],
            hidden_size = model2args["LSTMVAEGMM"]['hidden_size'],
            preprocessing = model2args["LSTMVAEGMM"]["preprocessing"],
            batch_size = model2args["LSTMVAEGMM"]["batch_size"],
            epochs = model2args["LSTMVAEGMM"]["epochs"],
            latent_dim = model2args["LSTMVAEGMM"]["latent_dim"],
            encoder_neurons = model2args["LSTMVAEGMM"]["encoder_neurons"],
            decoder_neurons = model2args["LSTMVAEGMM"]["decoder_neurons"]
        ),
        "LSTMVAEDISTGMM":LSTMVAEDISTGMMSKI(
            num_gmm = model2args["LSTMVAEDISTGMM"]["num_gmm"],
            window_size=model2args["LSTMVAEDISTGMM"]['window_size'],
            hidden_size = model2args["LSTMVAEDISTGMM"]['hidden_size'],
            preprocessing = model2args["LSTMVAEDISTGMM"]["preprocessing"],
            batch_size = model2args["LSTMVAEDISTGMM"]["batch_size"],
            epochs = model2args["LSTMVAEDISTGMM"]["epochs"],
            latent_dim = model2args["LSTMVAEDISTGMM"]["latent_dim"],
            encoder_neurons = model2args["LSTMVAEDISTGMM"]["encoder_neurons"],
            decoder_neurons = model2args["LSTMVAEDISTGMM"]["decoder_neurons"]
        ),
        "GRUVAEGMM":GRUVAEGMMSKI(
            num_gmm = model2args["GRUVAEGMM"]["num_gmm"],
            window_size=model2args["GRUVAEGMM"]['window_size'],
            hidden_size = model2args["GRUVAEGMM"]['hidden_size'],
            preprocessing = model2args["GRUVAEGMM"]["preprocessing"],
            batch_size = model2args["GRUVAEGMM"]["batch_size"],
            epochs = model2args["GRUVAEGMM"]["epochs"],
            latent_dim = model2args["GRUVAEGMM"]["latent_dim"],
            encoder_neurons = model2args["GRUVAEGMM"]["encoder_neurons"],
            decoder_neurons = model2args["GRUVAEGMM"]["decoder_neurons"]
        ),
        "LSTMAEGMM":LSTMAEGMMSKI(
            num_gmm = model2args["LSTMAEGMM"]["num_gmm"],
            window_size=model2args["LSTMAEGMM"]['window_size'],
            hidden_size = model2args["LSTMAEGMM"]['hidden_size'],
            preprocessing = model2args["LSTMAEGMM"]["preprocessing"],
            batch_size = model2args["LSTMAEGMM"]["batch_size"],
            epochs = model2args["LSTMAEGMM"]["epochs"],
            latent_dim = model2args["LSTMAEGMM"]["latent_dim"],
            encoder_neurons = model2args["LSTMAEGMM"]["encoder_neurons"],
            decoder_neurons = model2args["LSTMAEGMM"]["decoder_neurons"]
        )
    }
    
    transformer_DL = model2ski[model]
    
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
    


if __name__ == "__main__":
    # models = ["DAGMM", "lstmod", "LSTMAE", "telemanom","deeplog", "LSTMVAEGMM"]
    models = ["LSTMAEGMM","GRUVAEGMM","LSTMVAEDISTGMM"] 
    for m in models:
        print(f" < * > {m} " * 20)
        train(m)