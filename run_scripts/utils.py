import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tods.sk_interface.detection_algorithm.Telemanom_skinterface import TelemanomSKI
from tods.sk_interface.detection_algorithm.DeepLog_skinterface import DeepLogSKI
from tods.sk_interface.detection_algorithm.LSTMODetector_skinterface import LSTMODetectorSKI
from tods.sk_interface.detection_algorithm.AutoEncoder_skinterface import AutoEncoderSKI
from tods.sk_interface.detection_algorithm.VariationalAutoEncoder_skinterface import VariationalAutoEncoderSKI
from tods.sk_interface.detection_algorithm.LSTMAE_skinterface import LSTMAESKI
from tods.sk_interface.detection_algorithm.LSTMVAE_skinterface import LSTMVAESKI
from tods.sk_interface.detection_algorithm.DAGMM_skinterface import DAGMMSKI
from tods.sk_interface.detection_algorithm.LSTMVAEGMM_skinterface import LSTMVAEGMMSKI
from tods.sk_interface.detection_algorithm.LSTMVAEDISTGMM_skinterface import LSTMVAEDISTGMMSKI
from tods.sk_interface.detection_algorithm.GRUVAEGMM_skinterface import GRUVAEGMMSKI
from tods.sk_interface.detection_algorithm.LSTMAEGMM_skinterface import LSTMAEGMMSKI
from tods.sk_interface.detection_algorithm.LSTMGMM_skinterface import LSTMGMMSKI
from tods.sk_interface.detection_algorithm.LSTMVAEGMM2_skinterface import LSTMVAEGMM2SKI
from run_scripts.metric_tools import multi_threshold_eval
from run_scripts.plot_tools import plot_after_train,plot_zspace_3d

    
    
def train_step(args,transformer_DL,train_np,test_np,test_with_label_df):
    print("*"* 120)
    print("runing train step ...")
    transformer_DL.fit(train_np)
    prediction_score_DL = transformer_DL.predict_score(test_np) # shape = (n,1)
    
    y_true = test_with_label_df[args['anomal_col']]
    y_score = pd.Series(prediction_score_DL.flatten())
    res = multi_threshold_eval(args=args, pred_score=y_score, label=y_true)
    
    model_path = os.path.join(args['model_dir'],"{}_{}_{}".format(args['dataset_name'],args['model'],args['sub_dataset']))
    if args['model'] not in ['DAGMM',"lstmod", "LSTMAE", "telemanom"]:
        for primitive in transformer_DL.primitives:
            primitive._clf.model_.save(model_path,save_format="tf")
    
    best_f1_index = res['f1'].index(max(res['f1']))
    args['contamination'] = res['contamination'][best_f1_index]
    plot_after_train(
                args,
                df=test_with_label_df,
                predict=y_score
                     )
    
    
def eval_step(args,transformer_DL,test_np,test_with_label_df):
    print("> "* 50)
    print("runing eval step ...")
    model_path = os.path.join(args['model_dir'],"{}_{}_{}".format(args['dataset_name'],args['model'],args['sub_dataset']))
    # 不同的模型可能包含不同的自定义计算，导致模型在load时有不同的写法，这种不同不能体现在通用函数(eval_step)里，应该体现在模型内部。
    print("> "* 50)
    print("run predict ... ")
    pred_scores=None
    if args["model"] == "LSTMVAEGMM2":
        energy_latent = transformer_DL.primitives[0]._clf.latent_predict(model_path,test_np)
        samples,dim = energy_latent.shape
        y_true = np.array(test_with_label_df[args['anomal_col']]).reshape(-1,1)[-samples:,:]
        # energy_latent_label = np.concatenate((energy_latent,y_true),axis=1)
        # energy_latent_label_df = pd.DataFrame(energy_latent_label)
        # columns = ["energy","latent_1","latent_2",args['anomal_col']]
        # energy_latent_label_df.columns = columns
        # energy_latent_label_df[args['anomal_col']] = energy_latent_label_df[args['anomal_col']].astype(int)
        # save_path = os.path.join(args['plot_dir'],args['dataset_name'],"{}_{}_{}_3d.png".format(args['dataset_name'],args['model'],args['sub_dataset']))
        # plot_zspace_3d(energy_latent_label_df,col_names=columns[:3],anomal_col=args['anomal_col'],save_path=save_path)
        
        y_score = pd.Series(energy_latent[:,0].flatten())
        
        plot_after_train(
                    args,
                    df=test_with_label_df.iloc[-samples:,:],
                    predict=y_score
                        )
        
    else:
        pred_scores = transformer_DL.primitives[0]._clf.load_decision_function(model_path,test_np)
        y_true = test_with_label_df[args['anomal_col']]
        y_score = pd.Series(pred_scores.flatten())
        print("> "* 50)
        print("run eval ....")
        res = multi_threshold_eval(args=args, pred_score=y_score, label=y_true)
        
        best_f1_index = res['f1'].index(max(res['f1']))
        
        args['contamination'] = res['contamination'][best_f1_index]
        
        print("> "* 50)
        print("run plot ....")
        plot_after_train(
                    args,
                    df=test_with_label_df,
                    predict=y_score
                        )
    
    
dagmm_args = {
    "model":"DAGMM",
    "normalize":False,
    "comp_hiddens":[16,8,1],
    "est_hiddens":[8,4],
    "minibatch_size":1024,
    "epoch_size":100,
    "contaminations":[0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2],
    "epochs": 2,
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
    "anomal_col":"anomaly",
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
    "epochs":2,
    "n_hidden_layer":2,
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
    "contaminations":[0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1],
    "epochs":2,
    "dataset_dir":'datasets/SMAP',
    "batch_size":64,
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



lstmvaedistgmm_args = {
    "model":"LSTMVAEDISTGMM",
    "num_gmm":4,
    "preprocessing":False,
    "window_size":100, 
    "batch_size":128,
    "hidden_size":32,
    "encoder_neurons":[32,16],
    "decoder_neurons":[16,32],
    "latent_dim":2,
    "contaminations":[0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2],
    "contamination":0.01,
    "epochs": 1,
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
    "contaminations":[0.001, 0.005,0.007, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2],
    "contamination":0.01,
    "epochs": 2,
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
    "contaminations":[0.0001 * i for i in range(0,500,10)],
    "contamination":0.01,
    "epochs": 1,
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


lstmgmm_args = {
    "model":"LSTMGMM",
    "num_gmm":4,
    "preprocessing":False,
    "window_size":100, 
    "batch_size":64,
    "hidden_size":64,
    "contaminations":[0.0001 * i for i in range(0,500,10)],
    "contamination":0.01,
    "epochs": 1,
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

lstmvaegmm2_args = {
    "model":"LSTMVAEGMM2",
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
    "epochs": 6,
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
    "lamta":0.1, # loss funciton
    "contaminations":[0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2],
    "contamination": 0.01,
    "epochs": 1,
    "anomal_col": "anomaly",
    "plot": False,
    "plot_dir": "run_scripts/out/imgs",
    "metrics_dir": "run_scripts/out/metric",
    "model_dir": "run_scripts/out/models",
    "important_cols":['1','9','10','12','13','14','15','23'],
    "plot_cols":['9','10','12'],
    "use_important_cols":False,
    "sub_dataset":"null"
}

vis_start_end = {
    "MSL":{"start":21000,"end":22200},
    "SMAP":{"start":365000,"end":370000},
    "PSM":{"start":7000,"end":8200},
    "SMD":{"start":0,"end":120000},
    "SYN":{"start":0,"end":120000},
    "SWaT":{"start":0,"end":50000}
}

def train(model,dataset_name,dataset_dim,prepare_data,machine_name=None,num_gmm=None):
    model2args = {
        "DAGMM":dagmm_args,
        "lstmod":lstmod_args,
        "LSTMAE":lstmae_args,
        "LSTMVAE":lstmvae_args,
        "telemanom":telemanom_args,
        "deeplog":deeplog_args,
        "LSTMVAEGMM": lstmvaegmm_args,
        "LSTMVAEDISTGMM":lstmvaedistgmm_args,
        "GRUVAEGMM": gruvaegmm_args,
        "LSTMAEGMM": lstmaegmm_args,
        "LSTMGMM":lstmgmm_args,
        "LSTMVAEGMM2":lstmvaegmm2_args
    }
    args = model2args[model]
    args['model_dir'] = "run_scripts/out/models"
    args['dataset_dir'] = f'datasets/{dataset_name}'
    args['dataset_name'] = dataset_name
    args['dataset_dim'] = dataset_dim
    args['num_gmm']=num_gmm
    
    train_np=None
    test_np=None
    test_with_label_df=None
    
    if machine_name != None:
        train_np, test_np, test_with_label_df = prepare_data(args,machine_name)
        args['sub_dataset'] = machine_name
    else:
        train_np, test_np, test_with_label_df = prepare_data(args)  # 已归一化
        
    # dataset_name = args["dataset_name"]
    # test_np = test_np[vis_start_end[dataset_name]["start"]:vis_start_end[dataset_name]["end"]]
    # test_with_label_df = test_with_label_df[vis_start_end[dataset_name]["start"]:vis_start_end[dataset_name]["end"]]

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
        "LSTMVAE":LSTMVAESKI(
            window_size=model2args["LSTMVAE"]['window_size'],
            hidden_size = model2args["LSTMVAE"]['hidden_size'],
            preprocessing = model2args["LSTMVAE"]["preprocessing"],
            batch_size = model2args["LSTMVAE"]["batch_size"],
            epochs = model2args["LSTMVAE"]["epochs"],
            latent_dim = model2args["LSTMVAE"]["latent_dim"],
            encoder_neurons = model2args["LSTMVAE"]["encoder_neurons"],
            decoder_neurons = model2args["LSTMVAE"]["decoder_neurons"],
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
            decoder_neurons = model2args["LSTMVAEGMM"]["decoder_neurons"],
            lamta = model2args["LSTMVAEGMM"]["lamta"],
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
        ),
        "LSTMGMM":LSTMGMMSKI(
            num_gmm = model2args["LSTMAEGMM"]["num_gmm"],
            window_size=model2args["LSTMAEGMM"]['window_size'],
            hidden_size = model2args["LSTMAEGMM"]['hidden_size'],
            preprocessing = model2args["LSTMAEGMM"]["preprocessing"],
            batch_size = model2args["LSTMAEGMM"]["batch_size"],
            epochs = model2args["LSTMAEGMM"]["epochs"]
        ),
        "LSTMVAEGMM2":LSTMVAEGMM2SKI(
            num_gmm = model2args["LSTMVAEGMM2"]["num_gmm"],
            window_size=model2args["LSTMVAEGMM2"]['window_size'],
            hidden_size = model2args["LSTMVAEGMM2"]['hidden_size'],
            preprocessing = model2args["LSTMVAEGMM2"]["preprocessing"],
            batch_size = model2args["LSTMVAEGMM2"]["batch_size"],
            epochs = model2args["LSTMVAEGMM2"]["epochs"],
            latent_dim = model2args["LSTMVAEGMM2"]["latent_dim"],
            encoder_neurons = model2args["LSTMVAEGMM2"]["encoder_neurons"],
            decoder_neurons = model2args["LSTMVAEGMM2"]["decoder_neurons"]
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