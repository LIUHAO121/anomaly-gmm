import os
import tensorflow as tf
import pandas as pd
from run_scripts.metric_tools import multi_threshold_eval
from run_scripts.plot_tools import plot_after_train

    
    
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
    
    
def eval_step(args,transformer_DL,test_np,test_with_label_df):
    print("> "* 50)
    print("runing eval step ...")
    model_path = os.path.join(args['model_dir'],"{}_{}_{}".format(args['dataset_name'],args['model'],args['sub_dataset']))
    # 不同的模型可能包含不同的自定义计算，导致模型在load时有不同的写法，这种不同不能体现在通用函数(eval_step)里，应该体现在模型内部。
    print("> "* 50)
    print("run predict ... ")
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