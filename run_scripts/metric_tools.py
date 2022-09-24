
import numpy as np
import pandas as pd
import os
from glob import glob



def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def adjust_predicts(args, pred, label,threshold):
    assert len(pred) == len(label),"len(pred) = {},len(label) = {} ".format(len(pred),len(label))
    predict = pred >= threshold
    predict = np.asarray(predict).astype(int)
    label = np.asarray(label)
    latency = 0

    actual = label
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(predict)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    return predict
  

def multi_threshold_eval(args,pred_score,label):
    res = {"contamination":[],"thresholds":[],"precision":[],"recall":[],"f1":[]}
    contaminations = args['contaminations']
    for contamination in contaminations:
        threshold = np.percentile(pred_score, 100 * (1 - contamination))
        adjust_predict = adjust_predicts(
                                    args=args,
                                    pred=pred_score,
                                    label=label,
                                    threshold=threshold)

        f1, precision, recall, TP, TN, FP, FN = calc_point2point(adjust_predict,label)
        res['contamination'].append(round(contamination,5))
        res['thresholds'].append(round(threshold,5))
        res['precision'].append(round(precision,4))
        res['recall'].append(round(recall,4))
        res['f1'].append(round(f1, 4))
    res_df = pd.DataFrame(res)
    print(res_df)
    os.makedirs(args['metrics_dir'],exist_ok=True)
    res_df.to_csv(os.path.join(args['metrics_dir'],"{}_{}_{}.csv".format(args['dataset_name'], args['model'], args['sub_dataset'])))
    return res


def adjust_rolling_mean_predicts(args, pred, label,rolling_size=5):
    assert len(pred) == len(label),"len(pred) = {},len(label) = {} ".format(len(pred),len(label))
    pred_rolling = pred.rolling(rolling_size,center=True).max()

    predict = pred >= pred_rolling 
    
    predict = np.asarray(predict).astype(int)
    label = np.asarray(label)
    latency = 0

    actual = label
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(predict)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    return predict  


def multi_rolling_size_eval(args, pred_score,label):
    res = {"contamination":[],"precision":[],"recall":[],"f1":[]}
    rolling_sizes = args['rolling_sizes']
    for rolling_size in rolling_sizes:
        adjust_predict = adjust_rolling_mean_predicts(
            args=args,
            pred=pred_score,
            label=label,
            rolling_size=rolling_size,
        )
        f1, precision, recall, TP, TN, FP, FN = calc_point2point(adjust_predict,label)
        res['contamination'].append(round(rolling_size,5))
        res['precision'].append(round(precision,4))
        res['recall'].append(round(recall,4))
        res['f1'].append(round(f1, 4))
    res_df = pd.DataFrame(res)
    print(res_df)
    os.makedirs(args['metrics_dir'],exist_ok=True)
    res_df.to_csv(os.path.join(args['metrics_dir'],"{}_{}_{}.csv".format(args['dataset_name'], args['model'], args['sub_dataset'])))
    return res

def merge_smd_metric(metric_dir,model,dataset):

    metric_files = glob(f"{metric_dir}/{dataset}_{model}*.csv")
    res = {'contamination':[],'precision':[],'recall':[],'f1':[]}
    df_list = []
    
    for f in metric_files:
        metric_df = pd.read_csv(f)
        df_list.append(metric_df)
    df_num = len(df_list)
    assert df_num >1 , f"find no csv files file_num = {df_num} "
    for df in df_list:
        best_f1_df = df[df['f1']==df['f1'].max()]
        res['contamination'].append(best_f1_df['contamination'].tolist()[0])
        res['precision'].append(best_f1_df['precision'].tolist()[0])        
        res['recall'].append(best_f1_df['recall'].tolist()[0])  
        res['f1'].append(best_f1_df['f1'].tolist()[0]) 
        # res['thresholds'].append(best_f1_df['thresholds'].tolist()[0]) 
        
    res['contamination'] = [round(sum(res['contamination'])/len(res['contamination']),5)]
    res['precision'] = [round(sum(res['precision'])/len(res['precision']),4)]
    res['recall'] = [round(sum(res['recall'])/len(res['recall']),4)]
    res['f1'] = [round(sum(res['f1'])/len(res['f1']),4)]
    # res['thresholds'] = [round(sum(res['thresholds'])/len(res['thresholds']),5)]
    
    res_df = pd.DataFrame(res)
    
    res_df.to_csv(f"{metric_dir}/{dataset}_{model}_null.csv")
        
        
def merge_all_metric(metric_dir,models):
    datasets = ['MSL','SMAP','PSM']
    res = {'contamination':[],'precision':[],'recall':[],'f1':[]}
    for d in datasets:
        res = {'models':[],'contamination':[],'precision':[],'recall':[],'f1':[]}
        for m in models:
            metric_file = os.path.join(metric_dir,f"{d}_{m}_null.csv")
            df = pd.read_csv(metric_file)
            best_f1_df = df[df['f1']==df['f1'].max()]
            res['contamination'].append(round(best_f1_df['contamination'].to_list()[0],5))
            res['precision'].append(round(best_f1_df['precision'].to_list()[0],4))        
            res['recall'].append(round(best_f1_df['recall'].to_list()[0],4))  
            res['f1'].append(round(best_f1_df['f1'].to_list()[0],4))   
            res['models'].append(m)
        res_df = pd.DataFrame(res)
        res_df.to_csv(os.path.join(metric_dir,f"summary/{d}_metric.csv"))
    
    
def merge_pai_out_according_version(version,model,pai_out_dir,metric_dir):    
    """
    根据训练版本号合并数据,只用于消融实验结果汇总
    """
    res_documents = os.listdir(pai_out_dir)
    datasets=["MSL","SMAP","PSM","SWaT","SMD","ASD"]
    # datasets = ["SMD"]
    out={"variable":[]}
    for d in datasets:
        out[d]=[]
    for doc in res_documents:
        
        if version in doc and model in doc:
            doc_infos = doc.split("_")
            num_gmm = doc_infos[3]
            position = doc_infos[-1]  # 暂时
            doc_dir = os.path.join(pai_out_dir, doc)
            doc_metric_dir = os.path.join(doc_dir,"metric")
            variable_value = None
            if "GMM" in version:
                variable_value = num_gmm
            elif "posi" in version:
                variable_value = position
            else:
                variable_value = version
            out['variable'].append(variable_value) 
            for dataset in datasets:
                metric_file = os.path.join(doc_metric_dir,"{}_{}_null.csv".format(dataset,model))
                metric_df = pd.read_csv(metric_file)
                best_f1 = round(metric_df['f1'].max(),5)
                out[dataset].append(best_f1)
    out = pd.DataFrame(out)         
    summary_dir = os.path.join(metric_dir,"summary")
    os.makedirs(summary_dir,exist_ok=True)
    out.to_csv(os.path.join(summary_dir,f"{version}_{model}.csv"))
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Tensorflow Training')
    parser.add_argument('--models', type=str, nargs='+', default=[])
    parser.add_argument('--version', type=str, default="version")
    args = parser.parse_args()
    models = args.models
    metric_dir = "run_scripts/out/metric"
    
    
    for m in models:
        dataset="SMD"
        metric_files = glob(f"{metric_dir}/{dataset}_{m}*.csv")
        if len(metric_files) > 1:
            merge_smd_metric(metric_dir=metric_dir,model=m,dataset=dataset)
        dataset="ASD"
        metric_files = glob(f"{metric_dir}/{dataset}_{m}*.csv")
        if len(metric_files) > 1:
            merge_smd_metric(metric_dir=metric_dir,model=m,dataset=dataset)
    
    
        # 汇总消融实验的结构
        merge_pai_out_according_version(
                            version=args.version,
                            model=m,
                            pai_out_dir="/mnt/nfs-storage/user/lhao/anomal/pai_out/",
                            metric_dir=metric_dir
                            )              
        
        
# python run_scripts/metric_tools.py --models LSTMVAEGMM --version ChangeGMMNUM