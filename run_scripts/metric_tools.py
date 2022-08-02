
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
        res['contamination'].append(round(contamination,4))
        res['thresholds'].append(round(threshold))
        res['precision'].append(round(precision,4))
        res['recall'].append(round(recall,4))
        res['f1'].append(round(f1, 4))
    res_df = pd.DataFrame(res)
    print(res_df)
    res_df.to_csv(os.path.join(args['metrics_dir'],"{}_{}_{}.csv".format(args['dataset_name'], args['model'], args['sub_dataset'])))
    return res


def merge_smd_metric(metric_dir,model):
    columns = ['contamination', 'thresholds', 'precision', 'recall', 'f1']
    dataset = "SMD"
    metric_files = glob(f"{metric_dir}/{dataset}_{model}_machine*.csv")
    res = {'contamination':[],"thresholds":[],'precision':[],'recall':[],'f1':[]}
    df_list = []
    
    for f in metric_files:
        metric_df = pd.read_csv(f)
        df_list.append(metric_df)
    df_num = len(df_list)
    for col in columns:
        value_series = pd.Series([0.0 for i in range(len(df_list[0]['contamination']))])
        for df in df_list:
            value_series += df[col]
        value_series /= df_num
        res[col] = list(value_series)
        res[col] = [round(i,4) for i in res[col]]
    res_df = pd.DataFrame(res)
    res_df.to_csv(f"{metric_dir}/{dataset}_{model}_null.csv")
        
        
def merge_all_metric(metric_dir,models):
    datasets = ['MSL','SMAP','PSM','SMD','SYN']
    res = {'contamination':[],'precision':[],'recall':[],'f1':[]}
    for d in datasets:
        res = {'models':[],'contamination':[],'precision':[],'recall':[],'f1':[]}
        for m in models:
            metric_file = os.path.join(metric_dir,f"{d}_{m}_null.csv")
            df = pd.read_csv(metric_file)
            best_f1_df = df[df['f1']==df['f1'].max()]
            res['contamination'].append(round(best_f1_df['contamination'].to_list()[0],4))
            res['precision'].append(round(best_f1_df['precision'].to_list()[0],4))        
            res['recall'].append(round(best_f1_df['recall'].to_list()[0],4))  
            res['f1'].append(round(best_f1_df['f1'].to_list()[0],4))   
            res['models'].append(m)
        res_df = pd.DataFrame(res)
        res_df.to_csv(os.path.join(metric_dir,f"summary/{d}_metric.csv"))
    
    
    
    
if __name__ == "__main__":

    metric_dir = "run_scripts/out/metric"
    models = [ "DAGMM", "lstmod", "LSTMAE","LSTMVAE",  "telemanom", "deeplog", "LSTMVAEGMM","LSTMAEGMM","GRUVAEGMM","LSTMVAEDISTGMM","LSTMGMM"]
    
    # for m in models:
    #     merge_smd_metric(metric_dir=metric_dir,model=m)
    
    
    merge_all_metric(metric_dir=metric_dir,models=models)
                                