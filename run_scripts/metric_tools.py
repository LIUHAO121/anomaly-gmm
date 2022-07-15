
import numpy as np
import pandas as pd
import os



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
    
    assert len(pred) == len(label)

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
                            if args['dataset_name'] != "SMD":
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
        res['contamination'].append(round(contamination,3))
        res['thresholds'].append(round(threshold,2))
        res['precision'].append(round(precision,2))
        res['recall'].append(round(recall,2))
        res['f1'].append(round(f1,2))
    res = pd.DataFrame(res)
    print(res)
    res.to_csv(os.path.join(args['metrics_dir'],"{}_{}_{}.csv".format(args['dataset_name'], args['model'], args['sub_dataset'])))
