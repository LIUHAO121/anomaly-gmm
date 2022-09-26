from genericpath import exists
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd


def plot_one_column(df,col_name,save_path):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    plt.plot(df[col_name], label=col_name)
    plt.legend()
    plt.savefig(save_path)
    
def plot_one_column_with_label(df,col_name, anomal_col, save_path):
    a = df.loc[df[anomal_col] == 1]
    outlier_index=list(a.index)
    fig = plt.figure(facecolor='white',figsize=(50,10))
    
    ax = fig.add_subplot(111)
    plt.plot(df[col_name], label=col_name)
    ax.scatter(a.index ,a[col_name], color='red', label = 'Anomaly', s=16)
    plt.legend()
    plt.savefig(save_path)
    

 
def plot_one_column_dense(df,col_name,save_path):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    sns.kdeplot(df[col_name],label=col_name)
    plt.legend()
    plt.savefig(save_path)

def plot_predict(df,col_name,anomal_col,predict,threshold,save_path):
    df = df.reset_index()
    a = df.loc[df[anomal_col] == 1]

    fig=plt.figure(facecolor='white',figsize=(35,20))
    ax1 = fig.add_subplot(211)
    ax1.plot(df[col_name], color='black', label = 'Normal', linewidth = 1.5)
    ax1.scatter(a.index ,a[col_name], color='red', label = 'Anomaly', s = 20)
    plt.legend(fontsize=25,loc="upper right")
    ax2 = fig.add_subplot(212)
    ax2.plot(predict, color='blue', label = 'Score', linewidth = 0.5)
    ax2.plot(threshold, color='green', label = 'threshold', linewidth = 1.5)
    plt.legend(fontsize=25, loc="upper right")
    plt.savefig(save_path)
    plt.close('all')
    print("save picture {} ...".format(save_path))
    

def plot_predict_to_many_imgs(args, df,col_name,anomal_col,predict,threshold,save_dir,segment=400):
    df = df.reset_index()
    data_len = df.shape[0]
    img_num = data_len//segment
    rolling_size = args['contamination']
    predict_rolling_series = predict.rolling(rolling_size,center=True).max()
    predict_rolling_series[predict_rolling_series.isnull()] = predict[predict_rolling_series.isnull()]
        
    for i in range(img_num):
        start=i*segment
        end = (i+1)*segment
        part_df = df.iloc[start:end,:].reset_index(drop=True) #不要尝试在数据帧列中插入索引
        a = part_df.loc[part_df[anomal_col] == 1]
        
        part_predict = predict[start:end]
        part_threshold=threshold[start:end].reset_index(drop=True)
        part_rolling = predict_rolling_series[start:end].reset_index(drop=True)
        part_name = "{}_{}".format(start,end)
        
        fig=plt.figure(facecolor='white',figsize=(30,25))
        ax1 = fig.add_subplot(211)
        ax1.plot(part_df[col_name], color='black', label = 'Normal', linewidth = 2)
        ax1.scatter(a.index ,a[col_name], color='red', label = 'Anomaly', s =30)

        ax2 = fig.add_subplot(212)
        ax2.plot(part_predict, color='blue', label = 'Score', linewidth = 2)

        if args.get('sub_dataset',None) != None:
            save_path = os.path.join(save_dir,"{}_{}_{}_{}_{}_predict.png".format(args['dataset_name'],args['sub_dataset'],args['model'],args['sub_dataset'],part_name))
        else:
            save_path = os.path.join(save_dir,"{}_{}_{}_{}_predict.png".format(args['dataset_name'],args['model'],args['sub_dataset'],part_name))
        plt.savefig(save_path)
        plt.close('all')
        print("save picture {} ...".format(save_path))
    
    

def plot_multi_columns(df,col_names,save_path):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    for col_name in col_names:
        plt.plot(df[col_name], label=col_name)
    plt.legend(col_names,title='multi columns')
    plt.savefig(save_path)
     
    
def plot_anomal_multi_columns(df, col_names, anomal_col,save_path):
    a = df.loc[df[anomal_col] == 1]
    outlier_index=list(a.index)
    # print("outlier_index: ",outlier_index)
    fig, ax = plt.subplots(figsize=(10,6))   
    for col_name in col_names:
        ax.plot(df[col_name], label=col_name)
    for col_name in col_names:
        ax.scatter(a.index ,a[col_name], color='red',label="anomaly", s=16)
    plt.legend(col_names)
    plt.savefig(save_path)
    

def plot_anomal_multi_columns_3d(df,col_names, anomal_col,save_path):
    assert len(col_names) <= 3 ,"too many cols for 3d plot"
    a = df.loc[df[anomal_col] == 1]
    outlier_index=list(a.index)
    # print("outlier_index: ",outlier_index)
    fig = plt.figure(figsize=(25,25))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[col_names[0]], 
               df[col_names[1]], 
               zs=df[col_names[2]],
               s=3, lw=1, label="inliers", c="blue")

    ax.scatter(df.loc[outlier_index, col_names[0]], 
               df.loc[outlier_index, col_names[1]], 
               df.loc[outlier_index, col_names[2]],
                lw=1, s=3, c="red", label="outliers")
    ax.legend()
    plt.savefig(save_path)
    
    
def plot_zspace_3d(df,col_names,anomal_col,save_path):
    assert len(col_names) <= 3 ,"too many cols for 3d plot"
    df = df.reset_index()
    a = df.loc[df[anomal_col] == 1]
    outlier_index=list(a.index)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[col_names[0]], 
               df[col_names[1]], 
               zs=df[col_names[2]],
               s=5, lw=1, label="normal", c="blue")

    ax.scatter( df.loc[outlier_index, col_names[0]], 
                df.loc[outlier_index, col_names[1]],
                zs=df.loc[outlier_index, col_names[2]],
                lw=1, s=15, c="red", label=anomal_col)
    ax.legend()
    plt.savefig(save_path)
    print("save 3d picture to {}".format(save_path))
    
def plot_before_train(args, df):

    """
    df 必须包括标注列
    """
    os.makedirs(os.path.join(args['plot_dir'],args['dataset_name']),exist_ok=True)
    for col in df.columns[:-1]:
        plot_one_column_with_label(
            df=df,
            col_name=col,
            anomal_col=args['anomal_col'],
            save_path=os.path.join(args['plot_dir'],args['dataset_name'],"{}_{}.png".format(args['dataset_name'],col)))
        
    # plot_anomal_multi_columns_3d(
    #                     df,
    #                     col_names=args['plot_cols'],
    #                     anomal_col=args['anomal_col'],
    #                     save_path=os.path.join(args['plot_dir'],'{}_multicols_3d.png'.format(args['dataset_name']))
    #                     )  

def plot_after_train(args,df,predict):
    
    """
    df 必须包括标注列
    """
    # threshold =  np.percentile(predict, 100 * (1 - args['contamination']))
    threshold =  np.percentile(predict, 100 * (1 - 0.1))
    threshod_series = pd.Series([threshold for i in range(len(predict))])
      # for drawing 
    
    for col in df.columns[:-1][:1]:
        os.makedirs(os.path.join(args['plot_dir'],args['dataset_name']),exist_ok=True)
        plot_predict_to_many_imgs(
                    args,
                    df, 
                    col_name=col,
                    anomal_col=args['anomal_col'], 
                    predict=predict, 
                    threshold=threshod_series,
                    save_dir=os.path.join(args['plot_dir'],args['dataset_name']),
                    segment=500
                     )


def diffience(series,window=1):
    series_cp = series.copy()
    s_len = len(series_cp)
    for j in range(window):
        series_cp[j]=0.0
    for i in range(window,s_len):
        series_cp[i] = abs(series[i] - series[i-window:i].mean())
    return series_cp 
        
def diffience2(series,window=1):
    series_cp = series.copy()
    diff_list1=[] # 1阶
    diff_list2=[] # 2阶
    s_len = len(series_cp)
    for i in range(s_len-1):
        diff_list1.append(abs(series_cp[i+1]-series_cp[i]))
    for j in range(s_len-2):
        diff_list2.append(abs(diff_list1[j+1]-diff_list1[j]))
    diff_list2.insert(0,0.0)
    diff_list2.insert(0,0.0)
    return pd.Series(diff_list2)
    
    
def plot_generate():
    anomaly_types = ["point_global","point_contextual","collective_global","collective_seasonal","collective_trend"]
    plt.figure(figsize=(15, 8))
    for index,anomaly_type in enumerate(anomaly_types):
        data_dir = "datasets/SYN"
        path=os.path.join(data_dir,anomaly_type + ".csv")
        df=pd.read_csv(path)
        anomal_col = "anomaly"
        value_col = "col_0"
        a = df.loc[df[anomal_col] == 1]
        
        plt.subplot(2,5,index+1)
        plt.plot(df[value_col],color='black', linewidth = 1)
        plt.scatter(x=a.index,y=a[value_col],color='red', s = 1)
        title = " ".join(anomaly_type.split("_"))
        plt.title(title)
        plt.legend(fontsize=1)
        plt.subplot(2,5,index+6)
        plt.plot(diffience(df[value_col],window=1),color='blue', linewidth = 1)
    plt.subplot(2,5,1)
    plt.ylabel("time series")
    plt.subplot(2,5,6)
    plt.ylabel("difference")
    
    plt.savefig("difference.png")
    
    
if __name__ == "__main__":
    plot_generate()
   