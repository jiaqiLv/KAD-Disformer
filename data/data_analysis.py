import pandas as pd
from os.path import join,exists
import numpy as np
from scipy.stats import spearmanr

def calc_interval(timestamps):
    diff = np.diff(timestamps)
    intervals, counts = np.unique(diff, return_counts=True)
    tmp = np.hstack((intervals.reshape(-1,1),counts.reshape(-1,1)))
    return tmp[np.argmax(tmp[:,1])][0]

def fill_missing_points(tv_df, interval):
    timestamps, values, labels = tv_df['timestamp'].values, tv_df['value'].values, tv_df['label'].values
    start = timestamps[0]
    f_t, f_v, f_l = [],[],[]
    index = 0
    while start <= timestamps[-1]:
        if start == timestamps[index]:
            f_t.append(start)
            f_v.append(values[index])
            f_l.append(labels[index])
            index += 1
        else:
            f_t.append(start)
            f_v.append(np.nan)
            f_l.append(0)
        start += interval
    r_df = pd.DataFrame({
        'timestamp':f_t,
        'value':f_v,
        'label':f_l
    })
    r_df['value'] = r_df['value'].interpolate()
    return r_df

def smooth_values(values, p=3):
    mean,std = np.mean(values),np.std(values)
    lower, upper = mean-p*std,mean+p*std
    results = []
    for value in values:
        if lower <= value <= upper:
            results.append(value)
        else:
            results.append(lower if value<lower else upper)
    return results

def calc_spearmanr(values, points):
    if points <= 0 or points > len(values):
        raise ValueError("points must be greater than 0 and less than or equal to the length of the values.")
    slices = [values[i:i+points] for i in range(0,len(values),points)]
    spearman_coeffs = []
    for i in range(len(slices)-1):
        coeff,_ = spearmanr(slices[i],slices[i+1])
        spearman_coeffs.append(coeff)
    if spearman_coeffs:
        return np.mean(spearman_coeffs)
    else:
        return None

def extract_seasonality(values, interval, t_w = 1.0, t_d = 1.0, t_season = 0.6):
    s_values = smooth_values(values)
    points_per_day = int(60 * 60 * 24 / interval)
    points_per_week = points_per_day*7
    if len(values)<points_per_day:
        return "NON"
    
    p_w = calc_spearmanr(s_values,points_per_week)
    p_d = calc_spearmanr(s_values,points_per_day)
    
    if p_w < t_season and p_d < t_season:
        return "NON"
    if p_w * t_w >= p_d * t_d:
        return "WEEK"
    if p_w * t_w < p_d * t_d:
        return "DAY"
    return "NON"
    


if __name__ == '__main__':
    # data_path = '/code/model/KAD-Disformer/data/KPI-Anomaly-Detection/Preliminary_dataset'
    # train_df = pd.read_csv(join(data_path,'train.csv'))

    data_path = '/code/model/KAD-Disformer/data/KPI-Anomaly-Detection/Finals_dataset'
    train_df = pd.read_csv(join(data_path,'phase2_train.csv'))
    test_df = pd.read_hdf(join(data_path,'phase2_ground_truth.hdf'))

    kpi_dfs = train_df.groupby('KPI ID')

    """1 The number of data items for each KPI"""
    
    # for name,df in kpi_dfs:
    #     print(f'{name}:{df.shape}')

    """2 Basic information for KPI groups"""
    # names, v_mean, v_max, v_min, v_std, v_sum, v_size = [], [], [], [], [], [], []
    # for name, df in kpi_dfs:
    #     names.append(name)
    #     v_mean.append(df["value"].mean())
    #     v_max.append(df["value"].max())    
    #     v_min.append(df["value"].min())    
    #     v_std.append(df["value"].std())
    #     v_sum.append(df["value"].sum())
    #     v_size.append(df.shape[0])

    # basic_analysis_df = pd.DataFrame({"name": names, "size": v_size, "mean": v_mean, 
    #                                 "max": v_max, "min": v_min, "std": v_std, "sum": v_sum})
    # print(basic_analysis_df.head())

    """3 sampling interval & deficiency analysis"""
    # names, intervals, misssing_rates = [],[],[]
    # for name,df in kpi_dfs:
    #     df = df.sort_values(by='timestamp')
    #     names.append(name)
    #     interval = calc_interval(df['timestamp'].values)
    #     intervals.append(interval)
    #     target_ckpts = int((df['timestamp'].values[-1]-df['timestamp'].values[0])/interval) + 1 
    #     misssing_rate = (target_ckpts-df.shape[0])/target_ckpts
    #     misssing_rates.append(misssing_rate)
    
    # interval_df = pd.DataFrame(
    #     {
    #         'name':names,
    #         'interval':intervals,
    #         'misssing rate':misssing_rates
    #     }
    # )
    # print(interval_df)

    """4 data preprocess"""
    for name,df in kpi_dfs:
        df = df.sort_values(by='timestamp')
        interval = calc_interval(df['timestamp'].values)
        r_df = fill_missing_points(df,interval)
        r_df['value'] = smooth_values(r_df['value'])
        """save data"""
        base_path = '/code/model/KAD-Disformer/data/KPI-Anomaly-Detection/Processed_dataset'
        r_df.to_csv(join(base_path,f'{name}.csv'),index=False)

    # print(train_df.head())
    # print(train_df['KPI ID'].unique())
    # print(train_df.nunique())