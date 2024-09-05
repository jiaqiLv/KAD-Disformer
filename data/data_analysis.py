import pandas as pd
from os.path import join,exists

if __name__ == '__main__':
    # data_path = '/code/model/KAD-Disformer/data/KPI-Anomaly-Detection/Preliminary_dataset'
    # train_df = pd.read_csv(join(data_path,'train.csv'))

    data_path = '/code/model/KAD-Disformer/data/KPI-Anomaly-Detection/Finals_dataset'
    train_df = pd.read_csv(join(data_path,'phase2_train.csv'))
    test_df = pd.read_hdf(join(data_path,'phase2_ground_truth.hdf'))

    kpi_dfs = train_df.groupby('KPI ID')
    # The number of data items for each KPI
    
    # for name,df in kpi_dfs:
    #     print(f'{name}:{df.shape}')

    # Basic information for KPI groups
    names, v_mean, v_max, v_min, v_std, v_sum, v_size = [], [], [], [], [], [], []
    for name, df in kpi_dfs:
        names.append(name)
        v_mean.append(df["value"].mean())
        v_max.append(df["value"].max())    
        v_min.append(df["value"].min())    
        v_std.append(df["value"].std())
        v_sum.append(df["value"].sum())
        v_size.append(df.shape[0])

    basic_analysis_df = pd.DataFrame({"name": names, "size": v_size, "mean": v_mean, 
                                    "max": v_max, "min": v_min, "std": v_std, "sum": v_sum})
    # print(basic_analysis_df.head())


    # print(train_df.head())
    # print(train_df['KPI ID'].unique())
    # print(train_df.nunique())