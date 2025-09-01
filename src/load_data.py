#load_data.py
import pandas as pd
import data_preprocessing

def load_process_save():
    df = pd.read_csv(r'C:\Users\20109\Desktop\Kaggle\NYC-Taxi-Trip-Duration\data\Raw Data\train.csv')
    val = pd.read_csv(r'C:\Users\20109\Desktop\Kaggle\NYC-Taxi-Trip-Duration\data\Raw Data\val.csv')
    test = pd.read_csv(r'C:\Users\20109\Desktop\Kaggle\NYC-Taxi-Trip-Duration\data\Raw Data\test.csv')


    df = data_preprocessing.preprocess(df)
    val = data_preprocessing.preprocess(val)
    test = data_preprocessing.preprocess(test)

    df = data_preprocessing.remove_outliers(df)

    df.to_csv(r"C:\Users\20109\Desktop\Kaggle\NYC-Taxi-Trip-Duration\data\Processed Data\train.csv", index=False)
    val.to_csv(r"C:\Users\20109\Desktop\Kaggle\NYC-Taxi-Trip-Duration\data\Processed Data\val.csv", index=False)
    test.to_csv(r"C:\Users\20109\Desktop\Kaggle\NYC-Taxi-Trip-Duration\data\Processed Data\test.csv", index=False)

    return df,val,test


