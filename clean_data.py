import numpy as np
import pandas as pd
import pickle

def clean(file, out, split):
    if ".xlsx" in file:
        dataframe = pd.read_excel(file)
        #standardization to make features have mean of 0, std. dev of 1
        #formula is x_std = (x - mean) / std
        for column in dataframe.columns:
            if "Y" in column: continue
            mean = dataframe.loc[:, column].mean()
            std = dataframe.loc[:, column].std()
            dataframe.loc[:, column] = dataframe.loc[:, column].map(lambda x : (x-mean)/std)
    else:
        dataframe = pd.read_csv(file, header=None)
    print(dataframe)
    dataframe = dataframe.sample(frac=1).reset_index(drop=True) 
    print(dataframe)
    length = len(dataframe)
    with open(out + "_train.sav", 'wb') as handle:
        pickle.dump(dataframe[:int(length*split)], handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(out + "_test.sav", 'wb') as handle:
        pickle.dump(dataframe[int(length*split):], handle, protocol=pickle.HIGHEST_PROTOCOL)
    


classification_file = "raw_datasets\Qualitative_Bankruptcy.data.txt"
regression_file = "raw_datasets\ENB2012_data.xlsx"

clean(classification_file, "clean_data/Qualitative_Bankruptcy", 0.8)
clean(regression_file, "clean_data/ENB2012_data", 0.8)