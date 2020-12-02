from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch

def remove_max(x):
    x[x.argmax()] = np.median(x)
    print(x.argmax(), ":", x[x.argmax()], "=>", np.median(x))
    return x

def groupby_datapoint(df,
                      gb='YYYYMMDD',
                      target='Qty'):
    
    df_ts = df.groupby([gb])[target].sum().sort_index()
    df_ts = df_ts.reset_index()
    #df_ts[gb] = pd.to_datetime(df_ts[gb], format='%Y%m%d')
    df_ts[gb] = df_ts[gb].astype(int)

    df_ts.set_index([gb], inplace=True)
    df_ts.sort_index(inplace=True)
    return df_ts 


def windowed_dataset(y, input_window = 5, output_window = 1, stride = 1, num_features = 1):
  
    '''
    create a windowed dataset
    
    : param y:                time series feature (array)
    : param input_window:     number of y samples to give model 
    : param output_window:    number of future y samples to predict  
    : param stide:            spacing between windows   
    : param num_features:     number of features (i.e., 1 for us, but we could have multiple features)
    : return X, Y:            arrays with correct dimensions for LSTM
    :                         (i.e., [input/output window size # examples, # features])
    '''
  
    L = y.shape[0]
    num_samples = (L - input_window - output_window) // stride + 1

    X = np.zeros([input_window, num_samples, num_features])
    Y = np.zeros([output_window, num_samples, num_features])    
    
    for ff in np.arange(num_features):
        for ii in np.arange(num_samples):
            start_x = stride * ii
            end_x = start_x + input_window
            X[:, ii, ff] = y[start_x:end_x, ff]

            start_y = stride * ii + input_window
            end_y = start_y + output_window 
            Y[:, ii, ff] = y[start_y:end_y, ff]

    return X, Y


def numpy_to_torch(x):
    return torch.from_numpy(x).type(torch.Tensor)

