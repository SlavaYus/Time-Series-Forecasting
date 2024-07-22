import torch
 
def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X += [feature]
        y += [target]
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X, dtype = torch.float32), torch.tensor(y, dtype = torch.float32)

