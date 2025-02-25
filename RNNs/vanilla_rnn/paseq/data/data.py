import numpy as np

import torch
from torch.nn.functional import one_hot

########################################################################################

# take the first M elements in the training dataset 
def load_data_cut(datapath, M, batch_size=1000, val_ratio=0.2):
    x = np.loadtxt(datapath)[:M]
    x = torch.from_numpy(x)  
    idcs = torch.randperm(x.shape[0]) 
    x = one_hot(x.long(), 2) 

    ########################################################################################

    D = len(x) 
    xtrain = x[int(val_ratio * D) :]
    xval = x[: int(val_ratio * D)] 

    ########################################################################################

    train_ds = torch.utils.data.TensorDataset(xtrain) 
    train_loader = torch.utils.data.DataLoader(  
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  
    ) 

    val_ds = torch.utils.data.TensorDataset(xval)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return train_loader, val_loader