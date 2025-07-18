import torch
import numpy as np
import random
import os


def seed_torch(seed: int=1029):
    """
    Fix all random seeds for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Random seed to use. Default is 1029.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_fn(x: str) -> torch.Tensor:
    """
    Load EDF file and return a tensor of shape (1, 1024, 256)
    with random window of length 4s.

    Parameters
    ----------
    x : str
        Path to the EDF file.
    
    Returns
    -------
    torch.Tensor
        A tensor of shape (1, 1024, 256)
        containing a random window of data.
    """
    x = torch.load(x)
    
    window_length = 4*256  
    data_length = x.shape[1]  

    # Calculate the maximum starting index for the windows
    max_start_index = data_length - window_length

    # Generate random indices
    if max_start_index>0:
        index = random.randint(0, max_start_index)
        x = x[:, index:index+window_length]
    x = x.to(torch.float)

    return x
