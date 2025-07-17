import torch
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder
import os
import numpy as np
import random
import scipy.io as sio
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import mne
import pandas as pd 
import csv

from Data_process.utils import EA

        
def set_seed(seed: int):
    """
    Set seed for reproducibility.
    This function ensures all aspects of randomness in Python, NumPy, and PyTorch
    are controlled by the same seed value.

    Warning: this function sacrifices some performance for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to set for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)   # CPU
    torch.cuda.manual_seed(seed)   # GPU
    torch.cuda.manual_seed_all(seed)   # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # prevent hash randomisation
    torch.backends.cudnn.deterministic = True  # ensure deterministic behavior of cudnn
    torch.backends.cudnn.benchmark = False # disable cudnn auto-tuner to ensure deterministic results


def temporal_interpolation(
        x: torch.Tensor,
        desired_sequence_length: int,
        mode: str='nearest',
        use_avg: bool=True
    ) -> torch.Tensor:
    """
    Interpolates the input tensor to a desired sequence length.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, C, T) or (C, T), where B is batch size,
        C is number of channels, and T is sequence length.
    desired_sequence_length : int
        The desired sequence length to interpolate to.
    mode : str, optional
        The interpolation mode. Default is 'nearest'.
    use_avg : bool, optional
        Whether to normalise the input tensor by subtracting the mean.
        Default is True.
    """
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)

    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(
            x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError(
            f"Input tensor must have 2 or 3 dimensions, got {len(x.shape)} dimensions."
        )


class EEGDataset(Dataset):
    """
    EEGDataset is a PyTorch Dataset class for handling EEG data.

    Parameters
    ----------
    feature : torch.Tensor
        The EEG features of shape (num_samples, num_channels, num_timesteps).
    label : torch.Tensor
        The labels corresponding to the EEG features of shape (num_samples,).
    subject_id : torch.Tensor, optional
        The subject IDs corresponding to the EEG features of shape (num_samples,).
    """
    def __init__(
            self,
            feature: torch.Tensor,
            label: torch.Tensor,
            subject_id: Optional[int]=None
        ):
        super(EEGDataset,self).__init__()

        self.x = feature
        self.y = label
        self.s = subject_id

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def get_num_class(self, num_class=[1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            label = self.y[i]
            label = int(label)
            if num_class[label]>0:
                num_class[label]-=1
                res[label].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y     
    
    def get_num_subject(self, num_class=[1,1,1,1,1,1,1,1]):
        res = [[] for _ in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            s = self.s[i]
            s = int(s)
            if num_class[s]>0:
                num_class[s]-=1
                res[s].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y


def get_data_PhysioP300(
        sub: int, data_path: str
    ) -> Tuple[DatasetFolder, DatasetFolder, DatasetFolder]:
    """
    Get data for a specific subject from the PhysioNet P300 dataset,
    and split it into training, validation, and testing sets.

    Parameters
    ----------
    sub : int
        Subject number (1-9)
    data_path : str
        Path to the data directory containing the subject data.
        The folder must have .sub{sub} files in the format:
        sub01.sub1, sub01.sub2, etc.

    Returns
    -------
    Tuple[DatasetFolder, DatasetFolder, DatasetFolder]
        A tuple containing torchvision.datasets.DatasetFolder objects
        for training, validation, and testing.
    """

    subject_dir = os.path.join(data_path, f'sub{sub:02d}')

    data = DatasetFolder(
        root=subject_dir,
        loader=torch.load,
        extensions=[f'.sub{sub}']
    )

    temp, test_dataset = train_test_split(
        data, test_size=0.2, stratify=[y for _, y in data]
    )
    train_dataset, valid_dataset = train_test_split(
        temp, test_size=0.1, stratify=[y for _, y in temp]
    )

    return train_dataset, valid_dataset, test_dataset


def get_data_BCIC(
        sub: int,
        data_path: str,
        is_few_EA: bool = False,
        target_length: int=-1,
        use_avg: bool=True,
        use_channels=None
    ) -> Tuple[EEGDataset, EEGDataset, EEGDataset]:
    """
    Get data for a specific subject from the BCICIV-2 datasets (2a and 2b)
    and prepare it for training, validation, and testing.
    Default ratio: 10% data of the training set are used for validation.

    Please use `Data_preprocess.process_function.Load_BCIC_2a_raw_data`
    and `Data_preprocess.process_function.Load_BCIC_2b_raw_data`
    to preprocess the data before using this function.

    Parameters
    ----------
    sub : int
        Subject number (1-9 for 2a, 1-10 for 2b)
    data_path : str
        Path to the data directory containing the subject data.
        The folder must have .mat files in the format:
        sub{sub}_train/Data.mat and sub{sub}_test/Data.mat
    is_few_EA : bool, optional
        Whether to apply the EA (Euclidean Alignment) to the EEG data.
        Default is False.
    target_sample : int, optional
        Target sample length for temporal interpolation.
        If -1, no interpolation is applied. \n
        Default is -1.

    Returns
    -------
    Tuple[EEGDataset, EEGDataset, EEGDataset]
        A tuple containing EEGDataset objects
        for training, validation, and testing.
    """
    
    target_session_1_path = os.path.join(
        data_path,r'sub{}_train/Data.mat'.format(sub))
    target_session_2_path = os.path.join(
        data_path,r'sub{}_test/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    session_2_data = sio.loadmat(target_session_2_path)

    R = None
    if is_few_EA is True:
        session_1_x = EA(session_1_data['x_data'], R)
        session_2_x = EA(session_2_data['x_data'], R)
    else:
        session_1_x = session_1_data['x_data']
        session_2_x = session_2_data['x_data']

    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)

    test_x_2 = torch.FloatTensor(session_2_x)      
    test_y_2 = torch.LongTensor(session_2_data['y_data']).reshape(-1)
    
    if target_length > 0:
        test_x_1 = temporal_interpolation(
            test_x_1, target_length, use_avg=use_avg)
        test_x_2 = temporal_interpolation(
            test_x_2, target_length, use_avg=use_avg)

    if use_channels is not None:
        test_dataset = EEGDataset(
            torch.cat([test_x_1,test_x_2],dim=0)[:,use_channels,:],
            torch.cat([test_y_1,test_y_2],dim=0))
    else:
        test_dataset = EEGDataset(
            torch.cat([test_x_1,test_x_2],dim=0),
            torch.cat([test_y_1,test_y_2],dim=0))

    source_train_x = []
    source_train_y = []
    source_train_s = []
    
    source_valid_x = []
    source_valid_y = []
    source_valid_s = []
    subject_id = 0

    for i in range(1, 10):
        if i == sub:
            continue
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        train_data = sio.loadmat(train_path)
    
        test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(i))
        test_data = sio.loadmat(test_path)
        if is_few_EA is True:
            session_1_x = EA(train_data['x_data'],R)
        else:
            session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)

        train_x, valid_x, train_y, valid_y = train_test_split(
            session_1_x, session_1_y, test_size = 0.1, stratify = session_1_y
        )
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),)) * subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),)) * subject_id)

        if is_few_EA is True:
            session_2_x = EA(test_data['x_data'], R)
        else:
            session_2_x = test_data['x_data']

        session_2_y = test_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(
            session_2_x, session_2_y,
            test_size = 0.1, stratify = session_2_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)
        subject_id += 1

    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    source_train_s = torch.cat(source_train_s, dim=0)

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    source_valid_s = torch.cat(source_valid_s, dim=0)

    if target_length > 0:
        source_train_x = temporal_interpolation(
            source_train_x, target_length, use_avg=use_avg)
        source_valid_x = temporal_interpolation(
            source_valid_x, target_length, use_avg=use_avg)

    if use_channels is not None:
        train_dataset = EEGDataset(
            source_train_x[:,use_channels,:],
            source_train_y,
            source_train_s
        )
    else:
        train_dataset = EEGDataset(
            source_train_x, source_train_y, source_train_s)

    if use_channels is not None:
        valid_datset = EEGDataset(
            source_valid_x[:,use_channels,:],
            source_valid_y,
            source_valid_s
        )
    else:
        valid_datset = EEGDataset(
            source_valid_x, source_valid_y, source_valid_s)
    
    return train_dataset, valid_datset, test_dataset


import random
mne.set_log_level("ERROR")

def min_max_normalize(x: torch.Tensor, data_max=None, data_min=None, low=-1, high=1):
    if data_max is not None:
        max_scale = data_max - data_min
        scale = 2 * (torch.clamp_max((x.max() - x.min()) / max_scale, 1.0) - 0.5)
        
    if len(x.shape) == 2:
        xmin = x.min()
        xmax = x.max()
        if xmax - xmin == 0:
            x = 0
            return x
    elif len(x.shape) == 3:
        xmin = torch.min(torch.min(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        xmax = torch.max(torch.max(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        constant_trials = (xmax - xmin) == 0
        if torch.any(constant_trials):
            # If normalizing multiple trials, stabilize the normalization
            xmax[constant_trials] = xmax[constant_trials] + 1e-6

    x = (x - xmin) / (xmax - xmin)

    # Now all scaled 0 -> 1, remove 0.5 bias
    x -= 0.5
    # Adjust for low/high bias and scale up
    x += (high + low) / 2
    x  = (high - low) * x
    
    if data_max is not None:
        x = torch.cat([x, torch.ones((1, x.shape[-1])).to(x)*scale])
    return x
    
    
use_channels_names=[
               'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
                'O1', 'O2'
    ]

# -- read Kaggle ERN
ch_names_kaggle_ern = list("Fp1,Fp2,AF7,AF3,AF4,AF8,F7,F5,F3,F1,Fz,F2,F4,F6,F8,FT7,FC5,FC3,FC1,FCz,FC2,FC4,FC6,FT8,T7,C5,C3,C1,Cz,C2,C4,C6,T8,TP7,CP5,CP3,CP1,CPz,CP2,CP4,CP6,TP8,P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,POz,PO8,O1,O2".split(','))

def read_csv_epochs(filename, tmin, tlen, use_channels_names=use_channels_names, data_max=None, data_min=None):
    sample_rate = 200
    raw = pd.read_csv(filename)
    
    data = torch.tensor(raw.iloc[:,1:-2].values) # exclude time EOG Feedback
    feed = torch.tensor(raw['FeedBackEvent'].values)
    stim_pos = torch.nonzero(feed>0)
    # print(stim_pos)
    datas = []
    
    # -- get channel id by use chan names
    if use_channels_names is not None:
        choice_channels = []
        for ch in use_channels_names:
            choice_channels.append([x.lower().strip('.') for x in ch_names_kaggle_ern].index(ch.lower()))
        use_channels = choice_channels
    if data_max is not None: use_channels+=[-1]
    
    xform = lambda x: min_max_normalize(x, data_max, data_min)
    
    for fb, pos in enumerate(stim_pos, 1):
        start_i = max(pos + int(sample_rate * tmin), 0)
        end___i = min(start_i + int(sample_rate * tlen), len(feed))
        # print(start_i, end___i)
        trial = data[start_i:end___i, :].clone().detach().cpu().numpy().T
        # print(trial.shape)
        info = mne.create_info(
            ch_names=[str(i) for i in range(trial.shape[0])],
            ch_types="eeg",  # channel type
            sfreq=200,  # frequency
            #
        )
        raw = mne.io.RawArray(trial, info)  # create raw
        # raw = raw.filter(5,40)
        # raw = raw.resample(256)
        
        trial = torch.tensor(raw.get_data()).float()

        trial = xform(trial)
        if use_channels_names is not None:
            trial = trial[use_channels]
        datas.append(trial)
    return datas
    
def read_kaggle_ern_test(
                    path     = "datas/",
                    subjects = [1,3,4,5,8,9,10,15,19,25],#
                    sessions = [1,2,3,4,5],#
                    tmin     = -0.7,
                    tlen     = 2,
                    data_max=None,
                    data_min=None,
                    use_channels_names = use_channels_names,
                    ):
    # -- read labels
    labels = pd.read_csv(os.path.join(path, 'KaggleERN', 'true_labels.csv'))['label']
    
    # -- read datas
    label_id = 0
    datas = []
    for i in tqdm(subjects):
        for j in sessions:
            filename = os.path.join(path, "KaggleERN", "test", "Data_S{:02d}_Sess{:02d}.csv".format(i,j))

            # -- read data
            for data in read_csv_epochs(filename, tmin=tmin, tlen=tlen, data_max=data_max, data_min=data_min, use_channels_names = use_channels_names): 
                label = labels[label_id]
                label_id += 1
                datas.append((data, int(label)))
    return datas

def read_kaggle_ern_train(
    path     = "datas/",
    subjects = [2,6,7,11,12,13,14,16,17,18,20,21,22,23,24,26, ],#
    sessions = [1,2,3,4,5],#
    tmin     = -0.7,
    tlen     = 2,
    data_max=None,
    data_min=None,
    use_channels_names = use_channels_names,
) -> list:
    # -- read labels
    labels = []
    with open(os.path.join(path, 'KaggleERN', 'TrainLabels.csv'), 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if(i>0): labels.append(row)
    labels = dict(labels) # [['S02_Sess01_FB001', '1'],
    
    # -- read datas
    datas = []
    for i in tqdm(subjects):
        for j in sessions:
            if i>9:
                if(i == 22 and j == 5):
                    print("Skipped error file " + "KagglERN/train/Data_S"+ str(i)+"_Sess0"+str(j)+".csv" )
                else:
                    filename = os.path.join(
                        path, "KaggleERN","train","Data_S"+ str(i)+"_Sess0"+str(j)+".csv")
            else:
                filename = os.path.join(
                    path, "KaggleERN", "train", "Data_S0"+ str(i)+"_Sess0"+str(j)+".csv")
            
            # -- read data
            for fb,trial in enumerate(
                read_csv_epochs(
                    filename, tmin=tmin, tlen=tlen,
                    data_max=data_max, data_min=data_min,
                    use_channels_names = use_channels_names
                ), 1):
                label = labels["S{:02d}_Sess{:02d}_FB{:03d}".format(i,j,fb)]
                datas.append((trial, int(label)))

    return datas
