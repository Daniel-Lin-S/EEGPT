from typing import Tuple, Optional
import os
from torchvision.datasets import DatasetFolder
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from sklearn.model_selection import train_test_split
import random
import numpy as np
from scipy.linalg import fractional_matrix_power


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

def EA(x: np.ndarray, new_R: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Perform the Euclidean Alignment (EA) on the input data.
    It reduces the covariance of the data to the identity matrix
    so that inter-subject and inter-session variability is minimised.

    Parameters
    ----------
    x : numpy.ndarray
        Input data of shape (n_samples, n_channels, n_samples).
    new_R : numpy.ndarray, optional
        New covariance matrix to apply after alignment to identity matrix.
        If None, only the alignment is performed.

    Returns
    -------
    numpy.ndarray
        Transformed data with reduced covariance.
        has the same shape as input x.
    """
    
    xt = np.transpose(x, axes=(0,2,1))
    E = np.matmul(x, xt)   # covariance matrix
    R = np.mean(E, axis=0)  # average covariance matrix across samples

    # apply whitening transformation
    R_mat = fractional_matrix_power(R, -0.5)
    new_x = np.einsum('n c s,r c -> n r s', x, R_mat)

    if new_R is None:
        return new_x

    new_x = np.einsum(
        'n c s,r c -> n r s', new_x,
        fractional_matrix_power(new_R, 0.5))
    
    return new_x


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
