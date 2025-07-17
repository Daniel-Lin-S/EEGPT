import numpy as np
from scipy.linalg import fractional_matrix_power
import os
import scipy.io as sio
from typing import Optional


def train_validation_split(x, y, validation_size, seed = None):
    '''
    Split the training set into a new training set and a validation set
    @author: WenChao Liu
    '''
    if seed:
        np.random.seed(seed)
    label_unique = np.unique(y)
    validation_x = []
    validation_y = []
    train_x = []
    train_y = []
    for label in label_unique:
        index = (y==label)
        label_num = np.sum(index)
        print("class-{}:{}".format(label,label_num))
        class_data_x = x[index]
        class_data_y = y[index]
        rand_order = np.random.permutation(label_num)
        class_data_x,class_data_y = class_data_x[rand_order],class_data_y[rand_order]
        print(class_data_x.shape)
        validation_x.extend(class_data_x[:int(label_num*validation_size)].tolist())
        validation_y.extend(class_data_y[:int(label_num*validation_size)].tolist())
        train_x.extend(class_data_x[int(label_num*validation_size):].tolist())
        train_y.extend(class_data_y[int(label_num*validation_size):].tolist())
    
    validation_x = np.array(validation_x)
    validation_y = np.array(validation_y).reshape(-1)
    
    train_x = np.array(train_x)
    train_y = np.array(train_y).reshape(-1)
    
    print(train_x.shape,train_y.shape)
    print(validation_x.shape,validation_y.shape)
    return train_x,train_y,validation_x,validation_y


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


def few_shot_data(sub,data_path, class_number = 4,shot_number = 1):
    
    sub_path = os.path.join(data_path,'sub{}_train'.format(sub),'Data.mat')
    data = sio.loadmat(sub_path)
    x,y = data['x_data'],data['y_data'].reshape(-1)
    result_x = []
    result_y = []
    for i in range(class_number):
        label_index = (y == i)
        result_x.extend(x[label_index][:shot_number])
        result_y.extend([i]*shot_number)
        
    return np.array(result_x),np.array(result_y)    

