"""
Pre-process BCIC-IV 2a and 2b datasets.

The downloaded datasets must be saved into datasets/downstream/Raw_data,
under folders named BCICIV_2a_gdf and BCICIV_2b_gdf folders respectively.

Use the following scripts to download and unzip the files
(please cd to datasets/downstream/Raw_data first)
```shell
wget https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip
unzip BCICIV_2a_gdf.zip -d BCICIV_2a_gdf
wget https://www.bbci.de/competition/download/competition_iv/BCICIV_2b_gdf.zip
unzip BCICIV_2b_gdf.zip -d BCICIV_2b_gdf
wget -O true_labels_2a.zip https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip
unzip true_labels_2a.zip -d BCICIV_2a_gdf
wget -O true_labels_2b.zip https://www.bbci.de/competition/iv/results/ds2b/true_labels.zip
unzip true_labels_2b.zip -d BCICIV_2b_gdf

# optionally remove the zip files
rm BCICIV_2a_gdf.zip
rm BCICIV_2b_gdf.zip
rm true_labels_2a.zip
rm true_labels_2b.zip
```

The processed data will be saved into
datasets/downstream/Data/BCIC_2a and BCIC_2b folders.

Dataset Information
----------------
Length of each event onset - BCIC 2a: 4 seconds, BCIC 2b: 4 seconds
Recording Frequency: - BCIC 2a: 250 Hz, BCIC 2b: 250 Hz
Number of classes - BCIC 2a: 4, BCIC 2b: 2
Number of subjects - BCIC 2a: 9, BCIC 2b: 9
Number of EEG electrodes: - BCIC 2a: 22, BCIC 2b: 3
"""

import os
import numpy as np
import scipy.io
from typing import List, Optional, Sequence
from braindecode.preprocessing import exponential_moving_standardize
import mne
import mne.io
import scipy.io as sio
import glob


class LoadData:
    def __init__(self, eeg_file_path: str):
        """
        Parameters
        ----------
        eeg_file_path : str
            The path to the EEG data files.
        """
        self.eeg_file_path = eeg_file_path
        self.raw_eeg_subject = None

    def load_raw_data_gdf(self, file_to_load):
        self.raw_eeg_subject = mne.io.read_raw_gdf(
            self.eeg_file_path + '/' + file_to_load)
        return self

    def load_raw_data_mat(self,file_to_load):
        self.raw_eeg_subject = sio.loadmat(self.eeg_file_path + '/' + file_to_load)

    def get_all_files(self, file_path_extension: str = None):
        if file_path_extension:
            return glob.glob(self.eeg_file_path+'/'+file_path_extension)
        return os.listdir(self.eeg_file_path)
    def filter(self,low,high):
        self.raw_eeg_subject.filter(low,high)

class LoadBCIC(LoadData):
    """
    Loade the training data of the BICI IV 2a dataset.

    It only takes the 22 EEG channels, 3 EOG channels are removed.
    """
    def __init__(self, file_to_load: str, *args):
        """
        Initialise the LoadBCIC class.

        Parameters
        ----------
        file_to_load : str
            The name of the file to load, e.g., 'A01T.gdf'.
        *args : tuple
            Additional arguments to be passed to the parent class LoadData.
        """

        # event keys of the cue labels (left, right, foot, tongue)
        self.stimcodes = ['769', '770', '771', '772']
        self.file_to_load = file_to_load
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        self.fs = None
        super(LoadBCIC, self).__init__(*args)

    def get_epochs(
            self, tmin: float=-4.5, tmax: float=5.0,
            bandpass: Optional[Sequence[float]] = None,
            resample: Optional[int] = None,
            baseline: Optional[tuple] = None,
            reject = False
        ):
        """
        Get the EEG signal from events in the raw data,
        together with the corresponding labels and sampling frequency.

        Parameters
        ----------
        tmin : float
            Start time before the event in seconds.
        tmax : float
            End time after the event in seconds.
        bandpass : Sequence[float], optional
            Bandpass filter frequencies in Hz, e.g., (0.5, 100). \n
            Only first 2 items will be used.
            Default is None.
        resample : int, optional
            If given, the data will be resampled to this frequency.
        baseline : tuple, optional
            Baseline correction applied to the epochs,
            must have the form (start, end) in seconds.
            If start or end is None, it will be set to the epoch start or end.
        reject : bool, optional
            If True, events with code 1 will be rejected and annotated as 'bad trial'.
            Default is False.
        """
        if len(bandpass) < 2:
            raise ValueError(
                'bandpass must be a sequence of two floats.'
            )

        file_path = os.path.join(self.eeg_file_path, self.file_to_load)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f'File {file_path} does not exist.'
            )

        raw_data = mne.io.read_raw_gdf(file_path)
        events, event_ids = mne.events_from_annotations(raw_data)
        self.fs = raw_data.info.get('sfreq')
        if reject == True:
            reject_events = mne.pick_events(events,[1])
            reject_oneset = reject_events[:,0]/self.fs
            duration = [4]*len(reject_events)
            descriptions = ['bad trial']*len(reject_events)
            blink_annot = mne.Annotations(reject_oneset, duration, descriptions)
            raw_data.set_annotations(blink_annot)

        # filter the events to only include the ones we are interested in
        stims =[value for key, value in event_ids.items() if key in self.stimcodes]
        epochs = mne.Epochs(
            raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
            baseline=baseline, preload=True, proj=False, reject_by_annotation=True)
        if bandpass is not None:
            epochs.filter(bandpass[0], bandpass[1], method = 'iir')
            # epochs.resample(128)
        if resample is not None:
            epochs.resample(resample)

        epochs = epochs.drop_channels(self.channels_to_remove)
        self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1])
        self.x_data = epochs.get_data()*1e6
        eeg_data = {
            'x_data': self.x_data,
            'y_labels': self.y_labels,
            'fs': self.fs
        }

        return eeg_data
    
class LoadBCIC_E(LoadData):
    """
    A class to load the test data of the BICI IV 2a dataset.
    """
    def __init__(self, file_to_load: str, lable_name: str, *args):
        """
        Paramaters
        ----------
        file_to_load : str
            The name of the gdf file containing the biosignals, e.g., 'A01E.gdf'.
        lable_name : str
            The name of the mat file containing the labels, e.g., 'A01E.mat'.
        *args : tuple
            Arguments to be passed to the parent class LoadData.
        """

        self.stimcodes = ('783')
        # self.epoched_data={}
        self.label_name = lable_name # the path of the test label
        self.file_to_load = file_to_load
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        super(LoadBCIC_E, self).__init__(*args)

    def get_epochs(self, tmin=-4.5, tmax=5.0, bandpass = False,resample = None,baseline=None):
        if len(bandpass) < 2:
            raise ValueError(
                'bandpass must be a sequence of two floats.'
            )

        gdf_file_path = os.path.join(self.eeg_file_path, self.file_to_load)
        if not os.path.exists(gdf_file_path):
            raise FileNotFoundError(
                f'File {gdf_file_path} does not exist.'
            )

        label_file_path = os.path.join(self.eeg_file_path, self.label_name)
        if not os.path.exists(label_file_path):
            raise FileNotFoundError(
                f'File {label_file_path} does not exist.'
            )

        self.load_raw_data_gdf(self.file_to_load)
        raw_data = self.raw_eeg_subject
        self.fs = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)
        stims =[value for key, value in event_ids.items() if key in self.stimcodes]
        
        epochs = mne.Epochs(
            raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
            baseline=baseline, preload=True, proj=False, reject_by_annotation=False)

        if bandpass is not None:
            epochs.filter(bandpass[0],bandpass[1],method = 'iir')
            # epochs.resample(128)
        if resample is not None:
            epochs.resample(resample)

        epochs = epochs.drop_channels(self.channels_to_remove)
        label_info  = sio.loadmat(label_file_path)
        #label_info shape:(288, 1)
        self.y_labels = label_info['classlabel'].reshape(-1) -1
        # print(self.y_labels)
        self.x_data = epochs.get_data()*1e6
        eeg_data={'x_data': self.x_data,
                  'y_labels': self.y_labels,
                  'fs': self.fs}
        return eeg_data
    
class LoadBCIC_2b:
    '''A class to load the test and train data of the BICI IV 2b datast'''
    def __init__(self,path,subject,tmin =0,tmax = 4,bandpass = None):
        self.tmin = tmin
        self.tmax = tmax
        self.bandpass = bandpass
        self.subject = subject
        self.path = path
        self.train_name = ['1','2','3']
        self.test_name = ['4','5']
        self.train_stim_code  = ['769','770']
        self.test_stim_code  = ['783']
        self.channels_to_remove = ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
        
    def get_train_data(self):
        data = []
        label = []
        for se in self.train_name:
            data_name = r'B0{}0{}T.gdf'.format(self.subject,se)
            label_name = r'B0{}0{}T.mat'.format(self.subject,se)
            data_path = os.path.join(self.path,data_name)
            label_path = os.path.join(self.path,label_name)
            data_x = self.get_epoch(data_path,True,self.tmin,self.tmax,self.bandpass)
            data_y = self.get_label(label_path)
            
            data.extend(data_x)
            label.extend(data_y)
        return np.array(data),np.array(label).reshape(-1)
    
    def get_test_data(self):
        data = []
        label = []
        for se in self.test_name:
            data_name = r'B0{}0{}E.gdf'.format(self.subject,se)
            label_name = r'B0{}0{}E.mat'.format(self.subject,se)
            data_path = os.path.join(self.path,data_name)
            label_path = os.path.join(self.path,label_name)
            data_x = self.get_epoch(data_path,False,self.tmin,self.tmax,self.bandpass)
            data_y = self.get_label(label_path)
            
            data.extend(data_x)
            label.extend(data_y)
        return np.array(data),np.array(label).reshape(-1)
            
    
    def get_epoch(self,data_path,isTrain = True,tmin =0,tmax = 4,bandpass = None):
        raw_data = mne.io.read_raw_gdf(data_path)
        events,events_id =  mne.events_from_annotations(raw_data)
        if isTrain:
            stims = [values for key,values in events_id.items() if key in self.train_stim_code]
        else:
            stims = [values for key,values in events_id.items() if key in self.test_stim_code]
        epochs = mne.Epochs(raw_data,events,stims,tmin = tmin,tmax = tmax,event_repeated='drop',baseline=None,preload=True, proj=False, reject_by_annotation=False)

        if bandpass is not None:
            epochs.filter(bandpass[0],bandpass[1],method = 'iir')

        epochs = epochs.drop_channels(self.channels_to_remove)
        eeg_data = epochs.get_data()*1e6
        return eeg_data
    
    def get_label(self, label_path):
        label_info = sio.loadmat(label_path)
        return label_info['classlabel'].reshape(-1)-1

def EMS(data: np.ndarray) -> np.ndarray:
    """
    Apply exponential moving standardisation to the data.
    Z-score by exponential moving average and moving standard deviation.

    Parameters
    ----------
    data : np.ndarray
        The data to be standardised, shape (n_samples, n_channels, n_times).

    Return
    -------
    np.ndarray
        The standardised data, with the same shape as input.
    """
    new_x = []

    for x in data:
        new_x.append(exponential_moving_standardize(x))

    return np.array(new_x)


def Load_BCIC_2a_raw_data(
        tmin: int=0, tmax: int=4,
        bandpass: List[int] = [0,38],
        resample: Optional[int] = None,
        apply_ems: bool = True
    ) -> None:
    """
    Load all the 9 subjects data from BCIC 2a dataset
    and pre-process the data by applying exponential moving standardization (EMS).
    The pre-processed data is then saved in the folder './Data/BCIC_2a'
    as mat files with the following structure:
    - sub{ID}_train/Data.mat
        - x_data: (n_samples, n_channels, n_times)
        - y_data: (n_samples,)
    - sub{ID}_test/Data.mat
        - x_data: (n_samples, n_channels, n_times)
        - y_data: (n_samples,)
    If bandpass is provided, the name of the folder will be changed to 
    './Data/BCIC_2a_{bandpass[0]}_{bandpass[1]}HZ'.

    Parameters
    ----------
    tmin : int, optional
        The start time of the epoch, by default 0.
    tmax : int, optional
        The end time of the epoch, by default 4.
    bandpass : List[int], optional
        The bandpass filter range, by default [0, 38].
    resample : int, optional
        If given, the data will be resampled to this frequency.
    apply_ems : bool, optional
        If True, apply exponential moving standardisation to the data,
        by default True.
    """
    # prepare IO
    data_path = os.path.join(root_path, 'BCICIV_2a_gdf')

    if bandpass is None:
        SAVE_path = os.path.join(root_path, 'Data', 'BCIC_2a')
    else:
        SAVE_path = os.path.join(root_path, 'Data', 'BCIC_2a_{}_{}HZ'.format(
            bandpass[0],bandpass[1]))

    if not os.path.exists(SAVE_path):
        os.makedirs(SAVE_path)

    # load data
    for sub in range(1, 10):
        print('------ Processing subject {} ------'.format(sub))
        data_name = r'A0{}T.gdf'.format(sub)
        data_loader = LoadBCIC(data_name, data_path)
        data = data_loader.get_epochs(
            tmin=tmin, tmax=tmax,bandpass = bandpass,resample = resample)
        train_x = np.array(data['x_data'])[:, :, :]
        train_y = np.array(data['y_labels']).reshape(-1)
        
        data_name = r'A0{}E.gdf'.format(sub)
        label_name = r'A0{}E.mat'.format(sub)
        data_loader = LoadBCIC_E(data_name, label_name, data_path)

        data = data_loader.get_epochs(
            tmin=tmin, tmax=tmax,bandpass = bandpass,resample = resample)
        test_x = np.array(data['x_data'])[:, :, :]
        test_y = np.array(data['y_labels']).reshape(-1)

        if apply_ems:
            train_x = EMS(train_x)
            test_x = EMS(test_x)
 
        print('Shape of training samples: ', train_x.shape)
        print('Shape of training labels:', train_y.shape)
        print('Shape of test samples:',test_x.shape)
        print('Shape of test labels:',test_y.shape)
            
        SAVE_test = os.path.join(SAVE_path, r'sub{}_test'.format(sub))
        SAVE_train = os.path.join(SAVE_path, 'sub{}_train'.format(sub))
        
        if not os.path.exists(SAVE_test):
            os.makedirs(SAVE_test)
        if not os.path.exists(SAVE_train):
            os.makedirs(SAVE_train)
            
        scipy.io.savemat(
            os.path.join(SAVE_train, "Data.mat"), {'x_data': train_x,'y_data': train_y})
        scipy.io.savemat(
            os.path.join(SAVE_test, "Data.mat"), {'x_data': test_x, 'y_data': test_y})
        
    print('Successfully saved all the data to {}'.format(SAVE_path))


def Load_BCIC_2b_raw_data(
    tmin: int=0, tmax: int=4,
    bandpass: List[int] = [0,38],
    use_ems: bool = True
) -> None:
    """
    Load all the 9 subjects data from BCIC 2b dataset,
    and save the data in the folder './Data/BCIC_2b'
    as mat files with the following structure:
    - sub{ID}_train/Data.mat
        - x_data: (n_samples, n_channels, n_times)
        - y_data: (n_samples,)
    - sub{ID}_test/Data.mat
        - x_data: (n_samples, n_channels, n_times)
        - y_data: (n_samples,)
    If bandpass is provided, the name of the folder will be changed to
    './Data/BCIC_2b_{bandpass[0]}_{bandpass[1]}HZ'.

    Parameters
    ----------
    tmin : int, optional
        The start time of each epoch, by default 0.
        This is relative to the stimulus onset.
    tmax : int, optional
        The end time of each epoch, by default 4.
        This is relative to the stimulus onset.
    bandpass : List[int], optional
        The bandpass filter range, by default [0, 38].
        If None, no bandpass filter will be applied.
    use_ems : bool, optional
        If True, apply exponential moving standardisation to the data,
        by default True.
    """
    
    data_path = os.path.join(root_path, 'BCICIV_2b_gdf')

    if bandpass is None:
        save_path = os.path.join(root_path, r'Data', 'BCIC_2b')
    else:
        save_path = os.path.join(
            root_path, r'Data',
            'BCIC_2b_{}_{}HZ'.format(bandpass[0],bandpass[1])
        )

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for sub in range(1, 10):
        print('------ Processing subject {} ------'.format(sub))

        load_raw_data = LoadBCIC_2b(data_path, sub, tmin, tmax, bandpass)
        save_train_path = os.path.join(save_path,r'sub{}_train'.format(sub))
        save_test_path = os.path.join(save_path,r'sub{}_test').format(sub)

        if not os.path.exists(save_train_path):
            os.makedirs(save_train_path)
        if not os.path.exists(save_test_path):
            os.makedirs(save_test_path)

        train_x, train_y = load_raw_data.get_train_data()
        test_x, test_y = load_raw_data.get_test_data()

        if use_ems:
            train_x = EMS(train_x)
            test_x = EMS(test_x)

        print('Shape of training samples: ', train_x.shape)
        print('Shape of training labels:', train_y.shape)
        print('Shape of test samples:', test_x.shape)
        print('Shape of test labels:', test_y.shape)

        scipy.io.savemat(
            os.path.join(save_train_path,'Data.mat'),
            {'x_data':train_x,'y_data':train_y}
        )
        
        scipy.io.savemat(
            os.path.join(save_test_path,'Data.mat'),
            {'x_data':test_x, 'y_data':test_y}
        )
        
    print('Successfully saved all the data to {}'.format(save_path))
    
if __name__ == '__main__':
    root_path = "Raw_data"
    Load_BCIC_2a_raw_data()
    Load_BCIC_2b_raw_data()
