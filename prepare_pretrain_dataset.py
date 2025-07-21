"""
Use this script to prepare the mixed pretraining dataset
for EEGPT.

Datasets: PhysioNetMI, TSU, SEED, M3CV.
Please follow the instructions in datasets/pretrian/readme.md
to download the datasets and place them in the correct directories.
"""

import os
import torch
import random
import tqdm
import mne
import pandas as pd
import copy

from torcheeg.datasets import (
    M3CVDataset, TSUBenckmarkDataset, SEEDDataset, CSVFolderDataset
)
from torcheeg import transforms
from torcheeg.datasets.constants import (
    SEED_CHANNEL_LIST, M3CV_CHANNEL_LIST, TSUBENCHMARK_CHANNEL_LIST
)

from models.EEGPT.configs import PRETRAIN_CHANNELS


data_root_path = "datasets/pretrain/io_root/"
file_root_path = "datasets/pretrain/"


PHYSIONETMI_CHANNEL_LIST = [
    'Fc5.', 'Fc3.', 'Fc1.', 
    'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 
    'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 
    'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 
    'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 
    'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.',
    'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 
    'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 
    'Oz..', 'O2..', 'Iz..'
]

PHYSIONETMI_CHANNEL_LIST = [x.strip('.').upper() for x in PHYSIONETMI_CHANNEL_LIST]


def temporal_interpolation(
        x : torch.Tensor,
        desired_sequence_length : int,
        mode: str='nearest'
    ):
    """
    Interpolate the temporal dimension of a tensor to a desired sequence length.
    Note: the input is normalised to have zero mean along the temporal dimension.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape (batch_size, channels, sequence_length)
        or (channels, sequence_length).
    desired_sequence_length : int
        The target length for the temporal dimension.
    mode : str, optional
        The interpolation mode to use. Default is 'nearest'.
        Other options include 'linear', 'bilinear', 'bicubic', etc.
    """
    x = x - x.mean(-2)

    if len(x.shape) == 2:    # without batch dimension
        return torch.nn.functional.interpolate(
            x.unsqueeze(0), desired_sequence_length, mode=mode
        ).squeeze(0)
    elif len(x.shape) == 3:  # with batch dimension
        return torch.nn.functional.interpolate(
            x, desired_sequence_length, mode=mode
        )
    else:
        raise ValueError(
            f"Input tensor must have 2 or 3 dimensions, got {len(x.shape)} dimensions."
        )



def get_physionet_dataset():
    channels_name = PHYSIONETMI_CHANNEL_LIST

    """
    In summary, the experimental runs were:

    1   Baseline, eyes open
    2   Baseline, eyes closed
    3   Task 1 (open and close left or right fist)                  -> 4 5
    4   Task 2 (imagine opening and closing left or right fist)     -> 0 1
    5   Task 3 (open and close both fists or both feet)             -> 6 7
    6   Task 4 (imagine opening and closing both fists or both feet)-> 2 3
    7   Task 1
    8   Task 2
    9   Task 3
    10  Task 4
    11  Task 1
    12  Task 2
    13  Task 3
    14  Task 4

    """
    session_id2task_id = {
        3:1, 4:2, 5:3, 6:4,
        7:1, 8:2, 9:3, 10:4,
        11:1, 12:2, 13:3, 14:4,
        
    }

    task2event_id = {
        0 : dict([('T1', 4), ('T2', 5)]),
        1 : dict([('T1', 0), ('T2', 1)]),
        2 : dict([('T1', 6), ('T2', 7)]),
        3 : dict([('T1', 2), ('T2', 3)])
    }

    
    if not os.path.exists(data_root_path + 'io/PhysioNetMI'):
        src_path = os.path.join(
            file_root_path, "./PhysioNetMI/files/eegmmidb/1.0.0/")
        ls = []
        channels_name = None
        for subject in range(1,110):
            for task in [0,1,2,3]:
                for session in [3,7,11]:
                    session += task
                    file_path = src_path + "S{:03d}".format(subject) + '/' + "S{:03d}R{:02d}.edf".format(subject,session)
                    raw = mne.io.read_raw_edf(file_path,preload=True)
                    
                    if channels_name is None:
                        channels_name = copy.deepcopy(raw.ch_names)
                    else:
                        assert channels_name == raw.ch_names
                        
                    event_id = task2event_id[session_id2task_id[session]-1]
                    # -- split epochs
                    epochs = mne.Epochs(raw, 
                        events = mne.events_from_annotations(
                            raw, event_id=event_id, chunk_duration=None)[0], 
                        tmin=0, tmax=0 + 6 - 1 / raw.info['sfreq'], 
                        preload=True, 
                        decim=1,
                        baseline=None, 
                        reject_by_annotation=False
                    )
                    
                    d = {
                        # "subject_id":[subject],
                        # "sess_id": [session],
                        # "task_id":[task],
                        "file_path":[file_path],
                        "labels":"".join([str(ev[-1]) for ev in epochs.events])
                    }
                    
                    ls.append(pd.DataFrame(d))

        table = pd.concat(ls, ignore_index=True)
        # print(table)
        print(channels_name)
        table.to_csv("./PhysioNetMI/physionetmi_meta.csv", index=False)

        def default_read_fn(
                file_path, task_id=None, session_id=None, subject_id=None, **kwargs
            ) -> mne.Epochs:
            """
            Read a file and return epochs.

            Parameters
            ----------
            file_path : str
                The path to the EDF file.
            task_id : int, optional
                A placeholder for CSVFolderDataset compatibility.
            session_id : int, optional
                A placeholder for CSVFolderDataset compatibility.
            subject_id : int, optional
                A placeholder for CSVFolderDataset compatibility.
            **kwargs : dict
                Additional keyword arguments (not used).
            """
            session_id = int(file_path.split('R')[-1].split('.')[0])

            raw = mne.io.read_raw_edf(file_path, preload=True)
            
            event_id = task2event_id[session_id2task_id[session_id]-1]
            # -- split epochs
            epochs = mne.Epochs(
                raw, 
                events = mne.events_from_annotations(
                    raw, event_id=event_id, chunk_duration=None)[0], 
                tmin=0,
                tmax=0 + 6 - 1 / raw.info['sfreq'], 
                preload=True, 
                decim=1,
                baseline=None, 
                reject_by_annotation=False
            )
            
            return epochs

        dataset = CSVFolderDataset(
            csv_path="./PhysioNetMI/physionetmi_meta.csv",
            read_fn=default_read_fn,
            io_path=data_root_path+'io/PhysioNetMI',
            online_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.To2d()
            ]),
            #    label_transform=transforms.Select('label'),
            num_worker=4
        )
    else:
        dataset = CSVFolderDataset(
            io_path=data_root_path + 'io/PhysioNetMI',
            online_transform=transforms.Compose([
                transforms.PickElectrode(
                    transforms.PickElectrode.to_index_list(
                        PRETRAIN_CHANNELS, PHYSIONETMI_CHANNEL_LIST)
                ),
                transforms.ToTensor(),
                #   transforms.RandomWindowSlice(window_size=160*4, p=1.0),
                transforms.Lambda(
                    lambda x: temporal_interpolation(x, 256 * 6) * 1e3
                ),  # scale from V to mV, resample to 256Hz.
                transforms.To2d()
            ]),
            label_transform=transforms.Compose([
                #   transforms.Select('labels'),
                #   transforms.StringToInt()
                transforms.Lambda(lambda x : 0)
            ])
        )

    return dataset


def get_TSU_dataset():
    src_path = os.path.join(file_root_path, "TSUBenchmark")

    if not os.path.exists(src_path):
        raise FileNotFoundError(
            "TSU Benchmark dataset not found. "
            f"Please ensure the dataset is downloaded to {src_path}."
        )

    dataset = TSUBenckmarkDataset(
        root_path=src_path,
        io_path=data_root_path+'io/tsu_benchmark',
        online_transform=transforms.Compose([
            transforms.PickElectrode(
                transforms.PickElectrode.to_index_list(
                    PRETRAIN_CHANNELS, TSUBENCHMARK_CHANNEL_LIST)
            ),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: temporal_interpolation(x, 256 * 4) / 1000
            ),    # scale from uV to mV, resample to 256Hz.
            transforms.To2d(),
        ]),
        label_transform=transforms.Select('trial_id'))

    return dataset


def get_M3CV_dataset():
    src_path = os.path.join(file_root_path, "M3CV")

    if not os.path.exists(src_path):
        raise FileNotFoundError(
            "TSU Benchmark dataset not found. "
            f"Please ensure the dataset is downloaded to {src_path}."
        )

    dataset = M3CVDataset(
        root_path=src_path,
        io_path=data_root_path+'io/m3cv',
        online_transform=transforms.Compose([
            transforms.PickElectrode(
                transforms.PickElectrode.to_index_list(
                    PRETRAIN_CHANNELS, M3CV_CHANNEL_LIST)
            ),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: temporal_interpolation(x, 256 * 4) / 1000
            ),  # scale from uV to mV, resample to 256Hz.
            transforms.To2d(),
        ]),
        label_transform=transforms.Compose([
            transforms.Select('subject_id'),
            transforms.StringToInt()
        ])
    )
    return dataset


def get_SEED_dataset():
    src_path = os.path.join(file_root_path, "SEED")

    if not os.path.exists(src_path):
        raise FileNotFoundError(
            "SEED dataset not found. "
            f"Please ensure the dataset is downloaded to {src_path}."
        )

    dataset = SEEDDataset(
        root_path=src_path,
        io_path=data_root_path + 'io/seed',
        online_transform=transforms.Compose([
            transforms.PickElectrode(
                transforms.PickElectrode.to_index_list(
                    PRETRAIN_CHANNELS, SEED_CHANNEL_LIST)
            ),
            transforms.ToTensor(),
            #   transforms.RandomWindowSlice(window_size=250*4, p=1.0),
            transforms.Lambda(
                lambda x: temporal_interpolation(x, 256 * 10) / 1000
            ),  # scale from uV to mV, resample to 256Hz.
            transforms.To2d(),                          
        ]),
        label_transform=transforms.Compose([
            transforms.Select('emotion'),
            transforms.Lambda(lambda x: x + 1)
        ])
    )

    return dataset


if __name__ == "__main__":
    for tag in ["PhysioNetMI", "tsu_benchmark", "seed", "m3cv"]:
        print(f"Processing dataset: {tag}")

        if tag == "PhysioNetMI":
            dataset = get_physionet_dataset()
        elif tag == "tsu_benchmark":
            dataset = get_TSU_dataset()
        elif tag == "m3cv":
            dataset = get_M3CV_dataset()
        elif tag == "seed":
            dataset = get_SEED_dataset()
        else:
            raise ValueError("Invalid tag")

        print(len(dataset))
        print('Number of samples: ', len(dataset))
        print("Shape of each sample: ", dataset[0][0].shape)

        # Train-validation split
        for i, (x, y) in tqdm.tqdm(enumerate(dataset)):
            dst = "./merged/"
            if random.random() < 0.1:
                dst += "ValidFolder/0/"
            else:
                dst += "TrainFolder/0/"
            os.makedirs(dst, exist_ok=True)
            data = x.squeeze_(0)

            if len(data.shape) != 2:
                raise ValueError(
                    f"Expected data to have 2 dimensions, got {len(data.shape)}."
                )

            if data.shape[0] != 58:
                raise ValueError(
                    f"Expected data to have 58 channels, got {data.shape[0]}."
                )
            
            if data.shape[1] < 1024:
                raise ValueError(
                    "Expected data to have at least 1024 time points, "
                    f"got {data.shape[1]}."
                )

            torch.save(data, dst + tag + f"_{i}.edf")

            del data, x
