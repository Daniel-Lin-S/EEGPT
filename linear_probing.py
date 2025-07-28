"""
Use linear probing (pre-trained backbone frozen + linear prediction head)
to evaluate the performance of backbone model.
A separate model will be trained on each subject of the dataset.
A held-out validation set is used for early stopping and
the results are reported on a held-out test set.

Results will be saved in {log_dir}/{model_name}_{dataset}_tb/ and
{log_dir}/{model_name}_{dataset}_csv/
- tb is a TensorBoard logger for visualisation
- csv is a CSV logger for saving training logs

To view the tensorboard, please run 
```bash
tensorboard --logdir logs/
```

Replace logs with the path to your log directory.
"""


import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    LearningRateMonitor, ModelCheckpoint, EarlyStopping
)
import math
import argparse
from functools import partial
import os

from models.classifiers import (
    EEGPTLinearClassifier, BaselineLinearClassifier
)
from utils import seed_torch
from utils.dataset_utils import get_data_BCIC, get_data_PhysioP300


# to load the public version of EEGPT, these parameters CANNOT be changed
EEGPT_pretrain_settings = {
    'depth' : 8,
    'embed_dim': 512,
    'num_heads': 8,
    'mlp_ratio': 4.0,
    'drop_rate': 0.0,
    'attn_drop_rate': 0.0,
    'drop_path_rate': 0.0,
    'init_std': 0.02,
    'qkv_bias': True,
    'norm_layer' : partial(nn.LayerNorm, eps=1e-6)
}

dataset_choices = ['BCIC2A', 'BCIC2B', 'PhysioP300']
backbone_choices = ['EEGPT', 'None']


def parse_args():
    parser = argparse.ArgumentParser(description="Train EEGPT on BCICIV-2B dataset.")
    # ------ I/O parameters ------
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to the dataset folder containing mat files "
        "preprocessed by `Data_process/process_function.py`."
    )
    parser.add_argument(
        "--backbone_checkpoint_path", type=str,
        required=False, default=None,
        help="Path to the pre-trained backbone model checkpoint. (.ckpt)"
        "Must be provided if using a backbone model."
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs/",
        help="Directory to save logs and checkpoints."
    )
    parser.add_argument(
        '--save_checkpoint', action='store_true',
        help="Whether to save the model checkpoint during training."
    )
    parser.add_argument(
        '--progress_bar', action='store_true',
        help="Whether to show the progress bar during training."
    )
    # ------ Dataset Parameters ------
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=dataset_choices,
        help="Dataset to use for training. "
    )
    parser.add_argument(
        "--n_input_channels", type=int, required=True,
        help="Number of input channels in the EEG data."
    )
    parser.add_argument(
        "--n_classes", type=int, required=True,
        help="Number of classes in the classification task."
    )
    parser.add_argument(
        "--n_subjects", type=int, default=9,
        help="Number of subjects in the dataset."
    )
    parser.add_argument(
        "--eegpt_channel_names", type=str, nargs='+',
        required=False, default=None,
        help="List of channel names to use for the EEGPT backbone. "
        "Must match the channels used in the pre-trained model. "
        "If not provided, EEGPT will use all channels."
    )
    # ------ Pre-processing parameters ------
    parser.add_argument(
        "--target_length", type=int, default=1024,
        help="Target length of the input sequences. (Interpolation used)"
    )
    parser.add_argument(
        "--apply_EA", action="store_true",
        help= "Use Euclidean Alignment (EA) to normalise the data."
    )
    # ------ Training parameters ------
    parser.add_argument(
        "--backbone", type=str, default="EEGPT",
        choices=backbone_choices,
        help="Backbone model to use for linear probing. "
    )
    parser.add_argument(
        "--seed", type=int, default=8,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for training."
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of workers for data loading. "
        "Set to 0 for no multiprocessing."
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100,
        help="Maximum number of training epochs."
    )
    parser.add_argument(
        "--max_lr", type=float, default=1e-3,
        help="Maximum learning rate allowed for the optimizer (OneCycleLR policy)."
    )
    parser.add_argument(
        "--start_lr", type=float, default=1e-4,
        help="Minimum learning rate allowed for the optimizer (OneCycleLR policy)."
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-6,
        help="Minimum learning rate allowed for the optimizer (OneCycleLR policy)."
    )
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA device index to use.")
    parser.add_argument(
        '--restore_checkpoint', action='store_true',
        help="Whether to restore the model from a checkpoint."
        "Must have checkpoints saved in previous runs."
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Patience for early stopping. "
        "If validation loss does not improve for this many epochs, training will stop."
    )

    return parser.parse_args()


args = parse_args()
seed_torch(args.seed)

if args.backbone == "EEGPT":
    model_name = 'EEGPT'
elif args.backbone == "None":
    model_name = 'baseline'

print(
    '---------- Training {} on {} for each subject ---------'.format(
        model_name, args.dataset
    ), flush=True
)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

for i in range(1, args.n_subjects+1):
    if "BCIC" in args.dataset:
        train_dataset, valid_dataset, test_dataset = get_data_BCIC(
            i, args.data_path,
            is_few_EA = args.apply_EA,
            target_length=args.target_length
        )
    elif args.dataset == "PhysioP300":
        train_dataset, valid_dataset, test_dataset = get_data_PhysioP300(
            i, args.data_path
        )
    else:
        raise ValueError(
            f"Unknown dataset: {args.dataset}. "
            f"Allowed choices: {dataset_choices}"
        )

    total_samples = len(train_dataset) + len(valid_dataset) + len(test_dataset)

    print('Prepared {} samples for subject {}'.format(
        total_samples, i
    ), flush=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False
    )

    # init model
    if args.backbone == "None":
        model = BaselineLinearClassifier(
            input_length=args.target_length,
            input_channels=args.n_input_channels,
            num_classes=args.n_classes,
            max_lr=args.max_lr,
            scheduler_kwargs={
                'steps_per_epoch': math.ceil(len(train_dataset)),
                'epochs': args.max_epochs,
                'div_factor': args.max_lr / args.start_lr,
                'final_div_factor': args.max_lr / args.min_lr,
            }
        )
        model_name = 'baseline'
    elif args.backbone == "EEGPT":
        if args.backbone_checkpoint_path is None:
            raise ValueError(
                "Please provide a valid path to the pre-trained backbone model."
            )
        model = EEGPTLinearClassifier(
            load_path=args.backbone_checkpoint_path,
            input_length=args.target_length,
            input_channels=args.n_input_channels,
            channels_to_use=args.eegpt_channel_names,
            num_classes=args.n_classes,
            EEGPT_pretrain_settings=EEGPT_pretrain_settings,
            max_lr = args.max_lr,
            scheduler_kwargs={
                'steps_per_epoch': math.ceil(len(train_dataset)),
                'epochs': args.max_epochs,
                'div_factor' : args.max_lr / args.start_lr,
                'final_div_factor': args.max_lr / args.min_lr,
            }
        )
        model_name = 'EEGPT'
    else:
        raise ValueError(
            f"Unknown backbone: {args.backbone}"
            f"Allowed choices: {backbone_choices}"
        )

    lr_monitor = LearningRateMonitor(
        logging_interval='step')

    early_stopping = EarlyStopping(
        monitor="valid_loss",
        patience=args.patience,
        mode="min",    # minimising the monitored metric
        verbose=True
    )

    callbacks = [lr_monitor, early_stopping]

    use_checkpoint = args.save_checkpoint or args.restore_checkpoint
    if use_checkpoint:
        checkpoint = ModelCheckpoint(
            dirpath=f'checkpoint/linear_probe_{model_name}_{args.dataset}/',
            save_last=True,
            monitor='valid_loss',
            mode='min',
            save_top_k=1,
            filename='{epoch:02d}-{step}-{val_loss:.2f}'
        )

        callbacks.append(checkpoint)

    trainer = pl.Trainer(
        accelerator='cuda',
        precision='16-mixed',
        max_epochs=args.max_epochs, 
        callbacks=callbacks,
        enable_checkpointing=use_checkpoint,
        enable_progress_bar=args.progress_bar,
        logger=[
            pl_loggers.TensorBoardLogger(
                args.log_dir, name=f"{model_name}_{args.dataset}_tb",
                version=f"subject{i}"
            ), 
            pl_loggers.CSVLogger(
                args.log_dir, name=f"{model_name}_{args.dataset}_csv",
                version=f"subject{i}"
            )
        ]
    )

    if args.restore_checkpoint:
        ckpt_path = 'best'
    else:
        ckpt_path = None

    trainer.fit(
        model, train_loader, valid_loader,
        ckpt_path=ckpt_path
    )

    print('Training finished for subject', i, flush=True)

    trainer.test(
        model, test_loader, ckpt_path=ckpt_path
    )
