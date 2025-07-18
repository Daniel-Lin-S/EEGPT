# Training in 256Hz data and 4s
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

import argparse
import math

from models.EEGPT import EEGPTPretrain, MODELS_CONFIGS, get_config
from utils import load_fn, seed_torch


torch.set_float32_matmul_precision("medium")
seed_torch(7)

parser = argparse.ArgumentParser(description='EEGPT Pretraining')

# ----- Model Settings -----
parser.add_argument(
    '--model_scale', type=str, default='tiny1',
    choices=list(MODELS_CONFIGS.keys()),
    help='Model scale to use for pretraining. '
    'Could be one of: ' + ', '.join(MODELS_CONFIGS.keys())
)
parser.add_argument(
    '--variant', type=str, default='D',
    choices=['A', 'B', 'C', 'D'],
    help='Model variant to use for pretraining. '
    'Used for ablation study.'
    'A - not using contrastive loss for the encoder, '
    'B - not using layer normalisation, '
    'C - not using skip connections, '
    'D - using the full settings.'
)
# ----- Training Settings -----
parser.add_argument(
    '--max_epochs', type=int, default=200,
    help='Maximum number of epochs for training.'
)
parser.add_argument(
    '--batch_size', type=int, default=64,
    help='Batch size for training.'
)
parser.add_argument(
    '--devices', type=int, nargs='+', default=[0],
    help='List of GPU devices to use for training. '
    'If empty, will use CPU.'
)
parser.add_argument(
    '--max_lr', type=float, default=5e-4,
    help='Maximum learning rate for the training.'
)


args = parser.parse_args()

# -------- prepare data --------
train_dataset = torchvision.datasets.DatasetFolder(
    root="../datasets/pretrain/merged/TrainFolder/",
    loader=load_fn,  extensions=['.edf'])
valid_dataset = torchvision.datasets.DatasetFolder(
    root="../datasets/pretrain/merged/ValidFolder/",
    loader=load_fn, extensions=['.edf'])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    num_workers=0, shuffle=True)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=args.batch_size,
    num_workers=0, shuffle=False
)

# -------- prepare model and trainer --------
optimiser_configs = {
    'max_lr' : args.max_lr,
    'max_epochs': args.max_epochs,
    'steps_per_epoch': math.ceil(len(train_loader) / len(args.devices)),
}

model = EEGPTPretrain(
    get_config(**(MODELS_CONFIGS[args.model_scale])),
    USE_LOSS_A = (args.variant != "A"),
    USE_LN = (args.variant != "B"),
    USE_SKIP = (args.variant != "C"),
    optimiser_configs=optimiser_configs,
)

if args.variant == 'A':
    ablation = 'no_contrastive_loss'
elif args.variant == 'B':
    ablation = 'no_layer_norm'
elif args.variant == 'C':
    ablation = 'no_skip_connections'
elif args.variant == 'D':
     ablation = 'full'
else:
     raise ValueError(f"Unknown variant: {args.variant}")

lr_monitor = LearningRateMonitor(logging_interval='epoch')
callbacks = [lr_monitor]

trainer = pl.Trainer(
    strategy='auto', devices=args.devices,
    max_epochs=args.max_epochs, callbacks=callbacks,
    logger=[
        pl_loggers.TensorBoardLogger(
            './logs/', name=f"EEGPT_{args.model_scale}_{ablation}_tb"),
        pl_loggers.CSVLogger(
            './logs/', name=f"EEGPT_{args.model_scale}_{ablation}_csv")
        ]
)

# -------- start training --------
trainer.fit(model, train_loader, valid_loader)
