import random 
import os
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from functools import partial
import random
import os 
import torch.nn.functional as F
from typing import List

from utils import set_seed

set_seed(7)

from Modules.models.EEGPT_mcae import EEGTransformer

from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
from utils_eval import get_metrics

use_channels_names = [      
               'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
                'O1', 'O2' ]

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

class LitEEGPTCausal(pl.LightningModule):
    """
    Assess the performance of EEGPT on BCIC2A dataset
    by linear probing. (the weights of EEGPT encoder are frozen)

    There are 22 channels in the dataset, the model only uses
    the channels specified in `use_channels_names`. (19 channels)

    Linear head size:
    - First linear probe: (embed_num * embed_dim) -> num_patches
    (32,768 parameters in this case)
    - Second linear probe: num_patches * num_patches -> num_classes
    (1,024 parameters in this case)

    Attributes
    ----------
    chans_num : int
        Number of channels used in the model.
    target_encoder : EEGTransformer
        The transformer encoder model for EEG data.
    chans_id : torch.Tensor
        Tensor containing the channel IDs for the transformer encoder.
    chan_conv : Conv1dWithConstraint
        Convolutional layer to match the number of channels
        of the input data to the number of channels used in the model.
    linear_probe1 : LinearWithConstraint
        First linear probe layer applied to reduce feature dimension
        (embed_num * embed_dim) of each patch.
    linear_probe2 : LinearWithConstraint
        Second linear probe layer to map the output to the number of classes.
    drop : torch.nn.Dropout
        Dropout layer to prevent overfitting.
        Applied before the linear prediction heads.
    loss_fn : torch.nn.CrossEntropyLoss
        Loss function used for training.
    running_scores : dict
        Dictionary to store running scores for different phases
        (train, valid, test).
    is_sanity : bool
        Flag to indicate if the model is in the sanity check phase.
        Used to skip validation during the initial sanity check.
    """

    def __init__(
            self, load_path: str="../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt",
            input_length: int=1024,
            input_channels: int=22,
            channels_to_use: List[str]=use_channels_names,
            num_classes: int=4,
            embed_num: int=4,
            embed_dim: int=512,
            patch_size: int=64,
            EEGPT_pretrain_settings: dict=EEGPT_pretrain_settings
        ):
        super().__init__()    
        self.chans_num = len(channels_to_use)
        # init model
        num_patches = (input_length - patch_size) // patch_size + 1

        target_encoder = EEGTransformer(
            img_size=[self.chans_num, input_length],
            patch_size=patch_size,
            embed_num=embed_num,
            embed_dim=embed_dim,
            **EEGPT_pretrain_settings
        )
            
        self.target_encoder = target_encoder
        self.chans_id       = target_encoder.prepare_chan_ids(channels_to_use)
        
        # -- load checkpoint
        pretrain_ckpt = torch.load(load_path)
        
        target_encoder_stat = {}
        for k,v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]]=v
                
        self.target_encoder.load_state_dict(target_encoder_stat)
        self.chan_conv       = Conv1dWithConstraint(
            input_channels, self.chans_num, 1, max_norm=1)
        self.linear_probe1   =   LinearWithConstraint(
            embed_num*embed_dim, num_patches, max_norm=1)
        self.linear_probe2   =   LinearWithConstraint(
            num_patches*num_patches, num_classes, max_norm=0.25)
        
        self.drop           = torch.nn.Dropout(p=0.50)
        
        self.loss_fn        = torch.nn.CrossEntropyLoss()
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity=True

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (B, C, T).
            B - batch size
            C - number of channels,
            T - number of time steps.
        
        Returns
        -------
        x : torch.Tensor
            The input data after channel-wise convolution.
            Shape: (B, chans_num, T),
            where chans_num is the number of channels used in the model.
        h : torch.Tensor
            The output logits after the linear probes.
            Shape: (B, num_classes).
        """
        
        x = self.chan_conv(x)  # (B, C, T)

        self.target_encoder.eval()  # frozen weights

        z = self.target_encoder(x, self.chans_id.to(x))  # (C, num_patches, embed_num, embed_dim)
        
        h = z.flatten(2)   # (B, num_patches, embed_num * embed_dim)
        
        h = self.linear_probe1(self.drop(h))    # (B, num_patches, num_patches)
        
        h = h.flatten(1)     # (B, num_patches*num_patches)
        
        h = self.linear_probe2(h)  # (B, num_classes)
        
        return x, h

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y = F.one_hot(y.long(), num_classes=4).float()
        
        label = y
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==torch.argmax(label, dim=-1))*1.0).mean()
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False)
        self.log('data_max', x.max(), on_epoch=True, on_step=False)
        self.log('data_min', x.min(), on_epoch=True, on_step=False)
        self.log('data_std', x.std(), on_epoch=True, on_step=False)
        
        return loss
    
    
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()
    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()
            
        label, y_score = [], []
        for x,y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        print(label.shape, y_score.shape)
        
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, False)
        
        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False)
        
        self.running_scores["valid"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))
        return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            list(self.chan_conv.parameters())+
            list(self.linear_probe1.parameters())+
            list(self.linear_probe2.parameters()),
            weight_decay=0.01)#
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'val_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
      
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )
        
# load configs
# -- LOSO 

# load configs
from utils import *
data_path = "../datasets/downstream/Data/BCIC_2a_0_38HZ"
import math
# used seed: 7
set_seed(8)
for i in range(1,10):
    all_subjects = [i]
    all_datas = []
    train_dataset,valid_dataset,test_dataset = get_data(i,data_path,1,is_few_EA = True, target_sample=1024)
    
    global max_epochs
    global steps_per_epoch
    global max_lr

    batch_size=64

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, num_workers=0, shuffle=False)
    
    max_epochs = 100
    steps_per_epoch = math.ceil(len(train_loader) )
    max_lr = 4e-4

    # init model
    model = LitEEGPTCausal()

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor]
    
    trainer = pl.Trainer(accelerator='cuda',
                         devices=[0,], 
                         max_epochs=max_epochs, 
                         callbacks=callbacks,
                         enable_checkpointing=False,
                         logger=[pl_loggers.TensorBoardLogger('./logs/', name="EEGPT_BCIC2A_tb", version=f"subject{i}"), 
                                 pl_loggers.CSVLogger('./logs/', name="EEGPT_BCIC2A_csv")])

    trainer.fit(model, train_loader, test_loader, ckpt_path='last')