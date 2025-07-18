from typing import Any, Dict, Tuple
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from functools import partial
import random
import copy

from .schedulers import CosineWDSchedule
from .utils import grad_logger, apply_mask
from .configs import PRETRAIN_CHANNELS, PRETRAIN_LENGTH
from .encoder import TransformerEncoder
from .predictor import TransformerPredictor
from .reconstructor import TransformerReconstructor


class EEGPTPretrain(pl.LightningModule):
    """
    A PyTorch Lightning module for pretraining the EEGGPT model.

    The model is trained with 2 losses:
    - reconstruction loss (MSE) for the reconstructed patches of EEG signal.
    - alignment loss (MSE) for the encoded representations from the target encoder
    and the main encoder.

    The alignment loss ensures the learnt representation
    from masked input is similar to the representations
    obtained from the original input. This encourages
    learning of more general semantics. 

    Attributes
    ----------
    encoder : EEGTransformer
        The encoder part of the model.
    target_encoder : EEGTransformer
        The momentum encoder mentioned in the paper,
        which is a copy of the encoder with frozen parameters.
        Updates of encoder in the training are propagated to the target encoder
        with a momentum update.
    predictor : EEGTransformerPredictor
        The predictor part of the model.
    reconstructor : EEGTransformerReconstructor
        The reconstructor part of the model.
    chans_id : torch.Tensor
        Tensor containing channel IDs for the model,
        will be used for channel embedding in the encoder.
    loss_fn : torch.nn.MSELoss
        The loss function used for training and validation.
    USE_LOSS_A : bool
        Whether to use the alignment loss on the encoder.
    USE_LN : bool
        Whether to use Layer Normalisation in the model.
    USE_SKIP : bool
        Whether to use skip connections in the predictor.
    optimiser_configs : Dict[str, Any]
        Configuration for the optimizer, such as learning rate.
    momentum_scheduler : Generator
        A generator for momentum values used to update the target encoder.
    wd_scheduler : CosineWDSchedule
        A weight decay scheduler for the optimizer.
    """

    def __init__(
            self,
            models_configs: dict,
            USE_LOSS_A: bool=True,
            USE_LN: bool=True,
            USE_SKIP: bool=True,
            lr_scheduler_name: str='OneCycleLR',
            optimiser_configs: Dict[str, Any] = {},
        ):
        """
        Parameters
        ----------
        models_configs : Dict
            The configuration for the model,
            containing parameters for encoder, predictor, and reconstructor.
            It should be an output of `configs.get_config` function.
        USE_LOSS_A : bool, optional
            Whether to use the loss from the encoder (contrastive loss).
            The default is True.
        USE_LN : bool, optional
            Whether to use Layer Normalisation in the model.
            The default is True.
        USE_SKIP : bool, optional
            Whether to use skip connections in the predictor.
            The reconstructor will use the output of the
            encoder if this is set to False,
            otherwise it uses the output of the predictor.
            The default is True.
        lr_scheduler_name : str, optional
            The type of learning rate scheduler to use.
        optimiser_configs : Dict[str, Any], optional
            Configuration for the optimizer, such as learning rate.
            The default is an empty dictionary.
        """
        super().__init__()    
        self.USE_LOSS_A = USE_LOSS_A
        self.USE_LN     = USE_LN
        self.USE_SKIP   = USE_SKIP
        self.optimiser_configs = optimiser_configs
        self.lr_scheduler_name = lr_scheduler_name

        n_channels = len(PRETRAIN_CHANNELS)

        encoder = TransformerEncoder(
            img_size=[n_channels, PRETRAIN_LENGTH],
            patch_size=32*2,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['encoder']
        )
        
        predictor = TransformerPredictor(
            num_patches=encoder.num_patches,
            use_part_pred=True,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['predictor']
        )
        
        reconstructor = TransformerReconstructor(
            num_patches=encoder.num_patches,
            patch_size=32*2,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['reconstructor']
        )

        target_encoder = copy.deepcopy(encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False
            
        self.encoder        = encoder
        self.target_encoder = target_encoder
        self.predictor      = predictor
        self.reconstructor  = reconstructor
        self.chans_id       = encoder.prepare_chan_ids(PRETRAIN_CHANNELS)
        
        self.loss_fn        = torch.nn.MSELoss()

    def make_masks(
            self,
            num_patchs: Tuple[int, int],
            mC_x: int=12,
            p_n_y: float=0.5,
            p_c_y: float=0.2
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create masks for the input data.
        For each patch, a random
        number of channels is masked.

        Parameters
        ----------
        num_patchs : Tuple[int, int]
            A tuple with 2 integers containing the number of pathces
            along the channel and temporal axes.
        mC_x : int, optional
            The number of channels to mask in the input.
            The default is 12.
        p_n_y : float, optional
            Probability of masking a channel in the target.
            The default is 0.5.
        p_c_y : float, optional
            Probability of masking a channel in the target
            after the initial masking.
            The default is 0.2.

        Returns
        -------
        mask_x : torch.Tensor
            A tensor containing the indices of the patches
            to keep for the input.
            (Indices should be in the range [0, N*C-1])
            Shape: (mN_x, mC_x)
            mN_x - number of temporal windows to keep in the input,
            mC_x - number of channels to keep in the input.
            So mT_x * mC_x patches are kept in the input.
        mask_y : torch.Tensor
            A tensor containing the indices of the patches
            on which to compute the reconstruction loss.
            Shape: (mP_y,),
            mP_y - number of patches to predict.
        """

        C, N = num_patchs

        while True:
            mask_x = []
            mask_y = []
            mask_y_bx = []

            for i in range(N):
                c_idx = torch.randperm(C) + i * C

                if random.random() > p_n_y:
                    mask_x.append(c_idx[:mC_x])
                    mask_y_bx.append(c_idx[mC_x:])
                else:
                    mask_y.append(c_idx)

            if len(mask_x)==0: continue
            if len(mask_y_bx)==0: continue

            mask_y_bx = torch.cat(mask_y_bx, dim=0)
            mask_y_bx = mask_y_bx[torch.rand(mask_y_bx.shape) < p_c_y]

            if len(mask_y_bx)==0: continue

            break
        
        return torch.stack(mask_x, dim=0), torch.cat(mask_y + [mask_y_bx], dim=0)

    def forward_target(
            self, x: torch.Tensor, mask_y: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the momentum encoder,
        and prepare masked patches.
        The outputs are used as ground truth.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, T),
            B - batch size,
            C - number of channels,
            T - number of patches.
        mask_y : torch.Tensor
            Mask tensor indicating which patches to predict,
            with shape (mP_y),
            mP_y - number of patches to predict.

        Returns
        -------
        h : torch.Tensor
            Encoded representation of the input tensor.
            Shape: (B, N, T, D)
            N - number of temporal windows,
            T - number of tokens obtained from the encoder,
            D - feature dimension (dimension of the embedding).
        y : torch.Tensor
            Masked patches of the input tensor,
            containing only the patches specified by the mask.
            Shape: (B, mP_y, bC*bT),
            bC - number of channels in each patch,
            bN - number of timepoints in each patch.
            If `USE_LN` is True, the tensor is normalised.
        """
        with torch.no_grad():
            h = self.target_encoder(x, self.chans_id.to(x))
            h = F.layer_norm(h, (h.size(-1),))  # normalise over feature-dim

            # -------- Masking the patches --------
            C, N = self.encoder.num_patches

            if x.shape[-1] % N != 0 or x.shape[-2] % C != 0:
                raise ValueError(
                    f"Input tensor shape {x.shape} is not compatible with "
                    f"the number of patches C={C} and N={N}."
                    "The last two dimensions of the input tensor "
                    "must be divisible by C and N respectively."
                )

            block_size_c, block_size_n = x.shape[-2] // C, x.shape[-1] // N
            x = x.view(x.shape[0], C, block_size_c, N, block_size_n)
            # Re-arrange dimensions to use `apply_mask`
            x = x.permute(0, 3, 1, 2, 4).contiguous()
            x = x.view(x.shape[0], C, N, block_size_c * block_size_n)   # (B, C, N, bC*bT)
            y = apply_mask(mask_y.to(x.device), x)

            if self.USE_LN:
                y = F.layer_norm(y, (y.size(-1),))

            return h, y

    def forward_context(
            self,
            x: torch.Tensor,
            mask_x: torch.Tensor,
            mask_y: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the main encoder
        to reconstruct masked patches.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, T),
            B - batch size,
            C - number of channels,
            T - number of patches.
        mask_x : torch.Tensor
            Mask tensor indicating which patches to keep,
            with shape (mN_x, mC_x),
            mN_x - number of patches to keep in the input,
            mC_x - number of channels to keep in the input.
        mask_y : torch.Tensor
            Mask tensor indicating which patches to predict,
            with shape (mP_y,),
            mP_y - number of patches to predict.

        Returns
        -------
        z : torch.Tensor
            Encoded representation of the masked input tensor.
        r : torch.Tensor
            Reconstructed patches of the input tensor,
            containing only the patches specified by the mask_y.
            Shape: (B, mP_y, bC*bT),
            bC - number of channels in each patch,
            bN - number of timepoints in each patch.
        """
        z = self.encoder(
            x, self.chans_id.to(x), mask_x=mask_x)

        z, comb_z = self.predictor(z, mask_x=mask_x)

        if not self.USE_SKIP:
            comb_z = z

        r = self.reconstructor(
            comb_z, self.chans_id.to(x), mask_y=mask_y
        )

        return z, r
    
    def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
        ) -> torch.Tensor:
        """
        Validation step for the model.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            A batch of data containing input tensors and labels.
            But only the input tensors are used in the validation step.
            (self-supervised learning)
        batch_idx : int
            The index of the batch in the current epoch.
            A placeholder for the PyTorch Lightning API,

        Returns
        -------
        loss : torch.Tensor
            The computed loss for the batch.
        """
        x, _ = batch
        mask_x, mask_y = self.make_masks(self.encoder.num_patches)
        h, y = self.forward_target(x, mask_y)
        z, r = self.forward_context(x, mask_x, mask_y)

        loss_recon = self.loss_fn(y, r)

        if self.USE_LOSS_A:
            loss_align = self.loss_fn(h, z)

            loss  = loss_align + loss_recon
        else:
            loss  = loss_recon

        if self.USE_LOSS_A:
            self.log(
                'valid_loss_align', loss_align,
                on_epoch=True, on_step=False, sync_dist=True
            )
            self.log(
                'valid_loss_recon', loss_recon,
                on_epoch=True, on_step=False, sync_dist=True
            )

        self.log(
            'valid_loss', loss,
            on_epoch=True, on_step=False, sync_dist=True
        )
                
        return loss
    
    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
        ) -> torch.Tensor:
        """
        Training step for the model.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            A batch of data containing input tensors and labels.
            But only the input tensors are used in the training step.
            (self-supervised learning)
        batch_idx : int
            The index of the batch in the current epoch.
            A placeholder for the PyTorch Lightning API,

        Returns
        -------
        loss : torch.Tensor
            The computed loss for the batch.
            This loss is a combination of the alignment loss
            and the reconstruction loss.
        """
        x, _ = batch
        mask_x, mask_y = self.make_masks(self.encoder.num_patches)
        h, y = self.forward_target(x, mask_y)
        z, r = self.forward_context(x, mask_x, mask_y)

        loss_recon = self.loss_fn(y, r)
        if self.USE_LOSS_A:
            loss_align = self.loss_fn(h, z)
            loss  = loss_align + loss_recon
        else:
            loss  = loss_recon
        
        # Logging the losses (across all GPUs)
        if self.USE_LOSS_A:
            self.log(
                'train_alignmengt_loss', loss_align,
                on_epoch=True, on_step=False, sync_dist=True
            )
            self.log(
                'train_loss_recon', loss_recon,
                on_epoch=True, on_step=False, sync_dist=True
            )

        self.log(
            'train_loss', loss,
            on_epoch=True, on_step=False, sync_dist=True
        )
                
        return loss
    
    def on_train_batch_start(
            self, batch: Any, batch_idx: int
        ) -> None:
        """
        Called at the start of each training batch.
        Updates the weight decay scheduler.

        Parameters
        ----------
        batch : Any
            The input batch of data.
        batch_idx : int
            The index of the batch in the current epoch.
        """
        self.wd_scheduler.step()
        
        return super().on_train_batch_start(batch, batch_idx)
    
    def on_train_batch_end(
            self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
        ) -> None:
        """
        Called at the end of each training batch.
        It updates the weights of target encoder
        using momentum update.

        Parameters
        ----------
        outputs : STEP_OUTPUT
            The output of the training step.
            This is usually the loss value.
        batch : Any
            The input batch of data.
        batch_idx : int
            The index of the batch in the current epoch.
        """
        grad_stats = grad_logger(self.encoder.named_parameters())
        self.log(
            'grad_stats.first_layer', grad_stats.first_layer,
            on_epoch=True, on_step=False, sync_dist=True
        )
        self.log(
            'grad_stats.last_layer', grad_stats.last_layer,
            on_epoch=True, on_step=False, sync_dist=True
        )
        self.log(
            'grad_stats.min', grad_stats.min,
            on_epoch=True, on_step=False, sync_dist=True
        )
        self.log(
            'grad_stats.max', grad_stats.max,
            on_epoch=True, on_step=False, sync_dist=True
        )
        
        # momentum update of target encoder
        with torch.no_grad():
            m = next(self.momentum_scheduler)  # momentum
            for param_q, param_k in zip(
                self.encoder.parameters(), self.target_encoder.parameters()
            ):
                param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)
                
        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    
    def on_load_checkpoint(
            self, checkpoint: Dict[str, Any]
        ) -> None:
        """
        Called when a checkpoint is loaded.
        Recofigure the optimizers.
        """
        res = super().on_load_checkpoint(checkpoint)

        self.configure_optimizers()
        return res
    
    def configure_optimizers(self):
        """
        Configure the following optimizers for the model:
        - AdamW for the encoder, predictor, and reconstructor,
        - OneCycleLR learning rate scheduler,
        - CosineWDSchedule for the weight decay scheduler,
        - Momentum scheduler for the target encoder (linear increase)
        """

        param_groups = [
            {
                'params': (p for n, p in self.encoder.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.predictor.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.reconstructor.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.encoder.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': (p for n, p in self.predictor.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': (p for n, p in self.reconstructor.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }
        ]
        
        optimizer = torch.optim.AdamW(param_groups, lr=6e-5)

        max_lr = self.optimiser_configs.get('max_lr', 5e-4)
        self.optimiser_configs.pop('max_lr', None)
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr,
            div_factor = 2,
            final_div_factor=8,
            pct_start = 0.2,
            **self.optimiser_configs
        )
        # lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.)

        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': 'step',  # Whether to update every 'step' or 'epoch'
            'frequency': 1,   # How often to update the learning rate
            'monitor': 'valid_loss',   # validation loss
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None,   # Custom name for `LearningRateMonitor` to use
        }

        steps_per_epoch = self.optimiser_configs.get('steps_per_epoch', 1000)
        max_epochs = self.optimiser_configs.get('max_epochs', 100)

        # -- Weight Decay Scheduler
        self.wd_scheduler = CosineWDSchedule(
            optimizer,
            ref_wd=1e-6,
            final_wd=1e-6,
            T_max=int(max_epochs * steps_per_epoch)
        )

        # -- Momentum Scheduler for updating the target encoder
        ema = [0.996, 1.0]  # Momentum values for the target encoder
        self.momentum_scheduler = (
            ema[0] + i * (ema[1] - ema[0]) / (steps_per_epoch * max_epochs)
            for i in range(int(steps_per_epoch * max_epochs) + 1)
        )

        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        ) 
