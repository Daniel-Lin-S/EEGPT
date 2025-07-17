import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import List, Tuple
from abc import ABC, abstractmethod

from Modules.models.EEGPT_mcae import EEGTransformer
from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
from utils_eval import get_metrics


class Classifier(pl.LightningModule, ABC):
    """
    A base class of PyTorch Lightning module for EEG classification tasks.
    This class provides a structure for training, validation, and testing.
    Optimiser: AdamW with weight decay.
    Learning rate scheduler: OneCycleLR with warm-up.

    Methods
    -------
    forward(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        Forward pass of the model.
        Must be defined by subclasses.
    """

    def __init__(
            self,
            input_length: int,
            input_channels: int,
            num_classes: int,
            max_lr: float,
            scheduler_kwargs: dict = {}
    ):
        """
        Parameters
        ----------
        input_length : int
            Number of time points of the input EEG signal.
        input_channels : int
            Number of input channels in the EEG data.
        num_classes : int
            Number of output classes for classification.
        max_lr : float
            Maximum learning rate for the optimiser.
        scheduler_kwargs : dict, optional
            Additional keyword arguments for the learning rate scheduler.
            Default is an empty dictionary.
        """
        super().__init__()

        self.max_lr = max_lr
        self.num_classes = num_classes
        self.input_length = input_length
        self.input_channels = input_channels
        self.scheduler_kwargs = scheduler_kwargs
        self.loss_fn        = torch.nn.CrossEntropyLoss()
        self.running_scores = {"train":[], "valid":[], "test":[]}

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model.
        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (B, C, T).
            B - batch size, C - number of channels,
            T - number of time steps.

        Returns
        -------
        torch.Tensor
            The processed input. Used if the model has a specific
            preprocessing step before the forward pass.
            If not, simply return the input `x`.
        torch.Tensor
            The output logits after the model's forward pass.
            Shape: (B, num_classes).
        """
        pass

    def training_step(
            self, batch: Tuple[torch.Tensor], batch_idx: int
        ) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : Tuple[torch.Tensor]
            A tuple containing the input data and labels.
            The input data shape is (B, C, T),
            where B is the batch size, C is the number of channels,
            and T is the number of time steps.
        batch_idx : int
            The index of the current batch during training.
            A placeholder for PyTorch Lightning.

        Returns
        -------
        loss : torch.Tensor
            The computed loss for the current batch.
            Used for backpropagation and optimisation.
        """
        x, y = batch
        y = F.one_hot(y.long(), num_classes=self.num_classes).float()
        
        label = y
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((
            torch.argmax(logit, dim=-1) == torch.argmax(label, dim=-1)) * 1.0
        ).mean()

        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False)
        self.log('data_max', x.max(), on_epoch=True, on_step=False)
        self.log('data_min', x.min(), on_epoch=True, on_step=False)
        self.log('data_std', x.std(), on_epoch=True, on_step=False)
        
        return loss

    def on_validation_epoch_start(self) -> None:
        """
        Called at the start of the validation epoch.
        This method resets the running scores for validation.
        """
        self.running_scores["valid"] = []
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of the validation epoch.
        This method computes and logs the validation metrics.

        If the model is in the sanity check phase, it skips
        the validation metrics computation and resets the flag.
        """
        if self.trainer.sanity_checking:
            return super().on_validation_epoch_end()

        label, y_score = [], []
        for x, y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        
        metrics = [
            "accuracy", "balanced_accuracy", "cohen_kappa",
            "f1_weighted", "f1_macro", "f1_micro"]
        results = get_metrics(
            y_score.cpu().numpy(), label.cpu().numpy(), metrics, False)
        
        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)

        return super().on_validation_epoch_end()

    def validation_step(
            self, batch: Tuple[torch.Tensor], batch_idx: int
        ) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : Tuple[torch.Tensor]
            A tuple containing the input data and labels.
            The input data shape is (B, C, T),
            where B is the batch size, C is the number of channels,
            and T is the number of time steps.
        batch_idx : int
            The index of the current batch during validation.
            A placeholder for PyTorch Lightning.
        """
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)

        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        
        self.running_scores["valid"].append(
            (label.clone().detach().cpu(), logit.clone().detach().cpu()))
        return loss

    def on_test_epoch_start(self) -> None:
        """
        Called at the start of the test epoch.
        This method resets the running scores for testing.
        """
        self.running_scores["test"] = []
        return super().on_test_epoch_start()

    def on_test_epoch_end(self) -> None:
        """
        Called at the end of the test epoch.
        This method computes and logs the test metrics.
        """
        label, y_score = [], []
        for x, y in self.running_scores["test"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)

        metrics = [
            "accuracy", "balanced_accuracy", "cohen_kappa",
            "f1_weighted", "f1_macro", "f1_micro"]
        results = get_metrics(
            y_score.cpu().numpy(), label.cpu().numpy(), metrics, False)

        for key, value in results.items():
            self.log(
                'test_'+key, value, on_epoch=True,
                on_step=False, sync_dist=True
            )

        return super().on_test_epoch_end()

    def test_step(
            self, batch: Tuple[torch.Tensor], batch_idx: int
        ) -> None:
        """
        Parameters
        ----------
        batch : Tuple[torch.Tensor]
            A tuple containing the input data and labels.
            The input data shape is (B, C, T),
            where B is the batch size, C is the number of channels,
            and T is the number of time steps.
        batch_idx : int
            The index of the current batch during testing.
            A placeholder for PyTorch Lightning.
        """
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)

        self.running_scores["test"].append(
            (label.clone().detach().cpu(), logit.clone().detach().cpu())
        )


    @abstractmethod
    def prepare_optimizer(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer.
        This method must be implemented by subclasses.

        Returns
        -------
        torch.optim.Optimizer
            An instance of a PyTorch optimizer.
            Typically, an AdamW optimizer with weight decay.
        """
        pass

    def configure_optimizers(self) -> Tuple[dict]:
        """
        Configures the optimizer and learning rate scheduler for training.

        Returns
        -------
        Tuple[dict]
            A tuple containing the optimizer and learning rate scheduler.
            The optimizer is an AdamW optimizer with weight decay.
            The learning rate scheduler is a OneCycleLR scheduler.
        """
        optimizer = self.prepare_optimizer()

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.max_lr,
            pct_start=0.2,  # 20% warm up
            **self.scheduler_kwargs
        )

        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True, # crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor`
        }
      
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )


class BaselineLinearClassifier(Classifier):
    def __init__(
            self, input_length: int,
            input_channels: int,
            num_classes: int, 
            max_lr: float,
            scheduler_kwargs: dict = {},
            dropout: float = 0.5,
            hidden_dim: int = 128
        ):
        """
        """
        super().__init__(
            input_length, input_channels, num_classes,
            max_lr, scheduler_kwargs
        )

        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(
            input_length * input_channels, hidden_dim)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear2 = torch.nn.Linear(
            hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (B, C, T).
            B - batch size, C - number of channels,
            T - number of time steps.

        Returns
        -------
        torch.Tensor
            The output logits after the linear layers.
            Shape: (B, num_classes).
        """
        x1 = self.flatten(x)
        x1 = self.linear1(x1)
        x1 = self.dropout(x1)
        logits = self.linear2(x1)

        return x, logits
    

    def prepare_optimizer(self):
        return torch.optim.AdamW(
            list(self.linear1.parameters()) +
            list(self.linear2.parameters()),
            weight_decay=0.01
        )
    

class EEGPTLinearClassifier(Classifier):
    """
    A PyTorch Lightning module for linear probing
    on EEG data using a pre-trained EEGPT model.

    Linear head size:
    - First linear probe: (embed_num * embed_dim) -> num_patches
    - Second linear probe: num_patches * num_patches -> num_classes

    Attributes
    ----------
    chans_num : int
        Number of channels used in the model.
    max_lr : float
        Maximum learning rate for the optimiser.
    scheduler_kwargs : dict
        Additional keyword arguments for the learning rate scheduler.
    target_encoder : EEGTransformer
        The transformer encoder model for EEG data.
    chans_id : torch.Tensor
        Tensor containing the channel IDs for the transformer encoder.
    chan_conv : Conv1dWithConstraint
        Convolutional layer to match the number of channels
        of the input data to the number of channels used in the model. \n
        With a kernel size of 1, it essentially applies the
        same linear transformation to each time point for consistency.
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
    """

    def __init__(
            self,
            load_path: str,
            input_length: int,
            input_channels: int,
            channels_to_use: List[str],
            num_classes: int,
            EEGPT_pretrain_settings: dict,
            embed_num: int=4,
            patch_size: int=64,
            max_lr: float=4e-4,
            scheduler_kwargs: dict={},
            freeze_backbone: bool=True
        ):
        """
        Parameters
        ----------
        load_path : str
            Path to the pre-trained EEGPT model checkpoint.
        input_length : int
            Length of the input EEG signal.
        input_channels : int
            Number of input channels in the EEG data.
        channels_to_use : List[str]
            List of channel names to use to map to the
            EEGPT channel embedding space.
        num_classes : int
            Number of output classes for classification.
        EEGPT_pretrain_settings : dict
            Dictionary containing the pre-training settings
            for the EEGPT model, such as depth, embed_dim.
        embed_num : int, optional
            Number of embedding tokens to take from
            the EEGPT model. Default is 4.
        patch_size : int, optional
            Size of the patches to be extracted from the input data.
            Default is 64.
        max_lr : float, optional
            Maximum learning rate for the optimiser.
            Default is 4e-4.
        scheduler_kwargs : dict, optional
            Additional keyword arguments for the learning rate scheduler.
            Default is an empty dictionary.
        """
        super().__init__(
            input_length, input_channels, num_classes, max_lr, scheduler_kwargs
        )

        self.chans_num = len(channels_to_use)

        # init model
        num_patches = (input_length - patch_size) // patch_size + 1

        target_encoder = EEGTransformer(
            img_size=[self.chans_num, input_length],
            patch_size=patch_size,
            embed_num=embed_num,
            **EEGPT_pretrain_settings
        )
            
        self.target_encoder = target_encoder
        self.freeze_backbone = freeze_backbone
        self.chans_id       = target_encoder.prepare_chan_ids(channels_to_use)
        
        # -- load checkpoint and freeze the model
        pretrain_ckpt = torch.load(load_path, weights_only=False)
        
        target_encoder_stat = {}
        for k, v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]]=v

        self.target_encoder.load_state_dict(target_encoder_stat)

        if self.freeze_backbone:
            self.target_encoder.eval()  # frozen weights

            for param in self.target_encoder.parameters():
                param.requires_grad = False

        # define the linear layers
        embed_dim = EEGPT_pretrain_settings['embed_dim']
        self.chan_conv       = Conv1dWithConstraint(
            input_channels, self.chans_num, 1, max_norm=1)
        self.linear_probe1   =   LinearWithConstraint(
            embed_num*embed_dim, num_patches, max_norm=1)
        self.linear_probe2   =   LinearWithConstraint(
            num_patches*num_patches, num_classes, max_norm=0.25)
        
        self.drop           = torch.nn.Dropout(p=0.50)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        x = self.chan_conv(x)  # shape: (B, C, T)

        # freeze weights
        if self.freeze_backbone:
            self.target_encoder.eval()  

        z = self.target_encoder(x, self.chans_id.to(x))
        # shape: (B, num_patches, embed_num, embed_dim)
        
        h = z.flatten(2)
        # shape: (B, num_patches, embed_num * embed_dim)
        
        h = self.linear_probe1(self.drop(h))  # (B, num_patches, num_patches)
        
        h = h.flatten(1)   # (B, num_patches*num_patches)
        
        h = self.linear_probe2(h)  # (B, num_classes)
        
        return x, h

    def prepare_optimizer(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for the EEGPT linear classifier.

        Returns
        -------
        torch.optim.Optimizer
            An instance of AdamW optimizer with weight decay.
        """
        return torch.optim.AdamW(
            list(self.chan_conv.parameters()) +
            list(self.linear_probe1.parameters()) +
            list(self.linear_probe2.parameters()),
            weight_decay=0.01
        )
