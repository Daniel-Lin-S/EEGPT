import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import math
from torch.nn.init import trunc_normal_

from .configs import CHANNEL_DICT
from .transformer_layers import TransformerBlock as Block
from .embedding_layers import PatchEmbed, PatchNormEmbed
from .utils import apply_mask, apply_mask_t


class TransformerEncoder(nn.Module):
    """
    EEG Transformer Model for EEG data processing.

    To load pre-trained weights:
    >>> model = EEGTransformer()
    >>> model.load_state_dict(torch.load('path_to_weights.pth'))
    >>> model.eval()  # Set to evaluation mode
    The parameters that must be kept the same to the pre-trained version:
    all hyper-parameters related to transformer blocks
    (depth, num_heads, embed_dim, mlp_ratio, qkv_bias,
    drop_rate, attn_drop_rate, drop_path_rate, norm_layer).

    In this model, EEG data is treated as a 2D image,
    and the tokens are the channels of the EEG data.
    """
    def __init__(
        self,
        img_size: Tuple[int]=(64,1000),
        patch_size: int=64,
        patch_stride: Optional[int]=None,
        embed_dim: int=768,
        embed_num: int=1,
        depth: int=12,
        num_heads: int=12,
        mlp_ratio: float=4.0,
        qkv_bias: bool=True,
        drop_rate: float=0.0,
        attn_drop_rate: float=0.0,
        drop_path_rate: float=0.0,
        norm_layer: nn.Module=nn.LayerNorm,
        patch_module: nn.Module=PatchEmbed,
        init_std: float=0.02,
        return_attention_layer: int=-1,
        **kwargs
    ):
        """
        Initialises the EEG Transformer model.

        Parameters
        ----------
        img_size : tuple, optional
            Size of the input image (height, width). Default is (64, 1000).
        patch_size : int, optional
            Size (number of time points) of each patch. Default is 64.
        patch_stride : int, optional
            Stride for patch extraction. If not given, it defaults to patch_size.
        embed_dim : int, optional
            Dimension of the embedding space. Default is 768.
            Each patch is converted into an embedding of this dimension.
        embed_num : int, optional
            Number of embedding tokens. Default is 1.
        depth : int, optional
            Number of transformer blocks in the model. Default is 12.
        num_heads : int, optional
            Number of attention heads in each transformer block. Default is 12.
        mlp_ratio : float, optional
            Ratio of the hidden dimension in the MLP
            to the embedding dimension. Default is 4.0. \n
            e.g. default embed_dim is 768,
            then hidden dimension is 768 * 4 = 3072.
        qkv_bias : bool, optional
            Whether to use bias (constant) term in the linear
            layers producing Q, K, V tensors in the attention mechanism. \n
            Default is True.
        drop_rate : float, optional
            Dropout rate applied to the output of the attention layer. \n
            Default is 0.0.
        drop_path_rate : float, optional
            Drop path rate for stochastic depth regularisation. \n
            Default is 0.0.
        attn_drop_rate : float, optional
            Dropout rate applied to the attention weights. \n
            Default is 0.0.
        norm_layer : nn.Module, optional
            Normalisation layer to be applied after each transformer block. \n
            Default is nn.LayerNorm.
        patch_module : nn.Module, optional
            Module for patch embedding. \n
            Default is PatchEmbed, which extracts patches from the input image.
            One could also use PatchNormEmbed for normalised patch embedding.
            The module must have attribute `num_patches`.
        init_std : float, optional
            Standard deviation of the (truncated) normal distribution
            used for initialising model weights. \n
            Default is 0.02.
        return_attention_layer : int, optional
            Layer index at which to return attention weights. \n
            If -1, no attention weights are returned. Default is -1.
        """
        super().__init__()
        self.img_size = img_size
        self.num_features = self.embed_dim = embed_dim
        self.embed_num = embed_num
        
        self.num_heads = num_heads

        self.patch_embed = patch_module(
            img_size=img_size,
            patch_size=patch_size,
            patch_stride=patch_stride,
            embed_dim=embed_dim
        )

        self.num_patches = self.patch_embed.num_patches

        # a dictionary-like object to map channel ids to embedded vectors
        self.chan_embed = nn.Embedding(len(CHANNEL_DICT), embed_dim)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                is_causal=False, use_rope= False, return_attention=(i+1)==return_attention_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.init_std = init_std

        # empty summary tokens (like [cls] in BERT) to store the output tokens.
        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, embed_dim))

        # Initialise summary token and model weights.
        trunc_normal_(self.summary_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def prepare_chan_ids(
            self, channels: List[str]
        ) -> torch.Tensor:
        """
        Convert channel names to channel ids.

        Parameter
        ----------
        channels : list of str
            List of channel names (e.g., ['C3', 'C4', 'O1', ...]).
        
        Returns
        -------
        torch.Tensor
            Tensor of shape (1, num_channels) containing channel ids.
        
        Raises
        ------
        AssertionError
            If any channel name is not in the predefined CHANNEL_DICT.
        """
        chan_ids = []
        for ch in channels:
            ch = ch.upper().strip('.')
            assert ch in CHANNEL_DICT
            chan_ids.append(CHANNEL_DICT[ch])

        return torch.tensor(chan_ids).unsqueeze_(0).long()

    def fix_init_weight(self) -> None:
        """
        Rescale the weights of attention projection and MLP layers
        in each transformer block to maintain stability during training.
        """
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m: nn.Module):
        """
        Initialise weights of the model.
        Linear layers are initialised with truncated normal distribution,
        layer norms are initialised with constant values,
        convolutional layers are initialised with truncated normal distribution
        with optional bias initialised to zero.
        Embedding layers are initialised with normal distribution.

        Parameters
        ----------
        m : nn.Module
            Module to initialise weights for.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
            self, x: torch.Tensor,
            chan_ids: Optional[torch.Tensor]=None,
            mask_x: Optional[torch.Tensor]=None,
            mask_t: Optional[torch.Tensor]=None
        ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, T)
            where B is batch size,
            C is number of channels,
            and T is time steps.
            C and T must match the img_size when initialising this class.
        chan_ids : torch.Tensor, optional
            Tensor of shape (1, C) containing selected channel ids.
            If given, C should be equal to img_size[0].
            If None, uses all channels, in which case the number of
            channels of the input must match img_size[0].
        mask_x : torch.Tensor, optional
            Tensor of shape (mN, mC) containing indices of patches
            of the patch embeddings to be masked out before transformer blocks.
            mN - number of temporal windows to keep, mN <= N.
            mC - number of channels to keep, mC <= C.
            If None, no masking is applied.
        mask_t : torch.Tensor, optional
            Tensor of shape (mP,) containing indices of patches
            of the output tokens to be masked out.
            Note that mP <= N * C.
            If None, no masking is applied.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, N, embed_num, D)
            B - batch size,
            N - number of patches
            (not be the same as input if mask_x or mask_t are provided),
            embed_num is number of embedding tokens,
            and D is embedding dimension.
        """
        B, C, T = x.shape

        if C != self.img_size[0] or T != self.img_size[1]:
            raise ValueError(
                f"Input shape {x.shape} does not match img_size {self.img_size}."
            )

        x = self.patch_embed(x)  # (B, N, C, D)

        if chan_ids is None:
            chan_ids = torch.arange(0,C)     
        chan_ids = chan_ids.to(x)

        # add positional embedding of channels
        x = x + self.chan_embed(chan_ids.long()).unsqueeze(0) # (1,C) -> (1,1,C,D)

        if mask_x is not None:
            mask_x = mask_x.to(x.device)
            x = apply_mask(mask_x, x)    # (B, mN, mC, D)

        B, N, C, _ = x.shape

        x = x.flatten(0, 1)   # (B * N, C, D)

        # concat summary token
        summary_token = self.summary_token.repeat((x.shape[0], 1, 1))
        x = torch.cat([x, summary_token], dim=1)  # (B * N, C + embed_num, D)
        
        # apply attention blocks
        for blk in self.blocks:
            x = blk(x)
            if blk.return_attention==True: return x

        # prepare output
        x = x[:, -summary_token.shape[1]:, :]   # (B * mN, embed_num, D)

        if self.norm is not None:
            x = self.norm(x)

        x = x.flatten(-2)
        x = x.reshape((B, N, -1))   # (B, mN, embed_num * D)

        if mask_t is not None:
            mask_t = mask_t.to(x.device)
            x = apply_mask_t(mask_t, x)   # (B, mmN, embed_num * D)

        x = x.reshape((B, N, self.embed_num, -1))

        return x
