import torch
import torch.nn as nn
import math
from torch.nn.init import trunc_normal_
from typing import Optional, Any

from .configs import CHANNEL_DICT
from .transformer_layers import TransformerBlock as Block
from .embedding_layers import RotaryEmbedding
from .utils import apply_mask, apply_mask_t


class TransformerReconstructor(nn.Module):
    """
    Designed to reconstruct EEG signals
    """
    def __init__(
        self,
        num_patches,
        patch_size: int=64,
        embed_dim: int=768,
        embed_num: int=1,
        use_pos_embed: bool=False,
        use_inp_embed: bool=True,
        reconstructor_embed_dim: int=384,
        depth: int=6,
        num_heads: int=12,
        mlp_ratio: float=4.0,
        qkv_bias: bool=True,
        drop_rate: float=0.0,
        attn_drop_rate: float=0.0,
        drop_path_rate: float=0.0,
        norm_layer: nn.Module=nn.LayerNorm,
        init_std: float=0.02,
        interpolate_factor: float=2.,
        return_attention_layer: int=-1,
        **kwargs: Any
    ):
        """
        Parameters
        ----------
        num_patches : tuple
            A tuple (C, N) where C is the number of channels
            and N is the number of temporal windows.
        patch_size : int, optional
            The size of the patch to reconstruct.
            corresponds to number of time points in each time window. \n
            Default is 64.
        embed_dim : int, optional
            Dimension of the input embedding,
            by default 768.
            Must be the same as the encoder embed_dim.
        embed_num : int, optional
            Number of input tokens, by default 1.
            Must be the same as the encoder embed_num.
        use_pos_embed : bool, optional
            Whether to use positional embedding,
            by default False.
        use_inp_embed : bool, optional
            Whether to use input embedding,
            by default True.
        use_part_pred : bool, optional
            Whether to use partial prediction,
            (only take the prediction of masked tokens),
            by default False.
        reconstructor_embed_dim : int, optional
            Dimension of the predictor embedding,
            by default 384. \n
        depth : int, optional
            Number of transformer blocks,
            by default 6.
        num_heads : int, optional
            Number of attention heads,
            by default 12.
        mlp_ratio : float, optional
            Ratio of the MLP hidden dimension to the embedding dimension,
            by default 4.0.
        qkv_bias : bool, optional
            Whether to use bias in the QKV linear layers,
            by default True.
        drop_rate : float, optional
            Dropout rate for the MLP layers,
            by default 0.0.
        attn_drop_rate : float, optional
            Dropout rate for the attention layers,
            by default 0.0.
        drop_path_rate : float, optional
            Stochastic depth rate for the transformer blocks,
            by default 0.0.
        norm_layer : nn.Module, optional
            Normalization layer to use,
            by default nn.LayerNorm.
        init_std : float, optional
            Standard deviation for the weight initialization,
            by default 0.02.
        interpolate_factor : float, optional
            Factor to interpolate the rotary embedding frequencies,
            by default 2.0.
        return_attention_layer : int, optional
            Layer index to return attention weights from,
            by default -1 (returns the last layer).
            If set to -1, no attention weights are returned.
        **kwargs : dict, optional
            Placeholder.
        """
        super().__init__()
        self.use_inp_embed = use_inp_embed
        self.use_pos_embed = use_pos_embed
        self.num_patches = num_patches

        if use_inp_embed:
            self.reconstructor_embed = nn.Linear(
                embed_dim, reconstructor_embed_dim, bias=True
            )
        
        if use_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, 1, embed_num, reconstructor_embed_dim)
            )
            trunc_normal_(self.pos_embed, std=init_std)

        # all-zero token for the query
        self.mask_token = nn.Parameter(torch.zeros(1, 1, reconstructor_embed_dim))

        # ---------- Define model layers ----------
        self.time_embed_dim = (reconstructor_embed_dim // num_heads) // 2
        self.time_embed = RotaryEmbedding(
            dim=self.time_embed_dim, interpolate_factor=interpolate_factor)
        self.chan_embed = nn.Embedding(
            len(CHANNEL_DICT), reconstructor_embed_dim
        )

        # Transformer blocks
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.reconstructor_blocks = nn.ModuleList([
            Block(
                dim=reconstructor_embed_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer,
                is_causal=False, use_rope=True, 
                return_attention=(i+1)==return_attention_layer)
            for i in range(depth)]
        )

        self.reconstructor_norm = norm_layer(reconstructor_embed_dim)
        self.reconstructor_proj = nn.Linear(
            reconstructor_embed_dim, patch_size, bias=True
        )

        # -------- Initialise weights --------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()
        

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.reconstructor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
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
            chan_ids: torch.Tensor,
            mask_x: Optional[torch.Tensor]=None,
            mask_y: Optional[torch.Tensor]=None
        ):
        """
        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape [B, N, C, D].
            B - batch size
            N - number of temporal windows
            C - number of channels
            D - feature dimension (dimension of the embedding)
        chan_ids : torch.Tensor
            A tensor of shape [C,] containing channel ids.
        mask_x : torch.Tensor, optional
            The mask used to select temporal windows
            for the input x for the encoder.
            (Indices must be in the range of 0 to N*C-1,
            with the form n*C+c)
            Shape: [mN, mC]
            mN - number of temporal windows to keep
            mC - number of channels to keep.
        mask_y : torch.Tensor, optional
            A tensor of shape [mP] containing indices of
            patches to reconstruct.

        Returns
        -------
        torch.Tensor
            A tensor of shape [B, N_y, bN*bC] containing the 
            reconstructed patches.
            N_y - number of patches to reconstruct,
            bN - number of temporal windows in each patch,
            bC - number of channels in each patch.
        """
        print('Shape of reconstructor input', x.shape)

        chan_ids = chan_ids.to(x).long()

        # map from encoder dimension to pedictor dimension
        if self.use_inp_embed:
            x = self.reconstructor_embed(x)

        C, N = self.num_patches
        B, _, eN, _= x.shape

        # add channels positional embedding to x
        chan_embed = self.chan_embed(chan_ids).unsqueeze(0) # (1,C) -> (1,1,C,D)
        
        if mask_x is not None:
            mask_x = mask_x.to(x.device)
            # select the temporal windows of the first channel
            mask_x = torch.floor(mask_x[:,0] / C).long().to(x.device)

            freqs_x = self.time_embed.prepare_freqs((1, N), x.device, x.dtype)
            freqs_x = freqs_x.contiguous().view((1, N, self.time_embed_dim))
            freqs_x = apply_mask_t(mask_x, freqs_x)   # (1, mN, 1, D)
            freqs_x = freqs_x.contiguous().view(
                (mask_x.shape[0], 1, self.time_embed_dim)) # mN, D // 2
            freqs_x = freqs_x.repeat((1, eN, 1)).flatten(0,1)
        else:
            freqs_x = self.time_embed.prepare_freqs(
                (eN, N), x.device, x.dtype)  # (N*C, time_dim)

        if mask_y is not None:
            mask_y       = mask_y.to(x.device)

            # create query mask_token ys
            N_y = mask_y.shape[0]
            chan_embed = chan_embed.repeat((1,N,1,1))
            chan_embed = apply_mask(mask_y, chan_embed)

            freqs = self.time_embed.prepare_freqs(
                (C, N), x.device, x.dtype
            )     # (N*C, time_dim)
            freqs_y = freqs.contiguous().view((1, N, C, self.time_embed_dim))
            freqs_y = apply_mask(mask_y, freqs_y)     # (1, mN, mC, time_dim)
            freqs_y = freqs_y.contiguous().view((N_y, self.time_embed_dim))

            y = self.mask_token.repeat((B, N_y, 1)) + chan_embed
            
            
            if self.use_pos_embed:
                x = x + self.pos_embed.repeat((B, x.shape[1], 1, 1)).to(x.device)

            # concat query mask_token ys
            x = x.flatten(1,2) # B N E D -> B NE D
            x = torch.cat([x,y], dim=1)
            freqs_x = torch.cat([freqs_x, freqs_y], dim=0).to(x)

        # transformer blocks
        for blk in self.reconstructor_blocks:
            x = blk(x, freqs_x)     # (B, N*C, D)
            if blk.return_attention==True: return x

        x = x[:, -N_y:, :]      # (B, N_y, D)

        x = self.reconstructor_norm(x) 

        x = self.reconstructor_proj(x)
        
        return x