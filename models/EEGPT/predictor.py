import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import math
from typing import Tuple, Any

from .transformer_layers import TransformerBlock as Block
from .embedding_layers import RotaryEmbedding


class TransformerPredictor(nn.Module):
    """ EEG Transformer """
    def __init__(
        self,
        num_patches: Tuple[int, int],
        embed_dim: int=768,
        embed_num: int=1,
        use_pos_embed: bool=False,
        use_inp_embed: bool=True,
        use_part_pred: bool=False,
        predictor_embed_dim: int=384,
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
        num_patches : Tuple[int, int]
            Number of patches in the input data.
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
        predictor_embed_dim : int, optional
            Dimension of the predictor embedding,
            by default 384.
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
        self.use_part_pred = use_part_pred
        self.use_pos_embed = use_pos_embed
        self.use_inp_embed = use_inp_embed
        self.num_patches = num_patches
        self.embed_num = embed_num

        if use_inp_embed:
            self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        if use_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, 1, embed_num, predictor_embed_dim)
            )
            trunc_normal_(self.pos_embed, std=init_std)

        # all-zero token used for querying
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, embed_num, predictor_embed_dim))

        # ----- Define Model layers ------
        self.time_embed_dim = (predictor_embed_dim // num_heads) // 2
        self.time_embed = RotaryEmbedding(
            dim=self.time_embed_dim, interpolate_factor=interpolate_factor)

        # transformer layers
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, is_causal=False, use_rope=True, 
                return_attention=(i+1)==return_attention_layer
            )
            for i in range(depth)]
        )

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # ------ Initialise weights ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
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

    def forward(self, x, mask_x=None, mask_t=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, mN, eN, D), where:
            B - batch size,
            mN - number of temporal windows not masked,
            eN - number of tokens from encoder,
            D - embedding dimension.
        mask_x : torch.Tensor, optional
            The mask used to select patches in the encoder.
            Each value should be in the range [0, N*C-1],
            with format (n*C+c), where n is the temporal window index
            and c is the channel index.
            Shape: (mN_x, mC_x)
            mN_x - number of temporal windows to keep in the input,
            mC_x - number of channels to keep in the input.
        mask_t : torch.Tensor, optional
            Temporal indices (in the range [0, N-1])
            of the temporal windows to predict.
            Shape (mN_y, ),
            mN_y - number of temporal windows to predict.

        Returns
        -------
        x : torch.Tensor
            Contains the prediction of all tokens.
            Shape (B, N, eN, D), where:
            B - batch size,
            N - number of temporal windows,
            eN - number of tokens from encoder,
            D - embedding dimension.
        cmb_x : torch.Tensor, optional
            Skipped tokens from the input,
            only predicted masked tokens are included,
            unmasked tokens are taken from the input.
            Shape (B, N, eN, D).
            Only available if use_part_pred is True.
        """
        if self.use_part_pred:
            inp_x = x

        if self.use_inp_embed:
            x = self.predictor_embed(x)

        C, N = self.num_patches
        B, _, eN, D    = x.shape

        print('Shape of predictor input ', x.shape)

        # -- frequencies for rotary embedding
        freqs = self.time_embed.prepare_freqs(
            (eN, N), x.device, x.dtype)   # (N, time_dim)

        if mask_x is not None:
            # extract the indices of the temporal windows to keep
            mask_x = torch.floor(mask_x[:, 0] / C).long()

            if mask_t is None:
                # default: predict all masked temporal windows
                mask_t = torch.tensor(
                    list(set(list(range(0, N))) - set(mask_x.tolist()))
                ).long()

            # Prepare query tokens (all-zeros)
            N_y = mask_t.shape[0]
            y = self.mask_token.repeat((B, N_y, 1, 1))

            x = torch.cat([x, y], dim=1)

            # Rearrange to the original temporal order
            mask_id = torch.concat(
                [mask_x.to(x.device), mask_t.to(x.device)], dim=0)            
            x = torch.index_select(x, dim=1, index=torch.argsort(mask_id))    
            
        if self.use_pos_embed:
            x = x + self.pos_embed.repeat(
                (B, x.shape[1], 1, 1)).to(x.device)
            
        B, N, eN, D    = x.shape
        x = x.flatten(1, 2)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x, freqs)   # (B, N*eN, D)
            if blk.return_attention == True: return x
  
        x = x.reshape((B, N, eN, D))

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)

        # replace unmasked tokens with the input
        if self.use_part_pred and mask_x is not None:
            cmb_x = torch.index_select(
                x, dim=1, index=mask_t.to(x.device)
            )
            cmb_x = torch.concat([inp_x, cmb_x], dim=1)
            cmb_x = torch.index_select(
                cmb_x, dim=1, index=torch.argsort(mask_id)
            )

            return x, cmb_x

        return x
